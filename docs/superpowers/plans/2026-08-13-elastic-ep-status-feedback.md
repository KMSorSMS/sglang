---
title: "Elastic EP status feedback implementation plan"
description: "TDD plan for bounded non-blocking Scheduler-to-DPC health feedback and the remaining pinned-snapshot boundary tests."
---

# Elastic EP status feedback implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make Scheduler-to-DPC Elastic EP health feedback bounded and non-blocking, then close the four remaining pinned-snapshot CPU test gaps.

**Architecture:** Add a backward-compatible pre-bind/connect socket-option seam to `get_zmq_socket()`. Use it only for the dedicated status PUSH/PULL channel: both ends conflate to the newest message, while the Scheduler sender also uses zero send timeout and zero linger. Keep publishers, protocol, consensus, event loops, and model execution unchanged.

**Tech Stack:** Python 3, pyzmq, pytest, unittest.mock, PyTorch CPU tensors, SGLang registered CPU unit tests.

## Global constraints

- Target branch is `JD-v0.5.17-eep` and pinned fault-tolerance behavior remains `52f0654e147fd67568e95d71dd7d064c92dfe7f8`.
- Health feedback never waits on DPC and buffered feedback occupies constant space.
- Apply `CONFLATE` before `bind()` or `connect()`; post-connect configuration is not acceptable.
- Do not change message schema, endpoint names, port layout, process ownership, consensus, model execution, or event-loop structure.
- Preserve publisher retry semantics: `last_status` changes only after successful send.
- Use `CustomTestCase`, `register_cpu_ci`, direct pytest entry points, and mocks only for unavailable or expensive external dependencies.
- Run remote tests only through the authorized WJL structured helper inside `bt-6.200.22.140` container `sgl0514-dev-wjl`.

---

### Task 1: Add the pre-connect socket-option seam

**Files:**
- Modify: `python/sglang/srt/utils/network.py:377`
- Create: `test/registered/unit/utils/test_zmq_socket_options.py`

**Interfaces:**
- Consumes: existing `get_zmq_socket(context, socket_type, endpoint=None, bind=True)` callers.
- Produces: `get_zmq_socket(..., socket_options: Optional[Mapping[int, int]] = None)` with options applied after `config_socket()` and before bind/connect.

- [ ] **Step 1: Write the failing real-socket and call-order tests**

Create a registered CPU test file containing a real pyzmq option assertion and a recording socket assertion. The central checks are:

```python
class TestZmqSocketOptions(CustomTestCase):
    def test_custom_options_are_applied_to_real_socket(self):
        context = zmq.Context()
        endpoint = f"inproc://status-options-{uuid.uuid4().hex}"
        socket = get_zmq_socket(
            context,
            zmq.PUSH,
            endpoint,
            bind=False,
            socket_options={
                zmq.CONFLATE: 1,
                zmq.SNDTIMEO: 0,
                zmq.LINGER: 0,
            },
        )
        try:
            self.assertEqual(socket.getsockopt(zmq.CONFLATE), 1)
            self.assertEqual(socket.getsockopt(zmq.SNDTIMEO), 0)
            self.assertEqual(socket.getsockopt(zmq.LINGER), 0)
        finally:
            socket.close(0)
            context.term()

    def test_custom_options_precede_connect(self):
        events = []
        socket = RecordingSocket(events)
        context = RecordingContext(socket)
        with patch("sglang.srt.utils.network.config_socket", side_effect=lambda *_: events.append("defaults")):
            get_zmq_socket(
                context,
                zmq.PUSH,
                "tcp://127.0.0.1:12345",
                bind=False,
                socket_options={zmq.CONFLATE: 1},
            )
        self.assertEqual(events, ["defaults", ("setsockopt", zmq.CONFLATE, 1), "connect"])
```

Define the recording doubles in the same test file:

```python
class RecordingSocket:
    def __init__(self, events):
        self.events = events

    def setsockopt(self, option, value):
        self.events.append(("setsockopt", option, value))

    def connect(self, endpoint):
        self.events.append("connect")


class RecordingContext:
    def __init__(self, socket):
        self._socket = socket

    def socket(self, socket_type):
        return self._socket
```

Register the file with
`register_cpu_ci(est_time=1, suite="base-a-test-cpu")` and inherit
`CustomTestCase`.

- [ ] **Step 2: Run the focused test and verify RED**

Run in the authorized container:

```bash
env PYTHONPATH=python python -m pytest test/registered/unit/utils/test_zmq_socket_options.py -q
```

Expected: both tests fail because `get_zmq_socket()` does not accept
`socket_options`.

- [ ] **Step 3: Implement the minimal backward-compatible seam**

Change the signature and both endpoint branches without changing defaults:

```python
def get_zmq_socket(
    context: zmq.Context,
    socket_type: zmq.SocketType,
    endpoint: Optional[str] = None,
    bind: bool = True,
    socket_options: Optional[Mapping[int, int]] = None,
) -> Union[zmq.Socket, Tuple[int, zmq.Socket]]:
    socket = context.socket(socket_type)

    if endpoint is not None and is_zmq_endpoint_ipv6(endpoint):
        socket.setsockopt(zmq.IPV6, 1)
    config_socket(socket, socket_type)
    for option, value in (socket_options or {}).items():
        socket.setsockopt(option, value)

    if endpoint is None:
        port = socket.bind_to_random_port("tcp://*")
        return port, socket
    if bind:
        socket.bind(endpoint)
    else:
        socket.connect(endpoint)
    return socket
```

Import `Mapping` from `typing`. Ensure the options loop occurs before
`bind_to_random_port()`, `bind()`, or `connect()` in every branch.

- [ ] **Step 4: Run focused and nearby utility tests**

```bash
env PYTHONPATH=python python -m pytest test/registered/unit/utils/test_zmq_socket_options.py -q
env PYTHONPATH=python python -m pytest test/registered/unit/managers/test_msgpack_ipc_roundtrip.py -q
```

Expected: all tests pass.

- [ ] **Step 5: Commit Task 1**

```bash
git add python/sglang/srt/utils/network.py test/registered/unit/utils/test_zmq_socket_options.py
git commit -m "feat(zmq): support pre-connect socket options"
```

### Task 2: Configure the Elastic EP status channel

**Files:**
- Modify: `python/sglang/srt/managers/scheduler_components/ipc_channels.py:56`
- Modify: `python/sglang/srt/managers/data_parallel_controller.py:155`
- Modify: `test/registered/unit/elastic_ep/test_control_plane.py`

**Interfaces:**
- Consumes: Task 1 `get_zmq_socket(..., socket_options=...)`.
- Produces: Scheduler PUSH options `{CONFLATE: 1, SNDTIMEO: 0, LINGER: 0}` and DPC PULL option `{CONFLATE: 1}`.

- [ ] **Step 1: Write failing wiring tests**

Add `TestElasticEPStatusSocketOptions` to the control-plane test. Patch the
module-local `get_zmq_socket` and assert the option mappings passed for the
controller endpoint. Import `SimpleNamespace`, `zmq`,
`SchedulerIpcChannels`, and the `data_parallel_controller` module. Use minimal
helpers rather than constructing a full DPC:

```python
def test_scheduler_status_sender_uses_latest_value_zero_wait_options(self):
    socket = MagicMock()
    port_args = SimpleNamespace(
        scheduler_input_ipc_name="ipc://scheduler-input",
        rpc_ipc_name="ipc://rpc",
        tokenizer_ipc_name="ipc://tokenizer",
        controller_input_ipc_name="ipc://controller-status",
        detokenizer_ipc_name="ipc://detokenizer",
        metrics_ipc_name="ipc://metrics",
    )
    with patch(
        "sglang.srt.managers.scheduler_components.ipc_channels.get_zmq_socket",
        return_value=socket,
    ) as get_socket, patch(
        "sglang.srt.managers.scheduler_components.ipc_channels.zmq.Context"
    ):
        SchedulerIpcChannels.create(
            port_args=port_args,
            is_rank_zero=True,
            skip_tokenizer_init=False,
            metrics_enabled=False,
            enable_scripted_runtime=False,
        )
    controller_call = next(
        call for call in get_socket.call_args_list
        if call.args[2] == port_args.controller_input_ipc_name
    )
    assert controller_call.kwargs["socket_options"] == {
        zmq.CONFLATE: 1,
        zmq.SNDTIMEO: 0,
        zmq.LINGER: 0,
}

def test_dpc_status_receiver_conflates_to_latest_value(self):
    context = MagicMock()
    socket = MagicMock()
    create_receiver = getattr(
        data_parallel_controller,
        "_create_scheduler_status_receiver",
        None,
    )
    assert create_receiver is not None
    with patch(
        "sglang.srt.managers.data_parallel_controller.get_zmq_socket",
        return_value=socket,
    ) as get_socket:
        result = create_receiver(context, "ipc://controller-status")
    assert result is socket
    get_socket.assert_called_once_with(
        context,
        zmq.PULL,
        "ipc://controller-status",
        True,
        socket_options={zmq.CONFLATE: 1},
    )
```

Extract the DPC receiver construction into a private function
`_create_scheduler_status_receiver(context, endpoint)` and test it with a
patched `get_zmq_socket`, asserting `{zmq.CONFLATE: 1}`. The constructor calls
that helper in the same existing branch.

- [ ] **Step 2: Run the focused tests and verify RED**

```bash
env PYTHONPATH=python python -m pytest test/registered/unit/elastic_ep/test_control_plane.py -k "StatusSocketOptions" -q
```

Expected: sender assertion fails because no mapping is passed; receiver test
fails because `_create_scheduler_status_receiver` does not exist.

- [ ] **Step 3: Wire the sender and receiver options**

In `SchedulerIpcChannels.create()`:

```python
send_to_controller_raw = get_zmq_socket(
    context,
    zmq.PUSH,
    port_args.controller_input_ipc_name,
    False,
    socket_options={
        zmq.CONFLATE: 1,
        zmq.SNDTIMEO: 0,
        zmq.LINGER: 0,
    },
)
```

In `data_parallel_controller.py`:

```python
def _create_scheduler_status_receiver(context, endpoint):
    return get_zmq_socket(
        context,
        zmq.PULL,
        endpoint,
        True,
        socket_options={zmq.CONFLATE: 1},
    )
```

Replace only the existing `recv_from_scheduler = get_zmq_socket(...)` call
with the helper. Do not alter the tokenizer socket or event loop.

- [ ] **Step 4: Verify socket, publisher retry, and DPC behavior**

```bash
env PYTHONPATH=python python -m pytest test/registered/unit/elastic_ep/test_control_plane.py -k "StatusSocketOptions or ElasticEPStatusPublisher" -q
env PYTHONPATH=python python -m pytest test/registered/unit/managers/test_data_parallel_controller.py -q
```

Expected: all selected tests pass, including
`test_controller_does_not_cache_failed_publish`.

- [ ] **Step 5: Commit Task 2**

```bash
git add python/sglang/srt/managers/scheduler_components/ipc_channels.py python/sglang/srt/managers/data_parallel_controller.py test/registered/unit/elastic_ep/test_control_plane.py
git commit -m "fix(elastic-ep): bound controller status feedback"
```

### Task 3: Close the pinned-snapshot boundary test gaps

**Files:**
- Modify: `test/registered/unit/elastic_ep/test_control_plane.py`

**Interfaces:**
- Consumes: existing `ElasticEPState.submit_active_snapshot()`, `commit_active_snapshot()`, Mooncake dispatcher timeout, Scheduler retract helper, and EPLB rebalance interface.
- Produces: four explicitly named CPU regression tests required by the original port plan.

Add imports for `os`, `envs`, and `_MooncakeEPDispatcherImpl` before adding
the tests. Keep the existing Mooncake module stub local to its test.

- [ ] **Step 1: Add the suspect-versus-death authority tests**

Use the real state implementation and patch only the distributed collective:

```python
def test_a2a_timeout_only_marks_suspect(self):
    state = _make_state(world=4)
    state.active_ranks[2] = 0  # Mooncake timeout signal
    pg_health = torch.ones(4, dtype=torch.int32)
    state.submit_active_snapshot(pg_health, non_blocking=False)
    with patch("torch.distributed.all_reduce"):
        assert state.commit_active_snapshot(pg_health, MagicMock()) is True
    assert state.committed_active_ranks_cpu.tolist() == [1, 1, 1, 1]
    assert state.ep_suspect_ranks() == [2]

def test_pg_probe_is_death_authority(self):
    state = _make_state(world=4)
    pg_health = torch.tensor([1, 1, 0, 1], dtype=torch.int32)
    state.submit_active_snapshot(pg_health, non_blocking=False)
    with patch("torch.distributed.all_reduce"):
        assert state.commit_active_snapshot(pg_health, MagicMock()) is True
    assert state.committed_active_ranks_cpu.tolist() == [1, 1, 0, 1]
    assert state.ep_suspect_ranks() == []
```

- [ ] **Step 2: Add the pinned Mooncake timeout test**

Stub only `mooncake.mooncake_ep_buffer.Buffer`, clear
`SGLANG_MOONCAKE_EP_TIMEOUT_US`, construct `_MooncakeEPDispatcherImpl`, and
assert the pinned timeout reaches the buffer:

```python
def test_mooncake_fault_timeout_matches_pinned_snapshot(self):
    mooncake = ModuleType("mooncake")
    mooncake.__path__ = []
    buffer_module = ModuleType("mooncake.mooncake_ep_buffer")
    buffer_module.Buffer = object
    buffer = MagicMock()
    buffer.dispatch.return_value = (
        torch.empty(0),
        torch.empty(0),
        None,
        MagicMock(),
        MagicMock(),
    )
    state = _make_state(world=4)

    with patch.dict(
        sys.modules,
        {
            "mooncake": mooncake,
            "mooncake.mooncake_ep_buffer": buffer_module,
        },
    ), patch.dict(os.environ, {}, clear=False), patch.object(
        ElasticEPStateManager, "instance", return_value=state
    ):
        os.environ.pop("SGLANG_MOONCAKE_EP_TIMEOUT_US", None)
        dispatcher = _MooncakeEPDispatcherImpl(
            group=MagicMock(),
            router_topk=2,
            permute_fusion=False,
            num_experts=4,
            num_local_experts=1,
            hidden_size=8,
            params_dtype=torch.float16,
            return_recv_hook=False,
            deepep_mode=MagicMock(),
        )
        dispatcher._get_buffer = MagicMock(return_value=buffer)
        hidden = torch.zeros((1, 8))
        topk_ids = torch.zeros((1, 2), dtype=torch.int64)
        dispatcher._dispatch_core(hidden, topk_ids)
        assert buffer.dispatch.call_args.args[5] == -1
        dispatcher.first_execution = False
        dispatcher._dispatch_core(hidden, topk_ids)

    assert dispatcher.timeout_us == 50_000_000
    assert buffer.dispatch.call_args.args[5] == 50_000_000
```

- [ ] **Step 3: Add the retract/cache/EPLB integration test**

Build a `TestScheduler` with one mock running request and a finite rebalance
generator. Call the real `_retract_all_and_rebalance_on_rank_fault()` and assert:

```python
def test_retract_clears_cache_and_updates_expert_state(self):
    state = _make_state(world=4)
    state.committed_active_ranks_cpu[2] = 0
    state.submit_active_snapshot(
        torch.ones(4, dtype=torch.int32), non_blocking=False
    )
    sched = TestScheduler(world=4)
    req = MagicMock(retraction_count=0)
    batch = MagicMock()
    batch.reqs = [req]
    sched.running_batch = batch
    sched.cur_batch_for_debug = batch
    sched.chunked_req = MagicMock()
    sched.ipc_channels = MagicMock()
    sched._add_request_to_queue = MagicMock()
    eplb_manager = MagicMock()
    eplb_manager.rebalance.return_value = iter([None])
    sched.tp_worker = MagicMock()
    sched.tp_worker.model_runner.eplb_manager = eplb_manager

    with patch.object(
        ElasticEPStateManager, "instance", return_value=state
    ), patch("torch.cuda.synchronize"), patch.object(
        envs.SGLANG_ELASTIC_EP_MAX_RETRACTION, "get", return_value=3
    ):
        sched._retract_all_and_rebalance_on_rank_fault()

    batch.release_req.assert_called_once_with(0, 0, sched.server_args)
    assert state.pending_staging_slots == []
    eplb_manager.rebalance.assert_called_once()
    assert state.active_ranks.tolist() == state.committed_active_ranks_cpu.tolist()
    assert state.last_handled_committed_active_ranks_cpu.tolist() == state.committed_active_ranks_cpu.tolist()
```

Patch `torch.cuda.synchronize` and publisher output only; use the production
retract and rebalance orchestration.

- [ ] **Step 4: Run the four named tests**

```bash
env PYTHONPATH=python python -m pytest test/registered/unit/elastic_ep/test_control_plane.py -k "a2a_timeout_only_marks_suspect or pg_probe_is_death_authority or mooncake_fault_timeout_matches_pinned_snapshot or retract_clears_cache_and_updates_expert_state" -q
```

Expected: four tests pass. If a test exposes a behavior mismatch, stop and add
the smallest production change through a new RED/GREEN cycle; do not weaken
the pinned assertion.

- [ ] **Step 5: Commit Task 3**

```bash
git add test/registered/unit/elastic_ep/test_control_plane.py
git commit -m "test(elastic-ep): cover transport and cleanup boundaries"
```

### Task 4: Verify and publish the report revision

**Files:**
- Create outside the repository: `../<UTC timestamp>--elastic-ep-v0.5.17-port-review-zh.md`
- Preserve: `../elastic-ep-v0.5.17-port-review-zh.md`

**Interfaces:**
- Consumes: Tasks 1-3 commits and fresh container test output.
- Produces: timestamped Chinese review evidence with corrected root cause and final verification results.

- [ ] **Step 1: Run static checks**

```bash
env PYTHONPATH=python python -m compileall -q python/sglang/srt/utils/network.py python/sglang/srt/managers/scheduler_components/ipc_channels.py python/sglang/srt/managers/data_parallel_controller.py test/registered/unit/utils/test_zmq_socket_options.py test/registered/unit/elastic_ep/test_control_plane.py
env PYTHONPATH=python python -m ruff check python/sglang/srt/utils/network.py python/sglang/srt/managers/scheduler_components/ipc_channels.py python/sglang/srt/managers/data_parallel_controller.py test/registered/unit/utils/test_zmq_socket_options.py test/registered/unit/elastic_ep/test_control_plane.py
env PYTHONPATH=python python -m ruff format --check python/sglang/srt/utils/network.py python/sglang/srt/managers/scheduler_components/ipc_channels.py python/sglang/srt/managers/data_parallel_controller.py test/registered/unit/utils/test_zmq_socket_options.py test/registered/unit/elastic_ep/test_control_plane.py
```

Expected: exit code 0 for every command.

- [ ] **Step 2: Run focused and surrounding unit tests**

```bash
env PYTHONPATH=python python -m pytest test/registered/unit/utils/test_zmq_socket_options.py test/registered/unit/elastic_ep/test_control_plane.py test/registered/unit/managers/test_data_parallel_controller.py -q
env PYTHONPATH=python python -m pytest test/registered/unit -k "eplb or radix_cache or mooncake or msgpack_ipc" -q
```

Expected: all selected tests pass. Record exact counts, elapsed time, and exit
codes.

- [ ] **Step 3: Run repository integrity checks**

```bash
git diff --check origin/JD-v0.5.17...HEAD
git status --short
git log --oneline --decorate -8
```

Expected: diff check exits 0; status is clean before report creation.

- [ ] **Step 4: Create the timestamped Chinese report revision**

Copy the existing report content into a newly named UTC artifact using
`apply_patch`, then correct Section 7.5 and Section 11.4 to record:

- the observed unbounded-queue reproduction and why 10 ms alone is ineffective;
- the final pre-connect `CONFLATE=1`, `SNDTIMEO=0`, `LINGER=0` design;
- the necessary third-file `get_zmq_socket()` seam and its compatibility;
- the four added boundary tests;
- exact commits, commands, exit codes, pass counts, and remaining exclusions;
- `Generated at (UTC): <same timestamp as filename>` near the start.

Do not overwrite or delete the earlier report.

- [ ] **Step 5: Push the completed branch**

After verification and report creation, push `JD-v0.5.17-eep` to the Coding JD
upstream selected by the branch's existing Git routing hooks. Confirm the
remote branch resolves to the local HEAD.
