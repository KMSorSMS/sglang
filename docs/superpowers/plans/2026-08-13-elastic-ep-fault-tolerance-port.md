# Elastic EP Fault-Tolerance Port Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Port the complete Elastic EP fault-tolerance behavior pinned at Coding JD `wjl/eep-hang-patch@52f0654e147fd67568e95d71dd7d064c92dfe7f8` onto `JD-v0.5.17-eep` without replacing unrelated v0.5.17 architecture.

**Architecture:** Treat the pinned tree as a behavioral oracle. Add snapshot state and a scheduler control-plane mixin, connect it at v0.5.17's pre-forward and result-processing seams, then add the status/IPC and Mooncake/EPLB/cache support on their current interfaces. Each behavioral slice begins with a focused failing test and ends with a commit.

**Tech Stack:** Python 3, PyTorch distributed, pytest, SGLang scheduler/model-runner components, Mooncake Elastic EP, Ruff or repository-supported Python static checks.

## Global Constraints

- The only source behavior snapshot is `52f0654e147fd67568e95d71dd7d064c92dfe7f8`; do not use later source commits.
- Preserve unrelated `JD-v0.5.17` scheduler, model-runner, EPLB, cache, IPC, and result-processing behavior.
- Adapt behavior to v0.5.17 interfaces; do not overwrite whole target files when a current integration seam exists.
- Result-boundary paths must not contain cross-rank collectives.
- Snapshot consensus must occur before `run_batch` after the previous snapshot copy is safe to consume.
- Do not add process groups, background workers, cluster mutations, benchmarks, or multi-node soak tests.
- Acceptance is repository-supported static checks plus focused and surrounding unit tests.
- Remote tests run only through the WJL structured helper on exact host alias `bt-6.200.22.140`, inside existing container `sgl0514-dev-wjl`, at the same absolute repository path.

---

### Task 1: Active-rank snapshot state

**Files:**
- Create: `test/registered/unit/elastic_ep/test_control_plane.py`
- Modify: `python/sglang/srt/elastic_ep/elastic_ep.py`

**Interfaces:**
- Consumes: existing `ElasticEPState` device masks, CPU mirrors, and `ElasticEPStateManager` lifecycle.
- Produces: `submit_active_snapshot(forward_stream, copy_done)`, `commit_active_snapshot(process_group) -> bool`, `is_stale_snapshot() -> bool`, `ep_suspect_ranks() -> list[int]`, `has_ep_suspects() -> bool`, `mark_snapshot_handled() -> None`, `clear_pending_snapshots() -> None`, and `resync_active_to_committed() -> None`.

- [ ] **Step 1: Add focused tests for snapshot semantics**

Port the pinned source tests for submit/commit, stale fault detection, suspect filtering, resync, and staging reset into `test_control_plane.py`. Preserve these assertions:

```python
state.submit_active_snapshot(forward_stream, copy_done)
assert state.commit_active_snapshot(process_group=None) is True
assert state.pending_staging_slots == []

state.next_staging_slot = 1
state.clear_pending_snapshots()
assert state.next_staging_slot == 0
```

- [ ] **Step 2: Run the state tests and verify RED**

Run:

```bash
python -m pytest test/registered/unit/elastic_ep/test_control_plane.py -k "CommitSemantics or ClearPendingSnapshotsResetsSlot" -q
```

Expected: collection succeeds and tests fail because the v0.5.17 `ElasticEPState` lacks snapshot methods or staging fields.

- [ ] **Step 3: Adapt the pinned state machine to v0.5.17**

Merge the pinned fields and methods into the current dataclass while retaining v0.5.17 scale/rejoin fields and manager APIs. The state owns two CPU staging buffers, pending slot order, the next-slot selector, committed active ranks, and EP suspect consensus. `clear_pending_snapshots()` clears pending slots and resets `next_staging_slot` to zero.

- [ ] **Step 4: Run state tests and surrounding Elastic EP tests**

Run:

```bash
python -m pytest test/registered/unit/elastic_ep/test_control_plane.py -k "CommitSemantics or ClearPendingSnapshotsResetsSlot" -q
python -m pytest test/registered/unit -k elastic_ep -q
```

Expected: snapshot tests pass; any remaining failures identify later integration tasks rather than state-method defects.

- [ ] **Step 5: Review the state slice before continuing**

```bash
git diff --check
git diff -- python/sglang/srt/elastic_ep/elastic_ep.py test/registered/unit/elastic_ep/test_control_plane.py
```

### Task 2: Scheduler fault-tolerance control plane

**Files:**
- Create: `python/sglang/srt/managers/scheduler_elastic_ep_mixin.py`
- Modify: `test/registered/unit/elastic_ep/test_control_plane.py`

**Interfaces:**
- Consumes: Task 1 snapshot methods, current `ScheduleBatch`, current request queues, EPLB update entry points, `can_recover_ranks`, and scheduler streams.
- Produces: `SchedulerElasticEPMixin` with `_drain_elastic_ep_snapshot_copy(result) -> None`, `_admit_elastic_ep_forward(batch) -> bool`, fault/stall/recovery handlers, and in-flight batch retract helpers.

- [ ] **Step 1: Port the pinned scheduler-control tests**

Add the pinned tests for drain semantics, healthy admission, fault admission, stall admission, stream-before-commit ordering, recovery MIN consensus, deterministic escalation, and handler exception propagation. Keep the API contract explicit:

```python
ret = sched._drain_elastic_ep_snapshot_copy(result)
assert ret is None
assert sched._admit_elastic_ep_forward(batch) in (True, False)
```

- [ ] **Step 2: Run control-plane tests and verify RED**

Run:

```bash
python -m pytest test/registered/unit/elastic_ep/test_control_plane.py -k "Drain or Admission or Recovery or Handler" -q
```

Expected: tests fail because `scheduler_elastic_ep_mixin` or its methods do not exist.

- [ ] **Step 3: Implement the mixin against v0.5.17 scheduler APIs**

Port the pinned source control logic. Admission synchronizes `forward_stream`, commits the snapshot, publishes healthy state, and selects exactly one fault/stall/recovery branch. Result draining only synchronizes `result.copy_done`. Recovery readiness uses distributed MIN consensus; grammar serial selection uses MAX in Task 3. Handler exceptions synchronize device work and re-raise.

- [ ] **Step 4: Run the complete control-plane file**

```bash
python -m pytest test/registered/unit/elastic_ep/test_control_plane.py -q
```

Expected: all mixin and state tests pass.

- [ ] **Step 5: Review the control-plane slice before continuing**

```bash
git diff --check
git diff -- python/sglang/srt/managers/scheduler_elastic_ep_mixin.py test/registered/unit/elastic_ep/test_control_plane.py
```

### Task 3: Scheduler loops and model-forward integration

**Files:**
- Modify: `python/sglang/srt/managers/scheduler.py`
- Modify: `python/sglang/srt/disaggregation/decode.py`
- Modify: `python/sglang/srt/model_executor/model_runner.py`
- Modify: `test/registered/unit/elastic_ep/test_control_plane.py`

**Interfaces:**
- Consumes: Task 2 `_admit_elastic_ep_forward` and `_drain_elastic_ep_snapshot_copy`; Task 1 `submit_active_snapshot`.
- Produces: normal, overlap, and disaggregated-decode loops with a pre-`run_batch` admission gate; result paths with drain-only handling; model-forward snapshot submission; global grammar serial decision using MAX consensus.

- [ ] **Step 1: Add structural and grammar-consensus tests**

Port the pinned grammar tests and add AST/source assertions that every relevant `run_batch` call is dominated by `_admit_elastic_ep_forward`, all four result paths call `_drain_elastic_ep_snapshot_copy`, and the old `_handle_elastic_ep_result_boundary` name is absent.

- [ ] **Step 2: Run integration tests and verify RED**

```bash
python -m pytest test/registered/unit/elastic_ep/test_control_plane.py -k "Grammar or DrainSnapshotCopyRegression or EventLoop" -q
```

Expected: tests fail because the mixin is not inherited by `Scheduler`, event loops lack admission/drain calls, and the model runner does not submit snapshots.

- [ ] **Step 3: Integrate at current v0.5.17 seams**

Add `SchedulerElasticEPMixin` to scheduler composition. Insert this gate immediately before each Elastic EP `run_batch`:

```python
if (
    self.server_args.elastic_ep_backend is not None
    and not self._admit_elastic_ep_forward(batch)
):
    continue
```

Drain snapshot copies immediately before existing result processing without changing the v0.5.17 WAR barrier or plan bookkeeping. Submit active snapshots after the relevant model forward. Reduce the local grammar serial flag with distributed MAX so one serial rank forces all ranks serial.

- [ ] **Step 4: Run scheduler and model-runner regression tests**

```bash
python -m pytest test/registered/unit/elastic_ep/test_control_plane.py -q
python -m pytest test/registered/unit/managers -q
```

Expected: Elastic EP integration tests and existing manager result-processing tests pass.

- [ ] **Step 5: Review the forward integration before continuing**

```bash
git diff --check
git diff -- python/sglang/srt/managers/scheduler.py python/sglang/srt/disaggregation/decode.py python/sglang/srt/model_executor/model_runner.py test/registered/unit/elastic_ep/test_control_plane.py
```

### Task 4: Status publishing, environment, and controller IPC

**Files:**
- Create: `python/sglang/srt/managers/elastic_ep_status.py`
- Modify: `python/sglang/srt/environ.py`
- Modify: `python/sglang/srt/managers/data_parallel_controller.py`
- Modify: `python/sglang/srt/managers/scheduler.py`
- Modify: `python/sglang/srt/managers/scheduler_components/ipc_channels.py`
- Modify: `python/sglang/srt/ray/data_parallel_controller.py`
- Modify: `python/sglang/srt/server_args.py`
- Modify: `test/registered/unit/elastic_ep/test_control_plane.py`

**Interfaces:**
- Consumes: committed active-rank CPU mask and current v0.5.17 IPC/port construction.
- Produces: `ElasticEPStatusPublisher` implementations, `controller_input_ipc_name`, collision-free controller channel routing, metrics publishing, and pinned environment/default values.

- [ ] **Step 1: Add publisher and IPC tests**

Add tests that controller and metrics publishers receive committed masks, composite publishing isolates publisher failure, and the controller-input channel has a distinct port/name under current `PortArgs`.

- [ ] **Step 2: Run publisher tests and verify RED**

```bash
python -m pytest test/registered/unit/elastic_ep/test_control_plane.py -k "Publisher or Controller" -q
```

Expected: tests fail because publisher classes and controller-input wiring are absent.

- [ ] **Step 3: Implement publisher and IPC integration**

Adapt the pinned `elastic_ep_status.py` to current metrics APIs. Add the controller channel next to existing v0.5.17 IPC channels using an unused port offset. Preserve all current channels and server arguments. Wire scheduler-ready publication and controller consumption without making publication affect control decisions.

- [ ] **Step 4: Run focused IPC/server-argument regressions**

```bash
python -m pytest test/registered/unit/elastic_ep/test_control_plane.py -q
python -m pytest test/registered/unit/test_server_args_migration.py -q
```

Expected: publisher/control-plane and server-argument migration tests pass.

- [ ] **Step 5: Review status and IPC support before continuing**

```bash
git diff --check
git diff -- python/sglang/srt/managers/elastic_ep_status.py python/sglang/srt/environ.py python/sglang/srt/managers/data_parallel_controller.py python/sglang/srt/managers/scheduler.py python/sglang/srt/managers/scheduler_components/ipc_channels.py python/sglang/srt/ray/data_parallel_controller.py python/sglang/srt/server_args.py test/registered/unit/elastic_ep/test_control_plane.py
```

### Task 5: Mooncake, EPLB, and cache cooperation

**Files:**
- Modify: `python/sglang/srt/layers/moe/token_dispatcher/mooncake.py`
- Modify: `python/sglang/srt/eplb/eplb_algorithms/__init__.py`
- Modify: `python/sglang/srt/eplb/expert_location.py`
- Modify: `python/sglang/srt/eplb/expert_location_updater.py`
- Modify: `python/sglang/srt/mem_cache/radix_cache.py`
- Modify: `python/sglang/srt/eplb/eplb_manager.py`
- Modify: `test/registered/unit/elastic_ep/test_control_plane.py`

**Interfaces:**
- Consumes: Task 2 fault/stall/recovery decisions and current v0.5.17 expert-location/cache APIs.
- Produces: pinned finite Mooncake timeout semantics, PG-probe death authority, suspect-only A2A timeout handling, rank-aware expert placement updates, and safe cache cleanup/retract behavior.

- [ ] **Step 1: Add boundary tests for transport and state cleanup**

Add tests named `test_a2a_timeout_only_marks_suspect`, `test_pg_probe_is_death_authority`, `test_mooncake_fault_timeout_matches_pinned_snapshot`, and `test_retract_clears_cache_and_updates_expert_state`. Assert that the timeout path leaves the committed mask unchanged until process-group consensus and that cleanup calls the current v0.5.17 cache and expert-location interfaces.

- [ ] **Step 2: Run transport/integration tests and verify RED**

```bash
python -m pytest test/registered/unit/elastic_ep/test_control_plane.py -k "Stall or Fault or Cleanup or Timeout" -q
```

Expected: tests fail at the missing pinned timeout/suspect or integration behavior.

- [ ] **Step 3: Adapt transport, EPLB, and cache changes**

Apply only the semantic deltas required by the pinned source. Preserve v0.5.17's newer EPLB manager and cache abstractions. A Mooncake A2A timeout marks a suspect/retract condition; the process-group probe remains the sole death authority. Keep a finite fault timeout.

- [ ] **Step 4: Run focused and surrounding unit tests**

```bash
python -m pytest test/registered/unit/elastic_ep/test_control_plane.py -q
python -m pytest test/registered/unit -k "eplb or radix_cache or mooncake" -q
```

Expected: control-plane and available EPLB/cache/Mooncake unit tests pass.

- [ ] **Step 5: Review integration support before final verification**

```bash
git diff --check
git diff -- python/sglang/srt/layers/moe/token_dispatcher/mooncake.py python/sglang/srt/eplb python/sglang/srt/mem_cache/radix_cache.py test/registered/unit/elastic_ep/test_control_plane.py
```

### Task 6: Static checks and remote unit-test acceptance

**Files:**
- Modify only if checks expose a port defect: files changed in Tasks 1–5.
- Verify: all files changed since `ab9375de2`.

**Interfaces:**
- Consumes: completed local feature commits.
- Produces: clean static-check output, passing focused tests, and a final integration commit message that records source snapshot `52f0654e147fd67568e95d71dd7d064c92dfe7f8`.

- [ ] **Step 1: Run local diff and syntax checks**

```bash
git diff --check ab9375de2..HEAD
python -m compileall -q python/sglang/srt/elastic_ep python/sglang/srt/managers/scheduler_elastic_ep_mixin.py python/sglang/srt/managers/elastic_ep_status.py
```

Expected: exit code 0.

- [ ] **Step 2: Run repository-supported formatting/lint checks**

Run the repository formatter/linter on the exact changed Python paths discovered with `git diff --name-only ab9375de2..HEAD`. Expected: exit code 0 with no unformatted changed file.

- [ ] **Step 3: Verify the remote execution route and instructions**

Use `wjl_remote.py status --host bt-6.200.22.140`, read `/ufs/wjl/workspace/AGENTS.md` and applicable repository `AGENTS.md`, and confirm the same absolute repository path is mounted in `sgl0514-dev-wjl`.

- [ ] **Step 4: Run authorized focused tests in the existing container**

Use only:

```bash
python3 /Users/weijunlin.113/.agents/skills/wjl-workspace/scripts/wjl_remote.py container-exec --authorized-current-task --host bt-6.200.22.140 sgl0514-dev-wjl -- python -m pytest test/registered/unit/elastic_ep/test_control_plane.py -q
```

Then run the selected surrounding unit tests from Tasks 3–5 through the same structured `container-exec` route. Expected: every command exits 0.

- [ ] **Step 5: Record the pinned source in the single integration commit**

Stage the exact implementation and test paths from Tasks 1–5, then commit them together:

```bash
git add python/sglang/srt/disaggregation/decode.py python/sglang/srt/elastic_ep/elastic_ep.py python/sglang/srt/environ.py python/sglang/srt/eplb python/sglang/srt/layers/moe/token_dispatcher/mooncake.py python/sglang/srt/managers/data_parallel_controller.py python/sglang/srt/managers/elastic_ep_status.py python/sglang/srt/managers/scheduler.py python/sglang/srt/managers/scheduler_components/ipc_channels.py python/sglang/srt/managers/scheduler_elastic_ep_mixin.py python/sglang/srt/mem_cache/radix_cache.py python/sglang/srt/model_executor/model_runner.py python/sglang/srt/ray/data_parallel_controller.py python/sglang/srt/server_args.py test/registered/unit/elastic_ep/test_control_plane.py
git commit -m "feat(elastic-ep): port v0.5.17 fault tolerance" -m "Source behavior: wjl/eep-hang-patch@52f0654e147fd67568e95d71dd7d064c92dfe7f8"
```

The final handoff must report all commits, changed paths, exact test commands, exit codes, and any test not run.
