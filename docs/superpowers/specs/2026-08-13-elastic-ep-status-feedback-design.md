---
title: "Elastic EP status feedback design"
description: "Design for a bounded, non-blocking Scheduler-to-DPC health feedback channel on JD-v0.5.17-eep."
---

# Elastic EP status feedback design

## Goal

Prevent Elastic EP health feedback from blocking the Scheduler or accumulating
unbounded stale messages when the DataParallelController (DPC) is unavailable
or slow. Keep the inference hot path non-blocking and preserve the existing
v0.5.17 architecture.

## Context and root cause

`ControllerElasticEPStatusPublisher` converts the consensus-committed
`committed_active_ranks` mask to one boolean per DP worker and sends an
`ActiveRanksOutput` from the Scheduler to the DPC. The DPC combines that health
snapshot with `dp_active` to build its eligible request-routing set.

This message is latest-value feedback, not an ordered event log and not a fault
verdict. Intermediate snapshots have no value after a newer snapshot exists.

The current PUSH/PULL sockets inherit `SNDHWM=0` and `RCVHWM=0`, which make the
queues unbounded. The Scheduler-side send also inherits `SNDTIMEO=-1`. A slow
DPC can therefore cause unbounded message accumulation, while a transport that
cannot accept a message can block the Scheduler indefinitely. A container
reproduction showed that adding `SNDTIMEO=10ms` alone did not address the slow
consumer case: with the DPC side not reading, 20,000 messages of 4 KiB each
were accepted in 0.052 seconds because the queues remained unbounded.

## Selected design

Configure the dedicated health-feedback channel as a latest-value channel:

- Set `CONFLATE=1`, `SNDTIMEO=0`, and `LINGER=0` on the Scheduler PUSH socket.
- Set `CONFLATE=1` on the DPC PULL socket.
- Keep `SenderWrapper`, `ControllerElasticEPStatusPublisher`, and
  `CompositeElasticEPStatusPublisher` unchanged.
- Keep the existing DPC event-loop `NOBLOCK` receive unchanged.

`CONFLATE` is a connection-scoped ZeroMQ option and must be applied before the
socket binds or connects. Extend `get_zmq_socket()` with an optional
`socket_options` mapping that is applied after the repository defaults but
before `bind()` or `connect()`. Existing callers omit the argument and retain
their current behavior. The status sender and receiver pass only their
dedicated option mappings through this seam.

`CONFLATE=1` bounds each socket queue to the newest single-part
`ActiveRanksOutput` message. `SNDTIMEO=0` makes an unavailable send fail
immediately with `zmq.Again`; the existing composite publisher isolates that
failure from scheduling. `LINGER=0` prevents shutdown from waiting on an
undelivered feedback message.

The controller publisher updates `last_status` only after a successful send.
Therefore a failed send does not poison its deduplication cache and a later
publish retries the current snapshot.

## Alternatives rejected

### Add only `SNDTIMEO=10ms`

This does not bound queues and does not fire while an unbounded queue can still
accept messages. When it does fire, it can add 10 ms to a Scheduler hot-path
operation. It does not satisfy the memory or latency requirements.

### Use finite HWM values without conflate

`SNDHWM=1`, `RCVHWM=1`, and a zero timeout would bound memory, but a recovered
DPC could still consume an older queued snapshot before receiving the newest
state. Latest-value conflate matches the message semantics more directly.

### Add a thread or asynchronous forwarding queue

A dedicated sender thread would isolate blocking but add lifecycle, shutdown,
and synchronization complexity. It is unnecessary for a single latest-value
feedback channel and violates the minimal-change constraint.

## Data flow and invariants

1. Elastic EP commits a capacity-sized active-rank snapshot through the
   existing process-group consensus.
2. The controller publisher reduces it to one health boolean per DP worker.
3. If the boolean list equals `last_status`, publication is skipped.
4. Otherwise, the Scheduler attempts one immediate PUSH send.
5. A successful send updates `last_status`. An unavailable send raises
   `zmq.Again`, is isolated by the composite publisher, and leaves
   `last_status` unchanged for retry.
6. The DPC consumes at most the newest pending health snapshot and computes
   `eligible[i] = dp_active[i] and health_status[i]`.

The implementation must preserve these invariants:

- Health feedback never waits on the DPC.
- Buffered health feedback occupies constant space.
- The DPC does not need to replay superseded snapshots.
- Failure to publish cannot alter fault consensus or terminate the Scheduler.
- Socket schema, endpoint names, port layout, process ownership, and event-loop
  structure do not change.

## Testing strategy

Follow the repository's CPU unit-test conventions and TDD workflow.

1. Add a real pyzmq socket-option test for the dedicated Scheduler and DPC
   channel endpoints. It must fail before the socket configuration is added and
   then assert `CONFLATE=1`, Scheduler `SNDTIMEO=0`, and Scheduler `LINGER=0`.
2. Preserve and run the publisher retry regression proving a failed send does
   not update `last_status` and does not prevent the next send attempt.
3. Add the four missing pinned-snapshot boundary tests:
   `test_a2a_timeout_only_marks_suspect`,
   `test_pg_probe_is_death_authority`,
   `test_mooncake_fault_timeout_matches_pinned_snapshot`, and
   `test_retract_clears_cache_and_updates_expert_state`.
4. Run the focused Elastic EP control-plane tests, DPC tests, related
   Mooncake/EPLB/radix-cache tests, Python compilation, formatter/linter checks,
   and `git diff --check` in the existing `sgl0514-dev-wjl` container.

The socket test asserts configuration and retry semantics instead of timing a
large send loop, avoiding flaky timing and memory-sensitive CI behavior.

## Scope and compatibility

Production changes are limited to a backward-compatible pre-connect option seam
and the dedicated Scheduler-to-DPC status socket configuration in:

- `python/sglang/srt/utils/network.py`
- `python/sglang/srt/managers/scheduler_components/ipc_channels.py`
- `python/sglang/srt/managers/data_parallel_controller.py`

Tests remain under `test/registered/unit/` and use existing production APIs.
No protocol, dependency, environment-variable, event-loop, model-execution,
fault-consensus, or cluster-lifecycle change is permitted.

Real multi-node 4-to-8 scaling, Scheduler event-loop refactoring, performance
benchmarking, and soak testing remain outside this change.

## Acceptance criteria

1. The Scheduler PUSH socket reports `CONFLATE=1`, `SNDTIMEO=0`, and
   `LINGER=0`; the DPC PULL socket reports `CONFLATE=1`.
2. A feedback send that cannot enqueue returns immediately and remains
   best-effort through the existing composite publisher.
3. Failed publication leaves `last_status` unchanged so the latest snapshot is
   retried.
4. The DPC continues to route only to workers for which both `dp_active` and
   the latest health status are true.
5. The four pinned-snapshot Mooncake/PG/cache/EPLB boundary tests pass.
6. Focused and surrounding unit tests, static checks, and `git diff --check`
   pass in the authorized container environment.
