# Elastic EP fault-tolerance port design

## Goal

Port the complete Elastic EP fault-tolerance behavior from Coding JD branch
`wjl/eep-hang-patch` at the exact source commit
`52f0654e147fd67568e95d71dd7d064c92dfe7f8` onto local branch
`JD-v0.5.17-eep`. Preserve the architecture and unrelated improvements already
present in `JD-v0.5.17`.

The source commit is a pinned behavior snapshot. Commits after it are outside
the port. The resulting local history does not need to reproduce the source
branch's commit topology.

## Approach

Use a semantic snapshot port rather than replaying the source history or
copying whole source files. Compare the fault-tolerance behavior and tests at
the pinned source tree with the target tree, then adapt each behavior at the
corresponding v0.5.17 interface.

The port produces one integration commit whose message records the pinned
source commit. It excludes unrelated source-branch changes, CI changes, and
old-baseline implementation details.

## Functional scope

The target must preserve the following behavior from the pinned source tree:

1. Submit active-rank snapshots after model forward work and drain their async
   device-to-host copies before the staging data is consumed.
2. Commit snapshots through a global active-rank consensus at a scheduler
   boundary where no rank is executing Elastic EP all-to-all work.
3. Keep result-boundary processing free of cross-rank collectives. The result
   boundary only drains the snapshot copy.
4. Handle hard rank faults by reaching a consistent active-rank view,
   retracting affected in-flight batches, and rebalancing experts.
5. Handle soft all-to-all stalls consistently across ranks, including streak
   tracking, escalation, and the guard against removing all active ranks.
6. Admit recovering ranks only after every surviving rank agrees that recovery
   is ready, then execute recovery and rebalancing consistently.
7. Force a globally consistent serial/overlap decision when grammar state would
   otherwise make data-parallel ranks enter different scheduler paths.
8. Publish Elastic EP status without allowing publisher failures or stale
   publisher state to corrupt the control path.
9. Clear snapshot staging state completely during retract or recovery,
   including resetting the double-buffer slot selector.
10. Fail fast when a collective fault/recovery handler raises, so surviving
    ranks do not continue into a later mismatched collective.
11. Preserve the pinned source behavior for controller IPC, environment
    settings, Mooncake fault timeout handling, EPLB integration, and radix-cache
    cooperation required by the fault-tolerance path.

## Target architecture

### State and consensus

`python/sglang/srt/elastic_ep/elastic_ep.py` owns active-rank snapshots,
staging slots, process-group probing, committed state, suspect state, and
recovery readiness. Its public and internal interfaces will be adapted to the
v0.5.17 implementation rather than replaced wholesale.

### Scheduler control plane

An Elastic EP scheduler mixin owns admission, snapshot draining, retract,
stall, fault, and recovery orchestration. The normal, overlap, and
disaggregated-decode loops call the admission gate immediately before
`run_batch`. The gate synchronizes the previous snapshot copy before consensus
and returns whether the current forward may proceed.

The existing v0.5.17 scheduler plan objects, scheduler components, batch-result
processor, and write-after-read barrier remain authoritative. Elastic EP hooks
must compose with those boundaries.

### Forward-path snapshot submission

The v0.5.17 model runner submits the active snapshot after the relevant forward
work. The port will retain current model-runner control flow and insert only the
snapshot interaction required by the pinned source behavior.

### Status and controller integration

Elastic EP status publishing and controller-input IPC are added at the existing
v0.5.17 initialization and message-routing seams. Port assignments and channel
construction must not displace v0.5.17 channels.

### EPLB and cache integration

Fault, retract, rebalance, and recovery paths must update expert placement and
cache state coherently. Existing v0.5.17 EPLB and radix-cache APIs remain the
integration surface; older source implementations do not overwrite newer
target abstractions.

## Control flow

For each scheduler iteration with a batch:

1. Build the target v0.5.17 schedule plan and converge on the batch decision.
2. If Elastic EP is enabled, synchronize the previous snapshot copy and commit
   its active-rank snapshot through global consensus.
3. If consensus selects fault, stall, or recovery handling, execute the same
   control branch on all surviving ranks, retract as required, and skip the
   current forward.
4. Otherwise, run the batch and submit the next active-rank snapshot from the
   model-forward path.
5. When processing the prior result, drain its copy event without issuing a
   collective, then use the existing v0.5.17 result processor.

This ordering prevents an Elastic EP all-to-all operation on one rank from
waiting cyclically with a control-plane collective on another rank.

## Error handling

- A rank-local recovery-ready observation is never sufficient; readiness is
  reduced across surviving ranks before recovery begins.
- Handler exceptions synchronize local device work where required and are
  re-raised so the scheduler exits instead of entering another collective with
  inconsistent state.
- Status publication remains best-effort and cannot change the fault-tolerance
  decision.
- Snapshot cleanup resets pending slots and the next-slot selector.
- The port does not add a process group, background thread, or asynchronous
  control-plane state machine.

## Testing strategy

Use test-driven adaptation:

1. Port the pinned source's control-plane unit tests to v0.5.17 test APIs.
2. Run the focused tests before production changes and confirm they fail for
   missing fault-tolerance behavior.
3. Implement the smallest compatible production changes by component.
4. Run focused tests after each component, then the full Elastic EP unit-test
   set.
5. Run repository-supported static checks on every changed Python file.
6. Run the accepted unit tests in the existing `sgl0514-dev-wjl` container on
   exact SSH alias `bt-6.200.22.140` when the required runtime is unavailable
   locally. Remote commands use the structured WJL helper only.

The acceptance scope is static checks and unit tests. A 16-rank EAGLE v2 plus
grammar reproduction, cluster lifecycle changes, benchmarks, and soak tests are
outside this task.

## Acceptance criteria

1. The target behavior corresponds to the pinned source tree at
   `52f0654e147fd67568e95d71dd7d064c92dfe7f8`; no later source commit is used.
2. Result-boundary call paths contain no cross-rank collective.
3. Snapshot consensus occurs at the pre-`run_batch` admission boundary after
   the prior async snapshot copy is safe to consume.
4. Healthy, hard-fault, soft-stall, escalation, recovery, and handler-exception
   branches are consistent across surviving ranks.
5. Normal, overlap, and disaggregated-decode scheduler paths use compatible
   admission and result-drain semantics.
6. Existing v0.5.17 scheduler, model-runner, EPLB, cache, IPC, and result
   processing behavior unrelated to Elastic EP remains intact.
7. Focused Elastic EP unit tests and the selected surrounding regression tests
   pass.
8. Repository-supported static checks pass for all changed Python files.

## Out of scope

- Source commits after `52f0654e147fd67568e95d71dd7d064c92dfe7f8`.
- Mechanical preservation of the source branch's intermediate commits.
- Whole-file replacement when a v0.5.17 integration point exists.
- Changes to EAGLE, grammar semantics, Mooncake dispatch/combine algorithms, or
  scheduler overlap policy beyond the consistency gate required for fault
  tolerance.
- New process groups, background workers, or container/cluster mutations.
- Multi-node reproduction, performance benchmarking, and soak testing.
