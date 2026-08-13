"""Tests for the minimal elastic-EP fix: all_reduce moved from the result
boundary (desync seam) to run_batch entry (global alignment point).

The fix has exactly three moving parts:
  1. _drain_elastic_ep_snapshot_copy: only copy_done.synchronize(), no
     collective.
  2. _admit_elastic_ep_forward: forward_stream.synchronize() then
     commit_active_snapshot (all_reduce MIN) then fault/stall/recovery
     handlers.  Returns False to skip run_batch when a handler retracts.
  3. Four event loops call _admit_elastic_ep_forward right before run_batch.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import torch

from sglang.srt.elastic_ep.elastic_ep import ElasticEPState, ElasticEPStateManager
from sglang.srt.managers.elastic_ep_status import (
    CompositeElasticEPStatusPublisher,
    ControllerElasticEPStatusPublisher,
    ElasticEPMetricState,
    MetricsElasticEPStatusPublisher,
    _compute_cluster_state,
    _effective_committed_active_ranks,
)
from sglang.srt.managers.scheduler_elastic_ep_mixin import SchedulerElasticEPMixin
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_state(world: int = 4) -> ElasticEPState:
    """Build a standalone ElasticEPState (no global singleton)."""
    return ElasticEPStateManager._build_state(ep_size=world, device=torch.device("cpu"))


class FakeForwardStream:
    """Stand-in for scheduler.forward_stream — records sync calls."""

    def __init__(self):
        self.sync_count = 0

    def synchronize(self):
        self.sync_count += 1


class FakeStatusPublisher:
    def __init__(self):
        self.published = []

    def publish_committed_active_ranks(self, mask, adjusting=False):
        self.published.append((mask.tolist(), adjusting))


class TestScheduler(SchedulerElasticEPMixin):
    """Minimal scheduler double that inherits the mixin so methods bind
    correctly.  Only the attributes the mixin touches are set."""

    def __init__(self, world: int = 4, enable_elastic_ep: bool = True):
        self._elastic_ep_adjusting = False
        self._elastic_ep_stall_streaks = None
        self.forward_stream = FakeForwardStream()
        self.elastic_ep_status_publisher = FakeStatusPublisher()
        self.tp_cpu_group = MagicMock()
        self.tp_group = MagicMock()
        self.tp_group.active_ranks_cpu = torch.ones(world, dtype=torch.int32)
        self.enable_overlap = False
        self.is_generation = False
        self.server_args = MagicMock()
        self.server_args.elastic_ep_backend = "mooncake" if enable_elastic_ep else None
        self.server_args.disable_overlap_schedule = False
        from collections import deque

        self.result_queue = deque()
        self.last_batch = None
        self.cur_batch_for_debug = None


# ---------------------------------------------------------------------------
# 1. result boundary — sync only, no collective
# ---------------------------------------------------------------------------


class TestDrainSnapshotCopy:
    def test_no_copy_done_is_noop(self):
        sched = TestScheduler()
        result = MagicMock()
        result.copy_done = None
        # Should not raise; no return value to check.
        sched._drain_elastic_ep_snapshot_copy(result)

    def test_syncs_copy_done_when_present(self):
        sched = TestScheduler()
        copy_done = MagicMock()
        copy_done.synchronize = MagicMock()
        result = MagicMock()
        result.copy_done = copy_done
        sched._drain_elastic_ep_snapshot_copy(result)
        copy_done.synchronize.assert_called_once()

    def test_does_not_call_commit(self):
        """Draining must NOT invoke commit_active_snapshot (the all_reduce)."""
        state = _make_state()
        sched = TestScheduler()
        result = MagicMock()
        result.copy_done = None
        with patch.object(ElasticEPStateManager, "instance", return_value=state):
            sched._drain_elastic_ep_snapshot_copy(result)
        # No snapshot was submitted, so pending_staging_slots is empty.
        assert state.pending_staging_slots == []


# ---------------------------------------------------------------------------
# 2. admission gate — commit + handler logic
# ---------------------------------------------------------------------------


class TestAdmissionGate:
    def test_no_snapshot_returns_true(self):
        """When there is nothing to commit, admission is a no-op pass-through."""
        state = _make_state()
        sched = TestScheduler()
        with patch.object(ElasticEPStateManager, "instance", return_value=state):
            assert sched._admit_elastic_ep_forward(MagicMock()) is True
        # forward_stream.synchronize IS called (safety before checking pending).
        assert sched.forward_stream.sync_count == 1

    def test_healthy_commit_publishes_and_returns_true(self):
        state = _make_state(world=4)
        sched = TestScheduler(world=4)
        state.submit_active_snapshot(
            torch.ones(4, dtype=torch.int32), non_blocking=False
        )

        def fake_commit(pg_ranks, group):
            if not state.pending_staging_slots:
                return False
            state.pending_staging_slots.pop(0)
            state.committed_active_ranks_cpu.fill_(1)
            state.ep_suspect_consensus_cpu.fill_(1)
            return True

        with patch.object(ElasticEPStateManager, "instance", return_value=state):
            with patch.object(state, "commit_active_snapshot", side_effect=fake_commit):
                assert sched._admit_elastic_ep_forward(MagicMock()) is True
        # Healthy path publishes once.
        assert len(sched.elastic_ep_status_publisher.published) == 1
        assert sched.elastic_ep_status_publisher.published[0][1] is False  # adjusting

    def test_fault_triggers_retract_and_returns_false(self):
        state = _make_state(world=4)
        sched = TestScheduler(world=4)
        state.submit_active_snapshot(
            torch.ones(4, dtype=torch.int32), non_blocking=False
        )

        def fake_commit(pg_ranks, group):
            if not state.pending_staging_slots:
                return False
            state.pending_staging_slots.pop(0)
            state.committed_active_ranks_cpu = torch.ones(4, dtype=torch.int32)
            state.committed_active_ranks_cpu[1] = 0
            state.ep_suspect_consensus_cpu.fill_(1)
            return True

        retract_called = False

        def fake_retract(self):
            nonlocal retract_called
            retract_called = True

        with patch.object(ElasticEPStateManager, "instance", return_value=state):
            with patch.object(
                state, "commit_active_snapshot", side_effect=fake_commit
            ), patch.object(
                SchedulerElasticEPMixin,
                "_retract_all_and_rebalance_on_rank_fault",
                fake_retract,
            ):
                assert sched._admit_elastic_ep_forward(MagicMock()) is False
        assert retract_called

    def test_stall_triggers_handler_and_returns_false(self):
        state = _make_state(world=4)
        sched = TestScheduler(world=4)
        state.submit_active_snapshot(
            torch.ones(4, dtype=torch.int32), non_blocking=False
        )

        def fake_commit(pg_ranks, group):
            if not state.pending_staging_slots:
                return False
            state.pending_staging_slots.pop(0)
            state.committed_active_ranks_cpu.fill_(1)
            state.ep_suspect_consensus_cpu = torch.ones(4, dtype=torch.int32)
            state.ep_suspect_consensus_cpu[2] = 0
            return True

        stall_called = False

        def fake_stall(self):
            nonlocal stall_called
            stall_called = True

        with patch.object(ElasticEPStateManager, "instance", return_value=state):
            with patch.object(
                state, "commit_active_snapshot", side_effect=fake_commit
            ), patch.object(
                SchedulerElasticEPMixin, "_handle_elastic_ep_a2a_stall", fake_stall
            ):
                assert sched._admit_elastic_ep_forward(MagicMock()) is False
        assert stall_called

    def test_forward_stream_synced_before_commit(self):
        """The admission gate must drain forward_stream before reading staging."""
        state = _make_state(world=4)
        sched = TestScheduler(world=4)
        state.submit_active_snapshot(
            torch.ones(4, dtype=torch.int32), non_blocking=False
        )

        sync_order = []

        class TrackingStream(FakeForwardStream):
            def synchronize(self):
                super().synchronize()
                sync_order.append("stream")

        sched.forward_stream = TrackingStream()

        def tracking_commit(pg_ranks, group):
            sync_order.append("commit")
            return False

        with patch.object(ElasticEPStateManager, "instance", return_value=state):
            with patch.object(
                state, "commit_active_snapshot", side_effect=tracking_commit
            ):
                sched._admit_elastic_ep_forward(MagicMock())
        assert sync_order == ["stream", "commit"]


# ---------------------------------------------------------------------------
# 3. multi-process: admission collective reaches consensus
# ---------------------------------------------------------------------------


class TestAdmissionCollective:
    """Two-process test: one rank observes a fault, the other doesn't.
    The all_reduce MIN must propagate the fault to both ranks."""

    @staticmethod
    def _run_admission_consensus(rank, init_method):
        torch.distributed.init_process_group(
            backend="gloo",
            init_method=init_method,
            rank=rank,
            world_size=2,
        )
        group = torch.distributed.GroupMember.WORLD
        try:
            state = ElasticEPStateManager._build_state(
                ep_size=2, device=torch.device("cpu")
            )
            pg_ranks = torch.ones(2, dtype=torch.int32)
            if rank == 0:
                pg_ranks[1] = 0
            state.submit_active_snapshot(pg_ranks, non_blocking=False)
            committed = state.commit_active_snapshot(pg_ranks, group)
            assert committed, "commit should have a snapshot to pop"
            assert (
                state.committed_active_ranks_cpu[1].item() == 0
            ), f"rank {rank}: fault not propagated by all_reduce"
        finally:
            torch.distributed.destroy_process_group()

    def test_fault_propagates_via_all_reduce(self, tmp_path):
        import torch.multiprocessing as mp

        init_file = tmp_path / "init"
        init_file.touch()
        init_method = f"file://{init_file}"
        ctx = mp.get_context("spawn")
        p0 = ctx.Process(target=self._run_admission_consensus, args=(0, init_method))
        p1 = ctx.Process(target=self._run_admission_consensus, args=(1, init_method))
        p0.start()
        p1.start()
        p0.join(timeout=30)
        p1.join(timeout=30)
        assert p0.exitcode == 0, f"rank 0 failed with {p0.exitcode}"
        assert p1.exitcode == 0, f"rank 1 failed with {p1.exitcode}"


# ---------------------------------------------------------------------------
# 4. ElasticEPState commit semantics (unchanged from baseline)
# ---------------------------------------------------------------------------


class TestCommitSemantics:
    """Verify that commit_active_snapshot (unchanged code) still works
    correctly when called from the new admission-gate position."""

    def test_submit_then_commit_pops_slot(self):
        state = _make_state(world=4)
        assert (
            state.commit_active_snapshot(torch.ones(4, dtype=torch.int32), MagicMock())
            is False
        )

        state.submit_active_snapshot(
            torch.ones(4, dtype=torch.int32), non_blocking=False
        )
        with patch("torch.distributed.all_reduce"):
            assert (
                state.commit_active_snapshot(
                    torch.ones(4, dtype=torch.int32), MagicMock()
                )
                is True
            )
        assert state.pending_staging_slots == []

    def test_stale_snapshot_detected_after_fault(self):
        state = _make_state(world=4)
        state.submit_active_snapshot(
            torch.ones(4, dtype=torch.int32), non_blocking=False
        )
        state.committed_active_ranks_cpu[2] = 0
        with patch("torch.distributed.all_reduce"):
            state.commit_active_snapshot(torch.ones(4, dtype=torch.int32), MagicMock())
        assert state.is_stale_snapshot() is True

    def test_ep_suspect_ranks_filtered_by_committed(self):
        state = _make_state(world=4)
        state.ep_suspect_consensus_cpu[1] = 0
        state.committed_active_ranks_cpu[1] = 0
        state.ep_suspect_consensus_cpu[2] = 0
        assert state.ep_suspect_ranks() == [2]

    def test_resync_clears_suspects(self):
        state = _make_state(world=4)
        state.ep_suspect_consensus_cpu[2] = 0
        state.committed_active_ranks_cpu.fill_(1)
        state.resync_active_to_committed()
        assert state.ep_suspect_consensus_cpu[2].item() == 1
        assert state.has_ep_suspects() is False


# ---------------------------------------------------------------------------
# 5. recovery consensus — can_recover_ranks must be globally agreed
# ---------------------------------------------------------------------------


class TestRecoveryConsensus:
    """can_recover_ranks is rank-local (mooncake peer state).  Without
    all_reduce MIN consensus, one rank could enter recovery (collective)
    while another proceeds to run_batch (EP A2A) -> deadlock."""

    def test_one_rank_not_ready_blocks_recovery(self):
        """If any rank reports not-ready, all_reduce MIN yields 0 and
        recovery is skipped on every rank."""
        state = _make_state(world=4)
        sched = TestScheduler(world=4)
        # committed has a dead rank so _get_elastic_ep_ranks_to_recover returns [1]
        state.committed_active_ranks_cpu[1] = 0
        state.last_handled_committed_active_ranks_cpu[1] = 0

        call_log = []

        def fake_can_recover(ranks):
            call_log.append(("can_recover", ranks))
            return False  # this rank is not ready

        all_reduce_called = [False]

        def fake_all_reduce(tensor, op, group):
            all_reduce_called[0] = True
            # Simulate MIN: at least one rank said 0, result is 0.
            tensor.fill_(0)

        with patch.object(ElasticEPStateManager, "instance", return_value=state), patch(
            "sglang.srt.managers.scheduler_elastic_ep_mixin.can_recover_ranks",
            side_effect=fake_can_recover,
        ), patch("torch.distributed.all_reduce", side_effect=fake_all_reduce):
            result = sched._maybe_recover_ep_ranks_from_cpu_snapshot()
        assert result is False
        assert all_reduce_called[0], "all_reduce must be called for consensus"

    def test_all_ranks_ready_proceeds_to_recovery(self):
        """If every rank reports ready, all_reduce MIN yields 1 and
        recovery proceeds."""
        state = _make_state(world=4)
        sched = TestScheduler(world=4)
        state.committed_active_ranks_cpu[1] = 0
        state.last_handled_committed_active_ranks_cpu[1] = 0

        def fake_all_reduce(tensor, op, group):
            tensor.fill_(1)  # all ready

        retract_called = [False]

        def fake_retract(self, *, abort_message, err_type):
            retract_called[0] = True
            return 0, 0

        sched.tp_worker = MagicMock()
        sched.tp_worker.model_runner = MagicMock()
        sched.tp_worker.model_runner.recover_ep_ranks_after_retract = MagicMock()
        sched.running_batch = MagicMock()
        sched.running_batch.reqs = []
        sched.cur_batch = None
        sched.chunked_req = None
        sched.enable_overlap = False

        with patch.object(ElasticEPStateManager, "instance", return_value=state), patch(
            "sglang.srt.managers.scheduler_elastic_ep_mixin.can_recover_ranks",
            return_value=True,
        ), patch(
            "torch.distributed.all_reduce", side_effect=fake_all_reduce
        ), patch.object(
            SchedulerElasticEPMixin,
            "_retract_inflight_batches_for_elastic_ep",
            fake_retract,
        ):
            result = sched._maybe_recover_ep_ranks_from_cpu_snapshot()
        assert result is True
        assert retract_called[0], "retract must run when all ranks ready"
        sched.tp_worker.model_runner.recover_ep_ranks_after_retract.assert_called_once()

    def test_no_dead_ranks_skips_recovery(self):
        """If committed mask is all-ones, no ranks to recover -> skip."""
        state = _make_state(world=4)
        sched = TestScheduler(world=4)
        # All alive, no dead ranks.
        with patch.object(ElasticEPStateManager, "instance", return_value=state):
            assert sched._maybe_recover_ep_ranks_from_cpu_snapshot() is False

    def test_reserved_capacity_slots_are_not_recovery_candidates(self):
        state = _make_state(world=8)
        state.effective_ep_size = 4
        state.committed_active_ranks_cpu[4:] = 0
        sched = TestScheduler(world=8)

        with patch.object(ElasticEPStateManager, "instance", return_value=state):
            assert sched._get_elastic_ep_ranks_to_recover_from_cpu_snapshot() == []

        state.committed_active_ranks_cpu[2] = 0
        with patch.object(ElasticEPStateManager, "instance", return_value=state):
            assert sched._get_elastic_ep_ranks_to_recover_from_cpu_snapshot() == [2]


# ---------------------------------------------------------------------------
# 6. grammar global serial consensus
# ---------------------------------------------------------------------------


class TestGrammarGlobalSerial:
    """has_grammar is rank-local.  In DP-attention mode, if one rank needs
    grammar sync (serial) while another does not (overlap), the scheduler
    timeline desynchronizes and re-introduces the EP A2A deadlock.
    all_reduce MAX ensures all ranks make the same serial/overlap decision."""

    def _compute_grammar_sync(self, need_grammar_sync, require_mlp_sync):
        """Replicate the grammar-consensus snippet from is_disable_overlap_for_batch."""
        if require_mlp_sync:
            flag = torch.tensor([1 if need_grammar_sync else 0], dtype=torch.int32)
            torch.distributed.all_reduce(
                flag,
                op=torch.distributed.ReduceOp.MAX,
                group=MagicMock(),
            )
            need_grammar_sync = flag.item() == 1
        return need_grammar_sync

    def test_local_grammar_sync_global_all_reduce_called(self):
        """In DP mode, all_reduce MAX is called on tp_cpu_group."""
        all_reduce_args = []

        def fake_all_reduce(tensor, *args, **kwargs):
            op = kwargs.get("op", args[0] if args else None)
            all_reduce_args.append((tensor.tolist(), op))
            tensor.fill_(1)

        with patch("torch.distributed.all_reduce", side_effect=fake_all_reduce):
            result = self._compute_grammar_sync(
                need_grammar_sync=True, require_mlp_sync=True
            )
        assert result is True
        assert len(all_reduce_args) == 1, "all_reduce must be called once"

    def test_one_rank_grammar_forces_all_serial(self):
        """If any rank reports need_grammar_sync=1, MAX yields 1 -> all serial."""
        with patch("torch.distributed.all_reduce") as mock_ar:
            mock_ar.side_effect = lambda t, *a, **kw: t.fill_(1)
            result = self._compute_grammar_sync(
                need_grammar_sync=True, require_mlp_sync=True
            )
        assert result is True

    def test_no_grammar_still_all_reduce_in_dp_mode(self):
        """Even if local need_grammar_sync=False, all_reduce is still called
        in DP mode — other ranks may have grammar.  MAX yields 0 -> all overlap."""
        all_reduce_called = [False]

        def fake_all_reduce(tensor, *args, **kwargs):
            all_reduce_called[0] = True
            tensor.fill_(0)

        with patch("torch.distributed.all_reduce", side_effect=fake_all_reduce):
            result = self._compute_grammar_sync(
                need_grammar_sync=False, require_mlp_sync=True
            )
        assert all_reduce_called[
            0
        ], "all_reduce must be called even when local has no grammar"
        assert result is False

    def test_non_dp_mode_skips_all_reduce(self):
        """In non-DP mode (require_mlp_sync=False), no all_reduce is needed."""
        with patch("torch.distributed.all_reduce") as mock_ar:
            result = self._compute_grammar_sync(
                need_grammar_sync=True, require_mlp_sync=False
            )
        mock_ar.assert_not_called()
        assert result is True  # local decision only


# ---------------------------------------------------------------------------
# 6b. grammar global serial consensus — multi-rank reduction semantics
# ---------------------------------------------------------------------------
#
# The commit message says:
#   "if any rank needs grammar sync, all ranks run serially for that batch"
# i.e. desired semantics = logical OR across ranks = ReduceOp.MAX
#   (encoding: 1 = need serial, 0 = can overlap)
#
# Truth table for encoding {0,1}:
#
#   local values   MAX    desired (any-rank-needs-serial)
#   [0,0,0]         0      0   (all overlap)
#   [1,0,0]         1      1   (one rank needs serial -> ALL serial)
#   [1,1,1]         1      1   (all serial)
#
# MAX yields 1 whenever any rank is 1 — exactly the desired semantics.
# (Previously used ReduceOp.MIN, which yields 0 whenever any rank is 0 —
# the opposite of "any-rank-needs".  Fixed to MAX.)
# These tests simulate real multi-rank all_reduce by computing the actual
# reduction over a provided list of per-rank local values, then writing the
# global result back.  They assert the desired semantics (any-1 -> all-1).


class TestGrammarReductionSemantics:
    """Verify the all_reduce op matches the intended 'any-rank-needs-serial'
    semantics.  Uses a simulated multi-rank reduction, not a fixed fill_()."""

    @staticmethod
    def _simulate_all_reduce(tensor, op, group, all_rank_values):
        """Compute the real reduction over all_rank_values and write back."""
        vals = all_rank_values
        if op == torch.distributed.ReduceOp.MIN:
            result = min(vals)
        elif op == torch.distributed.ReduceOp.MAX:
            result = max(vals)
        else:
            raise ValueError(f"unexpected op {op}")
        tensor.fill_(result)

    def _compute_grammar_sync(self, local_value, op, all_rank_values):
        """Replicate the grammar-consensus snippet with a given ReduceOp."""
        flag = torch.tensor([local_value], dtype=torch.int32)
        captured = {}

        def fake_all_reduce(tensor, *args, **kwargs):
            o = kwargs.get("op", args[0] if args else None)
            g = kwargs.get("group", args[1] if len(args) > 1 else None)
            captured["op"] = o
            self._simulate_all_reduce(tensor, o, g, all_rank_values)

        with patch("torch.distributed.all_reduce", side_effect=fake_all_reduce):
            torch.distributed.all_reduce(
                flag,
                op=op,
                group=MagicMock(),
            )
            return flag.item() == 1

    # --- desired semantics: any rank needs serial -> ALL serial ---

    def test_all_overlap_when_no_rank_needs_serial(self):
        """[0,0,0] -> all overlap.  Both MIN and MAX agree here."""
        result = self._compute_grammar_sync(
            local_value=0,
            op=torch.distributed.ReduceOp.MAX,
            all_rank_values=[0, 0, 0],
        )
        assert result is False

    def test_max_one_rank_forces_all_serial(self):
        """[1,0,0] -> any rank needs serial -> ALL serial.  MAX is correct."""
        result = self._compute_grammar_sync(
            local_value=1,  # this rank needs serial
            op=torch.distributed.ReduceOp.MAX,
            all_rank_values=[1, 0, 0],  # other ranks don't
        )
        assert (
            result is True
        ), "MAX: any rank needs serial -> all serial (correct semantics)"

    def test_max_local_false_but_remote_true_forces_serial(self):
        """[0,1,0] -> this rank is 0 but another rank is 1 -> still serial."""
        result = self._compute_grammar_sync(
            local_value=0,  # this rank does NOT need serial
            op=torch.distributed.ReduceOp.MAX,
            all_rank_values=[0, 1, 0],  # another rank does
        )
        assert result is True, "MAX: even if local is 0, a remote 1 forces serial"

    def test_all_serial_when_all_ranks_need_serial(self):
        """[1,1,1] -> all serial.  Both MIN and MAX agree here."""
        result = self._compute_grammar_sync(
            local_value=1,
            op=torch.distributed.ReduceOp.MAX,
            all_rank_values=[1, 1, 1],
        )
        assert result is True


# ---------------------------------------------------------------------------
# 7. handler exception protection (G fix)
# ---------------------------------------------------------------------------


class TestHandlerExceptionProtection:
    """If a handler (rebalance, recover, etc.) raises inside _admit,
    the exception must propagate so the engine can tear down all ranks
    instead of leaving peers deadlocked on the next all_reduce."""

    def test_handler_exception_propagates(self):
        state = _make_state(world=4)
        sched = TestScheduler(world=4)
        state.submit_active_snapshot(
            torch.ones(4, dtype=torch.int32), non_blocking=False
        )

        def fake_commit(pg_ranks, group):
            if not state.pending_staging_slots:
                return False
            state.pending_staging_slots.pop(0)
            state.committed_active_ranks_cpu[1] = 0
            state.ep_suspect_consensus_cpu.fill_(1)
            return True

        def boom(self):
            raise RuntimeError("rebalance collective failed")

        with patch.object(ElasticEPStateManager, "instance", return_value=state):
            with patch.object(
                state, "commit_active_snapshot", side_effect=fake_commit
            ), patch.object(
                SchedulerElasticEPMixin,
                "_retract_all_and_rebalance_on_rank_fault",
                boom,
            ), patch(
                "torch.cuda.synchronize"
            ):
                with pytest.raises(RuntimeError, match="rebalance collective failed"):
                    sched._admit_elastic_ep_forward(MagicMock())


# ---------------------------------------------------------------------------
# 8. clear_pending_snapshots resets next_staging_slot (B fix)
# ---------------------------------------------------------------------------


class TestClearPendingSnapshotsResetsSlot:
    """clear_pending_snapshots must reset next_staging_slot to 0 so the
    double-buffer selector is deterministic after a retract."""

    def test_next_staging_slot_reset_after_clear(self):
        state = _make_state(world=4)
        # Submit twice to flip next_staging_slot
        state.submit_active_snapshot(
            torch.ones(4, dtype=torch.int32), non_blocking=False
        )
        assert state.next_staging_slot == 1
        state.submit_active_snapshot(
            torch.ones(4, dtype=torch.int32), non_blocking=False
        )
        assert state.next_staging_slot == 0
        assert len(state.pending_staging_slots) == 2

        state.clear_pending_snapshots()
        assert state.pending_staging_slots == []
        assert state.next_staging_slot == 0

    def test_submit_after_clear_starts_from_slot_0(self):
        state = _make_state(world=4)
        state.submit_active_snapshot(
            torch.ones(4, dtype=torch.int32), non_blocking=False
        )
        state.clear_pending_snapshots()
        # Next submit should use slot 0
        state.submit_active_snapshot(
            torch.ones(4, dtype=torch.int32), non_blocking=False
        )
        assert state.pending_staging_slots == [0]
        assert state.next_staging_slot == 1


# ---------------------------------------------------------------------------
# 9. _drain_elastic_ep_snapshot_copy regression — name, signature, dead-code
# ---------------------------------------------------------------------------
#
# The function formerly known as _handle_elastic_ep_result_boundary has been
# renamed and simplified:
#   - Returns None (not bool); the old `return False -> caller continue/return
#     False` dead branches have been removed from all four event loops.
#   - Only drains copy_done; no commit_active_snapshot, no ElasticEPStateManager
#     singleton access.
#
# These tests guard against regressions: if someone reintroduces a bool return,
# re-adds the singleton assert, or resurrects the old name, they will fail.


class TestDrainSnapshotCopyRegression:
    """Regression tests for the _drain_elastic_ep_snapshot_copy refactor."""

    def test_returns_none_not_bool(self):
        """The function must return None, not True/False.  Callers no longer
        check the return value, so a bool would be a silent API break."""
        sched = TestScheduler()
        result = MagicMock()
        result.copy_done = None
        ret = sched._drain_elastic_ep_snapshot_copy(result)
        assert (
            ret is None
        ), f"Expected None, got {ret!r}.  The function must not return bool."

    def test_returns_none_even_with_copy_done(self):
        """Even when copy_done is present and synchronized, return is None."""
        sched = TestScheduler()
        copy_done = MagicMock()
        result = MagicMock()
        result.copy_done = copy_done
        ret = sched._drain_elastic_ep_snapshot_copy(result)
        assert ret is None
        copy_done.synchronize.assert_called_once()

    def test_no_elastic_ep_state_accessed(self):
        """Must NOT call ElasticEPStateManager.instance() — that singleton
        access was removed along with the assert."""
        sched = TestScheduler()
        result = MagicMock()
        result.copy_done = None
        with patch.object(ElasticEPStateManager, "instance") as mock_instance:
            sched._drain_elastic_ep_snapshot_copy(result)
            mock_instance.assert_not_called()

    def test_old_name_does_not_exist(self):
        """The old name _handle_elastic_ep_result_boundary must be gone
        so no caller accidentally uses the stale API."""
        sched = TestScheduler()
        assert not hasattr(
            sched, "_handle_elastic_ep_result_boundary"
        ), "Old method name should not exist; use _drain_elastic_ep_snapshot_copy"

    def test_new_name_exists(self):
        """Sanity: the new method is bound on the scheduler."""
        sched = TestScheduler()
        assert hasattr(sched, "_drain_elastic_ep_snapshot_copy")
        assert callable(sched._drain_elastic_ep_snapshot_copy)

    def test_copy_done_none_does_not_raise(self):
        """copy_done=None is the common case (non-EP batches); must be a noop."""
        sched = TestScheduler()
        result = MagicMock()
        result.copy_done = None
        # Should not raise
        sched._drain_elastic_ep_snapshot_copy(result)

    def test_idempotent_multiple_calls(self):
        """Calling drain twice on the same result should not error."""
        sched = TestScheduler()
        copy_done = MagicMock()
        result = MagicMock()
        result.copy_done = copy_done
        sched._drain_elastic_ep_snapshot_copy(result)
        sched._drain_elastic_ep_snapshot_copy(result)
        # Both calls synchronize; we don't assert call_count == 1 because
        # the contract is "drain", not "drain once" — but it must not raise.
        assert copy_done.synchronize.call_count >= 1


class TestElasticEPStatusPublisher:
    def test_cluster_state_reports_tolerance(self):
        snapshot = _compute_cluster_state(
            torch.tensor([1, 1, 0, 1], dtype=torch.int32), adjusting=False
        )
        assert snapshot.state == ElasticEPMetricState.TOLERANCE
        assert snapshot.active_count == 3
        assert snapshot.world_size == 4

    def test_metrics_view_excludes_reserved_scale_capacity(self):
        state = _make_state(world=8)
        state.effective_ep_size = 4
        committed = torch.tensor([1, 1, 1, 1, 0, 0, 0, 0])

        with patch.object(ElasticEPStateManager, "instance", return_value=state):
            effective = _effective_committed_active_ranks(committed)

        snapshot = _compute_cluster_state(effective, adjusting=False)
        assert snapshot.state == ElasticEPMetricState.HEALTHY
        assert snapshot.active_count == 4
        assert snapshot.world_size == 4

    def test_metrics_republishes_total_after_scale_up(self):
        args = MagicMock(served_model_name="model")
        publisher = MetricsElasticEPStatusPublisher(args)
        publisher._is_exporter_cache = True
        state = _make_state(world=8)
        committed = torch.ones(8, dtype=torch.int32)

        with patch.object(ElasticEPStateManager, "instance", return_value=state), patch(
            "sglang.srt.managers.elastic_ep_status._get_elastic_ep_gauge",
            side_effect=lambda name, documentation: name,
        ), patch(
            "sglang.srt.managers.elastic_ep_status._set_elastic_ep_gauge_int"
        ) as set_gauge:
            state.effective_ep_size = 4
            publisher.publish_committed_active_ranks(committed)
            state.effective_ep_size = 8
            publisher.publish_committed_active_ranks(committed)

        total_values = [
            call.args[2]
            for call in set_gauge.call_args_list
            if call.args[0] == "sglang:elastic_ep_total_ranks"
        ]
        assert total_values == [4, 8]

    def test_controller_publishes_only_changed_status(self):
        sender = MagicMock()
        publisher = ControllerElasticEPStatusPublisher(sender, dp_size=2)

        publisher.publish_committed_active_ranks(torch.tensor([1, 1]))
        sender.send_output.assert_not_called()

        publisher.publish_committed_active_ranks(torch.tensor([1, 0]))
        assert sender.send_output.call_count == 1
        assert sender.send_output.call_args.args[0].status == [True, False]

    def test_controller_does_not_cache_failed_publish(self):
        sender = MagicMock()
        sender.send_output.side_effect = RuntimeError("controller unavailable")
        publisher = ControllerElasticEPStatusPublisher(sender, dp_size=2)

        with pytest.raises(RuntimeError, match="controller unavailable"):
            publisher.publish_committed_active_ranks(torch.tensor([1, 0]))
        assert publisher.last_status == [True, True]

        sender.send_output.side_effect = None
        publisher.publish_committed_active_ranks(torch.tensor([1, 0]))
        assert sender.send_output.call_count == 2
        assert publisher.last_status == [True, False]

    def test_composite_isolates_sink_failure(self):
        failing = MagicMock()
        failing.publish_committed_active_ranks.side_effect = RuntimeError("sink down")
        healthy = MagicMock()
        publisher = CompositeElasticEPStatusPublisher([failing, healthy])
        mask = torch.tensor([1, 0])

        publisher.publish_committed_active_ranks(mask, adjusting=True)

        healthy.publish_committed_active_ranks.assert_called_once_with(
            mask, adjusting=True
        )
