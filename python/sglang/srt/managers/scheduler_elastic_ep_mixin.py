from __future__ import annotations

import logging
from http import HTTPStatus
from typing import TYPE_CHECKING, Dict, List, Optional

import torch

from sglang.srt.elastic_ep.elastic_ep import (
    ElasticEPStateManager,
    can_recover_ranks,
)
from sglang.srt.environ import envs
from sglang.srt.managers.io_struct import AbortReq
from sglang.srt.managers.schedule_batch import FINISH_ABORT

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import ScheduleBatch
    from sglang.srt.managers.scheduler import EmbeddingBatchResult, Scheduler
    from sglang.srt.managers.utils import GenerationBatchResult

logger = logging.getLogger(__name__)


class SchedulerElasticEPMixin:
    _elastic_ep_adjusting: bool = False
    # Per-rank count of CONSECUTIVE a2a-stall events (lazy `{}` on first use;
    # one scheduler per process). Fed only by consensus-derived suspect lists,
    # so every rank holds an identical dict and escalates on the same boundary.
    _elastic_ep_stall_streaks: Optional[Dict[int, int]] = None

    def _drain_elastic_ep_snapshot_copy(
        self: Scheduler,
        result: GenerationBatchResult | EmbeddingBatchResult,
    ) -> None:
        """Drain the async D2H copy that submit_active_snapshot recorded on
        forward_stream before run_batch.

        This is the only thing the result boundary does now.  The all_reduce
        consensus (commit_active_snapshot) has moved to _admit_elastic_ep_forward,
        which runs at the run_batch entry — a point where every DP rank has
        already converged on the same batch.  Running all_reduce here (in
        pop_and_process, where grammar ranks pop early while overlap ranks are
        still mid-run_batch) placed a collective on a desynchronization seam and
        deadlocked with the EP a2a kernels inside run_batch.
        """
        if result.copy_done is not None:
            result.copy_done.synchronize()

    def _admit_elastic_ep_forward(self: Scheduler, batch: ScheduleBatch) -> bool:
        """Global consensus gate at run_batch entry.

        Runs commit_active_snapshot (all_reduce MIN) here — after every DP rank
        has selected the same batch, before any EP a2a kernel launches — so the
        collective can never straddle a desync seam. If the consensus surfaces a
        fault, a2a stall, or recovery, the matching handler retracts and we
        return False so the caller skips run_batch for this iteration.
        """
        elastic_ep_state = ElasticEPStateManager.instance()
        assert elastic_ep_state is not None
        # In the overlap path the previous batch's submit_active_snapshot used a
        # non-blocking D2H copy on forward_stream, and its copy_done has not
        # been synchronized yet (pop_and_process runs after run_batch on this
        # path). Drain forward_stream so the staging buffer is valid before
        # all_reduce reads it. On the grammar/serial path copy_done.synchronize
        # in pop_and_process already drained, so this is a no-op there.
        self.forward_stream.synchronize()
        if not elastic_ep_state.commit_active_snapshot(
            self.tp_group.active_ranks_cpu, self.tp_cpu_group
        ):
            return True
        try:
            if elastic_ep_state.is_stale_snapshot():
                self._retract_all_and_rebalance_on_rank_fault()
                return False
            if elastic_ep_state.has_ep_suspects():
                self._handle_elastic_ep_a2a_stall()
                return False
            if self._maybe_recover_ep_ranks_from_cpu_snapshot():
                return False
        except Exception:
            # A handler failure (e.g. rebalance collective error) leaves this
            # rank in an unrecoverable state.  Drain GPU so error logs and
            # core dumps are coherent, then re-raise — the scheduler process
            # will exit and the engine manager should tear down all peers to
            # prevent them from deadlocking on the next all_reduce.
            torch.cuda.synchronize()
            logger.exception(
                "Elastic EP handler failed in admission gate; "
                "re-raising to trigger engine shutdown"
            )
            raise
        if self._elastic_ep_stall_streaks:
            self._elastic_ep_stall_streaks = {}
        self._publish_active_ranks_from_committed_snapshot()
        return True

    def _publish_active_ranks_from_committed_snapshot(self: Scheduler):
        elastic_ep_state = ElasticEPStateManager.instance()
        assert elastic_ep_state is not None

        # `_elastic_ep_adjusting` is toggled only around fault/recovery handlers.
        # It lets metrics expose the transient ADJUST state without new collectives.
        self.elastic_ep_status_publisher.publish_committed_active_ranks(
            elastic_ep_state.committed_active_ranks_cpu,
            adjusting=self._elastic_ep_adjusting,
        )

    def _publish_elastic_ep_status_on_ready(self: Scheduler):
        """Publish initial cluster state when the scheduler is ready to serve."""
        if self.server_args.elastic_ep_backend is None:
            return
        elastic_ep_state = ElasticEPStateManager.instance()
        if elastic_ep_state is None:
            return

        # First publish must follow the same commit path as batch-time updates.
        # `committed_active_ranks_cpu` starts as an optimistic all-ones default;
        # without a synchronous commit here, metrics could report HEALTHY before
        # tp_group active-rank consensus (all_reduce MIN) has run.
        elastic_ep_state.submit_active_snapshot(
            self.tp_group.active_ranks_cpu,
            non_blocking=False,
        )
        elastic_ep_state.commit_active_snapshot(
            self.tp_group.active_ranks_cpu,
            self.tp_cpu_group,
        )
        self._publish_active_ranks_from_committed_snapshot()
        logger.info("Elastic EP initial state is published at engine ready handshake.")

    def _get_elastic_ep_ranks_to_recover_from_cpu_snapshot(
        self: Scheduler,
    ) -> List[int]:
        elastic_ep_state = ElasticEPStateManager.instance()
        assert elastic_ep_state is not None

        # Capacity slots above effective_ep_size are reserved for JD's runtime
        # scale-up protocol. They are intentionally inactive and must never be
        # treated as failed ranks by the recovery path.
        effective_ep_size = (
            elastic_ep_state.effective_ep_size
            or elastic_ep_state.committed_active_ranks_cpu.numel()
        )
        return (
            torch.nonzero(
                elastic_ep_state.committed_active_ranks_cpu[:effective_ep_size] == 0,
                as_tuple=False,
            )
            .flatten()
            .tolist()
        )

    def _retract_inflight_batches_for_elastic_ep(
        self: Scheduler,
        *,
        abort_message: str,
        err_type: str,
    ) -> tuple[int, int]:
        if self.enable_overlap:
            self.result_queue.clear()
        ElasticEPStateManager.instance().clear_pending_snapshots()
        torch.cuda.synchronize()

        max_retraction = envs.SGLANG_ELASTIC_EP_MAX_RETRACTION.get()
        abort_reason = FINISH_ABORT(
            message=abort_message.format(max_retraction=max_retraction),
            status_code=HTTPStatus.SERVICE_UNAVAILABLE,
            err_type=err_type,
        )

        # In decode, cur_batch_for_debug IS running_batch (same object). For a freshly
        # launched prefill, cur_batch_for_debug is a separate batch whose reqs are not
        # yet merged into running_batch — drain both in that case.
        batches = [self.running_batch]
        if (
            self.cur_batch_for_debug is not None
            and self.cur_batch_for_debug is not self.running_batch
        ):
            batches.append(self.cur_batch_for_debug)

        requeued_count = 0
        aborted_count = 0
        for batch in batches:
            for idx, req in enumerate(batch.reqs):
                batch.release_req(idx, 0, self.server_args)
                if req.retraction_count > max_retraction:
                    req.finished_reason = abort_reason
                    self.ipc_channels.send_to_tokenizer.send_output(
                        AbortReq(
                            finished_reason=abort_reason.to_json(),
                            rid=req.rid,
                            http_worker_ipc=req.http_worker_ipc,
                        ),
                        req,
                    )
                    aborted_count += 1
                else:
                    self._add_request_to_queue(req, is_retracted=True)
                    requeued_count += 1

        self.running_batch.filter_batch(keep_indices=[])
        self.last_batch = self.cur_batch_for_debug = None
        # Clear stale pointer to a chunked-prefill req we just released; the
        # next iteration's stash_chunked_request would otherwise dereference
        # its now-None req_pool_idx.
        self.chunked_req = None
        return requeued_count, aborted_count

    def _retract_all_and_rebalance_on_rank_fault(self: Scheduler):
        """Rank fault: drain GPU, retract in-flight decode reqs, rebalance."""
        # Enter transient ADJUST state before retraction/rebalance starts.
        self._elastic_ep_adjusting = True
        self._publish_active_ranks_from_committed_snapshot()
        elastic_ep_state = ElasticEPStateManager.instance()
        down_ranks = self._get_elastic_ep_ranks_to_recover_from_cpu_snapshot()
        # Attribution only: which ranks the EP a2a kernels also timed out on.
        ep_a2a_suspects = (
            torch.nonzero(
                elastic_ep_state.ep_suspect_consensus_cpu == 0, as_tuple=False
            )
            .flatten()
            .tolist()
        )
        logger.error(
            "Elastic EP rank fault detected. down_ranks=%s "
            "committed_active_ranks=%s ep_a2a_suspects=%s",
            down_ranks,
            elastic_ep_state.committed_active_ranks_cpu.tolist(),
            ep_a2a_suspects,
        )
        try:
            requeued_count, aborted_count = (
                self._retract_inflight_batches_for_elastic_ep(
                    abort_message=(
                        "Elastic EP rank fault; aborted after {max_retraction} retractions."
                    ),
                    err_type="ElasticEPRankFault",
                )
            )

            eplb_manager = self.tp_worker.model_runner.eplb_manager
            if eplb_manager is not None:
                gen = eplb_manager.rebalance()
                while True:
                    try:
                        next(gen)
                    except StopIteration:
                        break
            ElasticEPStateManager.instance().mark_snapshot_handled()
            # Propagate PG-confirmed deaths into the device mask the a2a
            # kernels read — the EP kernels may never have marked them on
            # their own. Safe: the retract helper above already
            # torch.cuda.synchronize()'d, so no in-flight kernel is reading
            # active_ranks.
            elastic_ep_state.resync_active_to_committed()
            # A confirmed fault invalidates the pre-fault stall pattern;
            # every rank resets in the same handler, so streaks stay
            # identical cluster-wide.
            self._elastic_ep_stall_streaks = {}
            logger.info(
                "Elastic EP rank fault handled. requeued=%s aborted=%s",
                requeued_count,
                aborted_count,
            )
        finally:
            # Exit ADJUST state even if fault handling raises unexpectedly.
            self._elastic_ep_adjusting = False
            self._publish_active_ranks_from_committed_snapshot()

    def _handle_elastic_ep_a2a_stall(self: Scheduler):
        """EP a2a timeout while PG probes report the suspects alive: the
        stalled step's outputs are corrupted, so retract them, but nobody
        died — revive the suspects and skip the EPLB rebalance. Escalate to
        the hard path only when the same rank is suspected in N consecutive
        stall events."""
        elastic_ep_state = ElasticEPStateManager.instance()
        # Consensus-derived (all_reduce MIN inside commit), hence identical on
        # every rank: streaks, escalation, and the branch taken below stay
        # deterministic, so no rank enters the rebalance collective alone.
        suspects = elastic_ep_state.ep_suspect_ranks()
        if self._elastic_ep_stall_streaks is None:
            self._elastic_ep_stall_streaks = {}
        streaks = self._elastic_ep_stall_streaks
        for rank in suspects:
            streaks[rank] = streaks.get(rank, 0) + 1
        # Consecutive semantics: a rank absent from this event loses its
        # streak entirely.
        for rank in list(streaks):
            if rank not in suspects:
                del streaks[rank]

        threshold = envs.SGLANG_ELASTIC_EP_STALL_ESCALATION_THRESHOLD.get()
        escalated = sorted(rank for rank in suspects if streaks[rank] >= threshold)
        if escalated:
            # Global-stall guard: escalating every remaining alive rank would
            # zero the whole committed mask and hand EPLB a zero-rank world
            # (div-by-zero) — a stall of ALL alive ranks is systemic (e.g.
            # cluster-wide backpressure), not simultaneous deaths that every
            # PG probe somehow missed. Keep retracting and leave the death
            # verdict to PG. Inputs are consensus-derived, so every rank
            # downgrades together.
            alive = int(elastic_ep_state.committed_active_ranks_cpu.sum().item())
            if len(escalated) >= alive:
                logger.error(
                    "Elastic EP a2a stall suspected ALL %s active ranks for "
                    "%s consecutive events; treating as global stall, not "
                    "deaths. stall_streaks=%s",
                    alive,
                    threshold,
                    streaks,
                )
                escalated = []
        if escalated:
            logger.error(
                "Elastic EP a2a stall escalated to rank fault after %s "
                "consecutive events. escalated=%s stall_streaks=%s",
                threshold,
                escalated,
                streaks,
            )
            # Force committed zeros BEFORE the hard handler: its down_ranks
            # derives from committed, and its mark_snapshot_handled absorbs
            # the forced zeros so the next boundary does not re-trigger. The
            # sticky committed latch then keeps the rank out until explicit
            # recovery.
            elastic_ep_state.committed_active_ranks_cpu[escalated] = 0
            self._retract_all_and_rebalance_on_rank_fault()
            return

        self._elastic_ep_adjusting = True
        self._publish_active_ranks_from_committed_snapshot()
        try:
            logger.error(
                "Elastic EP a2a stall detected (PG reports suspects alive). "
                "ep_suspects=%s stall_streaks=%s committed_active_ranks=%s",
                suspects,
                streaks,
                elastic_ep_state.committed_active_ranks_cpu.tolist(),
            )
            requeued_count, aborted_count = (
                self._retract_inflight_batches_for_elastic_ep(
                    abort_message=(
                        "Elastic EP a2a stall; aborted after "
                        "{max_retraction} retractions."
                    ),
                    err_type="ElasticEPA2AStall",
                )
            )
            # Resync strictly AFTER the retract helper: its
            # torch.cuda.synchronize drains any in-flight forward before we
            # mutate active_ranks. This revives the falsely-marked suspects
            # in the kernels' mask and clears the consumed suspect bits.
            elastic_ep_state.resync_active_to_committed()
            logger.info(
                "Elastic EP a2a stall handled. requeued=%s aborted=%s",
                requeued_count,
                aborted_count,
            )
            # committed/last_handled stay untouched: commit never dirtied
            # committed (EP marks do not reach it), so is_stale_snapshot()
            # stays False and no mark_snapshot_handled is needed.
        finally:
            # Ensure metrics leave ADJUST state on all code paths.
            self._elastic_ep_adjusting = False
            self._publish_active_ranks_from_committed_snapshot()

    def _maybe_recover_ep_ranks_from_cpu_snapshot(self: Scheduler) -> bool:
        ranks_to_recover = self._get_elastic_ep_ranks_to_recover_from_cpu_snapshot()
        if not ranks_to_recover:
            return False
        # can_recover_ranks queries local mooncake peer state; different ranks
        # may observe a recovering peer as ready at different times. If one
        # rank enters recovery (collective) while another proceeds to
        # run_batch (EP A2A), the recovery collective deadlocks. All_reduce
        # MIN on the existing tp_cpu_group ensures every rank makes the same
        # GO/NO-GO decision at the same admission boundary.
        local_ready = torch.tensor(
            [1 if can_recover_ranks(ranks_to_recover) else 0],
            dtype=torch.int32,
        )
        torch.distributed.all_reduce(
            local_ready,
            op=torch.distributed.ReduceOp.MIN,
            group=self.tp_cpu_group,
        )
        if local_ready.item() != 1:
            return False

        self._elastic_ep_adjusting = True
        self._publish_active_ranks_from_committed_snapshot()
        try:
            requeued_count, aborted_count = (
                self._retract_inflight_batches_for_elastic_ep(
                    abort_message=(
                        "Elastic EP rank recovery; aborted after "
                        "{max_retraction} retractions."
                    ),
                    err_type="ElasticEPRankRecovery",
                )
            )
            self.tp_worker.model_runner.recover_ep_ranks_after_retract(ranks_to_recover)
            # Recovery is the synchronized event through which a restarted
            # scheduler (whose streak dict starts empty) rejoins the group;
            # resetting on every recovery keeps all ranks' escalation state
            # identical afterwards.
            self._elastic_ep_stall_streaks = {}
            logger.info(
                "Elastic EP rank recovery handled. ranks=%s requeued=%s aborted=%s",
                ranks_to_recover,
                requeued_count,
                aborted_count,
            )
        finally:
            # Ensure metrics leave ADJUST state on all code paths.
            self._elastic_ep_adjusting = False
            self._publish_active_ranks_from_committed_snapshot()
        return True
