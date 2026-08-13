# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Elastic EP rank status publishing."""

import logging
from dataclasses import dataclass
from enum import IntEnum
from typing import Any, Optional

import torch

from sglang.srt.elastic_ep.elastic_ep import ElasticEPStateManager
from sglang.srt.environ import envs
from sglang.srt.managers.io_struct import ActiveRanksOutput
from sglang.srt.server_args import ServerArgs

logger = logging.getLogger(__name__)


class ElasticEPMetricState(IntEnum):
    # Numeric encoding intentionally ordered by severity.
    HEALTHY = 0
    TOLERANCE = 1
    ADJUST = 2
    DEAD = 3


@dataclass(frozen=True)
class ElasticEPClusterSnapshot:
    state: ElasticEPMetricState
    active_count: int
    world_size: int


def _compute_cluster_state(
    committed_active_ranks: torch.Tensor, adjusting: bool, min_serving_ranks: int = 1
) -> ElasticEPClusterSnapshot:
    world_size = committed_active_ranks.numel()
    active_count = int(committed_active_ranks.sum().item())

    # ADJUST is a transient control-plane state while fault/recovery handlers
    # run. active_count is still reported truthfully during ADJUST.
    if adjusting:
        state = ElasticEPMetricState.ADJUST
    elif active_count >= world_size:
        state = ElasticEPMetricState.HEALTHY
    elif active_count < min_serving_ranks:
        state = ElasticEPMetricState.DEAD
    else:
        state = ElasticEPMetricState.TOLERANCE

    return ElasticEPClusterSnapshot(
        state=state, active_count=active_count, world_size=world_size
    )


def _effective_committed_active_ranks(
    committed_active_ranks: torch.Tensor,
) -> torch.Tensor:
    """Exclude capacity reserved for ranks that runtime scale-up has not admitted."""
    elastic_ep_state = ElasticEPStateManager.instance()
    effective_ep_size = (
        elastic_ep_state.effective_ep_size if elastic_ep_state is not None else 0
    )
    if effective_ep_size <= 0:
        return committed_active_ranks
    return committed_active_ranks[:effective_ep_size]


class ElasticEPStatusPublisher:
    """Base publisher interface for Elastic EP status sinks."""

    def publish_committed_active_ranks(
        self,
        committed_active_ranks: torch.Tensor,
        adjusting: bool = False,
    ) -> None:
        pass


class ControllerElasticEPStatusPublisher(ElasticEPStatusPublisher):
    """Push per-DP boolean health to DataParallelController (existing behavior)."""

    def __init__(self, send_to_controller: Any, dp_size: int):
        self.send_to_controller = send_to_controller
        self.dp_size = dp_size
        self.last_status = [True] * dp_size

    def publish_committed_active_ranks(
        self,
        committed_active_ranks: torch.Tensor,
        adjusting: bool = False,
    ) -> None:
        assert committed_active_ranks.numel() % self.dp_size == 0
        dp_active_ranks = committed_active_ranks.reshape(self.dp_size, -1).prod(dim=1)
        status = dp_active_ranks.bool().tolist()

        if status == self.last_status:
            return
        self.send_to_controller.send_output(ActiveRanksOutput(status=status))
        self.last_status = status


_ELASTIC_EP_GAUGES: dict = {}


def _get_elastic_ep_gauge(name: str, documentation: str):
    if name not in _ELASTIC_EP_GAUGES:
        # Import after PROMETHEUS_MULTIPROC_DIR is prepared.
        from prometheus_client import Gauge

        _ELASTIC_EP_GAUGES[name] = Gauge(
            name=name,
            documentation=documentation,
            labelnames=["model_name", "scope"],
            multiprocess_mode="mostrecent",
        )
    return _ELASTIC_EP_GAUGES[name]


def _set_elastic_ep_gauge_int(gauge, labels: dict, value: int) -> None:
    # Keep publish path explicit: all Elastic EP gauge values are set as ints.
    gauge.labels(**labels).set(int(value))


class MetricsElasticEPStatusPublisher(ElasticEPStatusPublisher):
    """Publish cluster Elastic EP status metrics from node0 global rank0 only.

    We export only the cluster-level state (no per-rank metrics) to keep
    metric cardinality and downstream complexity low.
    """

    def __init__(self, server_args: ServerArgs):
        self.model_name = server_args.served_model_name
        self.min_serving_ranks = max(envs.SGLANG_ELASTIC_EP_MIN_SERVING_RANKS.get(), 1)
        self._is_exporter_cache: Optional[bool] = None
        self._last_published: Optional[tuple] = None
        self._last_total_ranks_published: Optional[int] = None

    def _is_exporter(self) -> bool:
        # Resolve once per process; non-exporters no-op forever after.
        if self._is_exporter_cache is not None:
            return self._is_exporter_cache
        if not torch.distributed.is_initialized():
            return False
        self._is_exporter_cache = torch.distributed.get_rank() == 0
        return self._is_exporter_cache

    def publish_committed_active_ranks(
        self,
        committed_active_ranks: torch.Tensor,
        adjusting: bool = False,
    ) -> None:
        if not self._is_exporter():
            return

        # committed_active_ranks is already consensus-reduced in elastic_ep.py.
        # This path performs local-only computation with no extra collectives.
        committed_cpu = _effective_committed_active_ranks(
            committed_active_ranks.detach().to("cpu")
        )
        snapshot = _compute_cluster_state(
            committed_cpu,
            adjusting,
            min_serving_ranks=self.min_serving_ranks,
        )
        labels = dict(model_name=self.model_name, scope="cluster")

        # Capacity is fixed, but the admitted world grows at runtime. Republish
        # total_ranks whenever effective_ep_size changes.
        if self._last_total_ranks_published != snapshot.world_size:
            _set_elastic_ep_gauge_int(
                _get_elastic_ep_gauge(
                    "sglang:elastic_ep_total_ranks",
                    "Total number of Elastic EP ranks (world size).",
                ),
                labels,
                snapshot.world_size,
            )
            self._last_total_ranks_published = snapshot.world_size

        state_int = int(snapshot.state)
        # Dedup on (state, active_count): a rank count change within the same
        # state (e.g. TOLERANCE 15 -> 14) must still be published.
        published_key = (state_int, snapshot.active_count)
        # Skip duplicate writes to reduce multiprocess metrics file churn.
        if published_key == self._last_published:
            return
        _set_elastic_ep_gauge_int(
            _get_elastic_ep_gauge(
                "sglang:elastic_ep_state",
                "Elastic EP state: 0=healthy,1=tolerance,2=adjust,3=dead.",
            ),
            labels,
            state_int,
        )
        _set_elastic_ep_gauge_int(
            _get_elastic_ep_gauge(
                "sglang:elastic_ep_active_ranks",
                "Number of currently healthy Elastic EP ranks.",
            ),
            labels,
            snapshot.active_count,
        )
        self._last_published = published_key


class CompositeElasticEPStatusPublisher(ElasticEPStatusPublisher):
    def __init__(self, publishers: list[ElasticEPStatusPublisher]):
        self.publishers = publishers

    def publish_committed_active_ranks(
        self,
        committed_active_ranks: torch.Tensor,
        adjusting: bool = False,
    ) -> None:
        for publisher in self.publishers:
            try:
                publisher.publish_committed_active_ranks(
                    committed_active_ranks, adjusting=adjusting
                )
            except Exception:
                # Status is observability/routing feedback, not a fault verdict.
                # One unavailable sink must not terminate the scheduler or
                # prevent the remaining sinks from receiving the snapshot.
                logger.exception(
                    "Elastic EP status publisher %s failed",
                    type(publisher).__name__,
                )


def create_elastic_ep_status_publisher(
    server_args: ServerArgs, send_to_controller: Any
) -> ElasticEPStatusPublisher:
    publishers: list[ElasticEPStatusPublisher] = []

    if (
        server_args.enable_dp_attention
        and server_args.elastic_ep_backend is not None
        and server_args.dp_size > 1
    ):
        publishers.append(
            ControllerElasticEPStatusPublisher(
                send_to_controller, server_args.max_ep_size or server_args.dp_size
            )
        )

    if (
        server_args.elastic_ep_backend is not None
        and server_args.enable_metrics
        and server_args.node_rank == 0
    ):
        publishers.append(MetricsElasticEPStatusPublisher(server_args))

    if not publishers:
        return ElasticEPStatusPublisher()
    # Always use the composite wrapper so even a single configured sink is
    # best-effort and cannot affect scheduler control decisions.
    return CompositeElasticEPStatusPublisher(publishers)
