"""
Test file to verify the correctness of parallel group calculations.

This test validates that the parallel group initialization creates the correct
groups for different parallelism configurations including:
- Tensor parallelism (TP)
- Pipeline parallelism (PP)
- Attention context parallelism (attn_cp)
- Attention data parallelism (attn_dp)
- MoE expert parallelism (EP)
- MoE data parallelism (moe_dp)

These tests call the ACTUAL initialize_model_parallel() function with mocked
distributed backend to verify the group construction logic.

## How These Tests Work

initialize_model_parallel() creates ALL groups for ALL ranks in a single call.
For example, when creating TP groups with tp_size=2 and world_size=8:

    group_ranks = [[0,1], [2,3], [4,5], [6,7]]  # ALL groups created
    _TP = init_model_parallel_group(group_ranks, local_rank, ...)

ALL ranks call this function and get the same complete group structure. Each rank
then figures out which specific group(s) it belongs to.

Our tests:
1. Mock the distributed backend (no real GPUs needed)
2. Mock init_model_parallel_group to capture the group_ranks parameter
3. Call the real initialize_model_parallel()
4. Verify group_ranks contains the expected complete group structure

We only need to simulate rank 0 because we're testing the group creation logic,
not the per-rank group membership logic.
"""

from __future__ import annotations

import sys
import types
from unittest.mock import Mock, patch

import pytest
import torch

from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci

register_cuda_ci(est_time=8, stage="base-b", runner_config="1-gpu-small")
register_amd_ci(est_time=8, suite="stage-b-test-1-gpu-small-amd")

# Import the actual parallel_state module
parallel_state = pytest.importorskip("sglang.srt.distributed.parallel_state")


def test_parallel_group_construction_tp8_attn_cp2():
    """
    Test parallel group construction for 8 GPU configuration with:
    - tensor_model_parallel_size = 8
    - attention_context_model_parallel_size = 2

    Expected groups based on docstring example:
        1 tensor model-parallel group:
            [g0, g1, g2, g3, g4, g5, g6, g7]
        4 attention context-parallel groups:
            [g0, g4], [g1, g5], [g2, g6], [g3, g7]

    This test calls the ACTUAL initialize_model_parallel() and verifies the groups.

    Note: We simulate only rank 0 here, but initialize_model_parallel() creates
    ALL groups for ALL ranks in a single call. We capture these groups via mocking
    and verify the complete group structure.
    """
    world_size = 8

    # Mock the distributed backend
    # Note: get_rank() returns 0 because we're testing from a single process,
    # but initialize_model_parallel() still creates all groups for all ranks
    with (
        patch.object(parallel_state, "_WORLD", None),
        patch.object(parallel_state, "_TP", None),
        patch.object(parallel_state, "_ATTN_CP", None),
        patch.object(parallel_state, "_ATTN_TP", None),
        patch.object(parallel_state, "_PP", None),
        patch("torch.distributed.is_initialized", return_value=True),
        patch("torch.distributed.get_world_size", return_value=world_size),
        patch("torch.distributed.get_rank", return_value=0),
        patch("torch.distributed.get_backend", return_value="nccl"),
    ):

        # Mock init_model_parallel_group to capture the groups being created
        created_groups = {}

        def mock_init_model_parallel_group(group_ranks, local_rank, backend, **kwargs):
            group_name = kwargs.get("group_name", "unknown")
            created_groups[group_name] = group_ranks

            # Create a mock group object
            mock_group = Mock()
            mock_group.device_group = Mock()
            return mock_group

        with (
            patch.object(
                parallel_state,
                "init_model_parallel_group",
                side_effect=mock_init_model_parallel_group,
            ),
            patch.object(parallel_state, "get_world_group") as mock_world_group,
        ):

            # Mock world group
            mock_world = Mock()
            mock_world.device_group = Mock()
            mock_world.local_rank = 0
            mock_world_group.return_value = mock_world

            # Call the actual function
            parallel_state.initialize_model_parallel(
                tensor_model_parallel_size=8,
                pipeline_model_parallel_size=1,
                attention_context_model_parallel_size=2,
            )

            # Verify TP groups
            tp_groups = created_groups.get("tp", [])
            assert len(tp_groups) == 1, f"Expected 1 TP group, got {len(tp_groups)}"
            assert tp_groups[0] == [
                0,
                1,
                2,
                3,
                4,
                5,
                6,
                7,
            ], f"Wrong TP group: {tp_groups[0]}"

            # Verify ATTN_CP groups
            attn_cp_groups = created_groups.get("attn_cp", [])
            assert (
                len(attn_cp_groups) == 4
            ), f"Expected 4 ATTN_CP groups, got {len(attn_cp_groups)}"
            expected_attn_cp = [
                [0, 4],
                [1, 5],
                [2, 6],
                [3, 7],
            ]
            assert (
                attn_cp_groups == expected_attn_cp
            ), f"Wrong ATTN_CP groups: {attn_cp_groups}"

            print("TP=8, Attn CP=2 group construction verified")

            # Cleanup
            parallel_state.destroy_model_parallel()


def test_parallel_group_construction_tp8_moe_ep4_cp2():
    """
    Test parallel group construction for 8 GPU configuration with:
    - tensor_model_parallel_size = 8
    - expert_model_parallel_size = 4
    - moe_data_model_parallel_size = 2

    Expected groups:
        1 tensor model-parallel group:
            [g0, g1, g2, g3, g4, g5, g6, g7]
        2 MoE expert-parallel groups:
            [g0, g1, g2, g3], [g4, g5, g6, g7]
        4 MoE data-parallel groups:
            [g0, g4], [g1, g5], [g2, g6], [g3, g7]
    """
    world_size = 8

    # Mock the distributed backend
    with (
        patch.object(parallel_state, "_WORLD", None),
        patch.object(parallel_state, "_TP", None),
        patch.object(parallel_state, "_MOE_EP", None),
        patch.object(parallel_state, "_MOE_DP", None),
        patch.object(parallel_state, "_MOE_TP", None),
        patch.object(parallel_state, "_PP", None),
        patch("torch.distributed.is_initialized", return_value=True),
        patch("torch.distributed.get_world_size", return_value=world_size),
        patch("torch.distributed.get_rank", return_value=0),
        patch("torch.distributed.get_backend", return_value="nccl"),
    ):

        # Mock init_model_parallel_group to capture the groups being created
        created_groups = {}

        def mock_init_model_parallel_group(group_ranks, local_rank, backend, **kwargs):
            group_name = kwargs.get("group_name", "unknown")
            created_groups[group_name] = group_ranks

            # Create a mock group object
            mock_group = Mock()
            mock_group.device_group = Mock()
            return mock_group

        with (
            patch.object(
                parallel_state,
                "init_model_parallel_group",
                side_effect=mock_init_model_parallel_group,
            ),
            patch.object(parallel_state, "get_world_group") as mock_world_group,
        ):

            # Mock world group
            mock_world = Mock()
            mock_world.device_group = Mock()
            mock_world.local_rank = 0
            mock_world_group.return_value = mock_world

            # Call the actual function
            parallel_state.initialize_model_parallel(
                tensor_model_parallel_size=8,
                expert_model_parallel_size=4,
                pipeline_model_parallel_size=1,
                moe_data_model_parallel_size=2,
            )

            # Verify TP groups
            tp_groups = created_groups.get("tp", [])
            assert len(tp_groups) == 1, f"Expected 1 TP group, got {len(tp_groups)}"
            assert tp_groups[0] == [
                0,
                1,
                2,
                3,
                4,
                5,
                6,
                7,
            ], f"Wrong TP group: {tp_groups[0]}"

            # Verify MOE_EP groups
            moe_ep_groups = created_groups.get("moe_ep", [])
            assert (
                len(moe_ep_groups) == 2
            ), f"Expected 2 MOE_EP groups, got {len(moe_ep_groups)}"
            expected_moe_ep = [
                [0, 1, 2, 3],
                [4, 5, 6, 7],
            ]
            assert (
                moe_ep_groups == expected_moe_ep
            ), f"Wrong MOE_EP groups: {moe_ep_groups}"

            # Verify MOE_DP groups
            moe_dp_groups = created_groups.get("moe_dp", [])
            assert (
                len(moe_dp_groups) == 4
            ), f"Expected 4 MOE_DP groups, got {len(moe_dp_groups)}"
            expected_moe_dp = [
                [0, 4],
                [1, 5],
                [2, 6],
                [3, 7],
            ]
            assert (
                moe_dp_groups == expected_moe_dp
            ), f"Wrong MOE_DP groups: {moe_dp_groups}"

            print("TP=8, MoE EP=4, MoE CP=2 group construction verified")

            # Cleanup
            parallel_state.destroy_model_parallel()


class TestGroupLocalActiveRanks:
    @staticmethod
    def _construct_mooncake_group(ranks, max_world_size=None):
        class FakeMooncakeBackendOptions:
            instances = []

            def __init__(self, active_ranks, recovered_rank, max_world_size=None):
                self.active_ranks = active_ranks
                self.recovered_rank = recovered_rank
                self.max_world_size = max_world_size
                self.__class__.instances.append(self)

        mooncake_module = types.ModuleType("mooncake")
        mooncake_pg_module = types.ModuleType("mooncake.pg")
        mooncake_pg_module.MooncakeBackendOptions = FakeMooncakeBackendOptions
        mooncake_module.pg = mooncake_pg_module

        with (
            patch.dict(
                sys.modules,
                {"mooncake": mooncake_module, "mooncake.pg": mooncake_pg_module},
            ),
            patch.object(parallel_state, "is_cuda_alike", return_value=False),
            patch("torch.distributed.get_rank", return_value=ranks[0]),
            patch("torch.distributed.new_group", return_value=Mock()) as new_group,
        ):
            group = parallel_state.GroupCoordinator(
                group_ranks=[ranks],
                local_rank=0,
                torch_distributed_backend="mooncake",
                use_pynccl=False,
                use_pymscclpp=False,
                use_custom_allreduce=False,
                use_torch_symm_mem_all_reduce=False,
                use_hpu_communicator=False,
                use_xpu_communicator=False,
                use_npu_communicator=False,
                max_world_size=max_world_size,
            )

        assert new_group.call_count == 2
        return group, FakeMooncakeBackendOptions.instances

    def test_nonzero_contiguous_ranks_use_group_local_mirrors(self):
        ranks = [32, 33, 34, 35]

        try:
            group, options = self._construct_mooncake_group(ranks, max_world_size=8)
        except ValueError as error:
            pytest.fail(
                "Mooncake capacity must be measured in group-local slots, not "
                f"global ranks: {error}"
            )

        assert len(options) == 2
        assert [option.active_ranks.tolist() for option in options] == [
            [1, 1, 1, 1, 0, 0, 0, 0],
            [1, 1, 1, 1, 0, 0, 0, 0],
        ]
        assert [option.recovered_rank for option in options] == [False, False]
        assert [option.max_world_size for option in options] == [8, 8]
        assert group.active_ranks.tolist() == [1, 1, 1, 1]
        assert group.active_ranks_cpu.tolist() == [1, 1, 1, 1]

        options[0].active_ranks[2] = 0
        assert group.active_ranks.tolist() == [1, 1, 0, 1]
        assert group.active_ranks_cpu.tolist() == [1, 1, 1, 1]

        options[1].active_ranks[2] = 0
        assert group.active_ranks.tolist() == [1, 1, 0, 1]
        assert group.active_ranks_cpu.tolist() == [1, 1, 0, 1]

    def test_strided_ranks_use_group_local_mirrors_without_capacity_extension(self):
        group, options = self._construct_mooncake_group([1, 5, 9])

        assert len(options) == 2
        assert [option.active_ranks.tolist() for option in options] == [
            [1, 1, 1],
            [1, 1, 1],
        ]
        assert [option.recovered_rank for option in options] == [False, False]
        assert [option.max_world_size for option in options] == [None, None]
        assert group.active_ranks.tolist() == [1, 1, 1]
        assert group.active_ranks_cpu.tolist() == [1, 1, 1]

        options[0].active_ranks[1] = 0
        assert group.active_ranks.tolist() == [1, 0, 1]
        assert group.active_ranks_cpu.tolist() == [1, 1, 1]

        options[1].active_ranks[1] = 0
        assert group.active_ranks.tolist() == [1, 0, 1]
        assert group.active_ranks_cpu.tolist() == [1, 0, 1]

    def test_max_world_size_smaller_than_group_is_rejected(self):
        with pytest.raises(ValueError, match="group size"):
            self._construct_mooncake_group([32, 33, 34, 35], max_world_size=3)


if __name__ == "__main__":
    # Run tests without requiring GPUs
    import sys

    try:
        test_parallel_group_construction_tp8_attn_cp2()
        test_parallel_group_construction_tp8_moe_ep4_cp2()

        sys.exit(0)
    except AssertionError as e:
        print(f"\n Test failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n Unexpected error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
