import subprocess
import sys
import textwrap

import pytest

from sglang.test.ci.ci_register import register_cpu_ci


register_cpu_ci(est_time=30, suite="base-c-test-cpu")


_HUMMING_TEST_SETUP = r"""
import sys
import types
from enum import Enum
from types import SimpleNamespace
from unittest.mock import patch

import torch

try:
    __import__("humming")
    fake_humming_modules = {}
except ModuleNotFoundError:
    humming = types.ModuleType("humming")
    humming.dtypes = SimpleNamespace(float8e4m3=object())

    humming_config = types.ModuleType("humming.config")

    class GemmType(Enum):
        INDEXED = "indexed"
        GROUPED_CONTIGUOUS = "grouped_contiguous"
        GROUPED_MASKED = "grouped_masked"

    humming_config.GemmType = GemmType

    humming_layer = types.ModuleType("humming.layer")

    class HummingInputSchema:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    humming_layer.HummingInputSchema = HummingInputSchema
    humming_layer.HummingMethod = type("HummingMethod", (), {})

    humming_schema = types.ModuleType("humming.schema")
    humming_schema.BaseWeightSchema = type("BaseWeightSchema", (), {})
    fake_humming_modules = {
        "humming": humming,
        "humming.config": humming_config,
        "humming.layer": humming_layer,
        "humming.schema": humming_schema,
    }

with patch.dict(sys.modules, fake_humming_modules):
    from sglang.srt.layers.moe.moe_runner import humming as humming_runner
    from sglang.srt.layers.quantization import humming_utils


class MooncakeBackend:
    def is_mooncake(self):
        return True
"""


def _run_humming_test(source: str) -> None:
    result = subprocess.run(
        [sys.executable, "-c", _HUMMING_TEST_SETUP + textwrap.dedent(source)],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_mooncake_configures_humming_for_fp8_group_128_dispatch():
    _run_humming_test(
        """
        class ForbiddenDispatcher:
            def set_quant_config(self, *args, **kwargs):
                raise AssertionError(
                    "Mooncake must not use the DeepEP quantization API"
                )

        layer = SimpleNamespace(dispatcher=ForbiddenDispatcher())
        with patch.object(
            humming_utils,
            "get_moe_a2a_backend",
            return_value=MooncakeBackend(),
        ):
            use_fp8 = humming_utils.configure_humming_deepep_dispatch(layer)

        assert use_fp8 is True
        assert layer._humming_uses_deepep_fp8_dispatch is True
        schema = humming_utils.make_humming_deepep_input_schema("w13", 512)
        assert schema.a_dtype == "float8e4m3"
        assert schema.input_scale_group_size == 128
        """
    )


def test_mooncake_token_major_scales_are_not_transposed_when_axes_match():
    _run_humming_test(
        """
        num_experts = 2
        num_tokens = 4
        num_groups = 4
        hidden_size = num_groups * 128

        scale_storage = torch.arange(
            num_experts * num_groups * num_tokens, dtype=torch.float32
        ).reshape(num_experts, num_groups, num_tokens)
        token_major_scales = scale_storage.transpose(1, 2)
        assert token_major_scales.shape == (num_experts, num_tokens, num_groups)
        assert not token_major_scales.is_contiguous()

        dispatch_output = SimpleNamespace(
            hidden_states=torch.empty(
                (num_experts, num_tokens, hidden_size),
                dtype=torch.float8_e4m3fn,
            ),
            hidden_states_scale=token_major_scales,
            topk_ids=torch.zeros((num_tokens, 1), dtype=torch.int64),
            topk_weights=torch.ones((num_tokens, 1), dtype=torch.float32),
            masked_m=torch.full(
                (num_experts,), num_tokens, dtype=torch.int32
            ),
            expected_m=num_tokens,
        )
        layer = SimpleNamespace(
            _humming_uses_deepep_fp8_dispatch=True,
            humming_metas={
                "w13": SimpleNamespace(
                    a_dtype=humming_runner.dtypes.float8e4m3,
                    input_scale_group_size=128,
                )
            },
        )

        with patch.object(
            humming_runner,
            "get_moe_a2a_backend",
            return_value=MooncakeBackend(),
            create=True,
        ):
            runner_input = humming_runner.pre_permute_deepep_ll_to_humming(
                dispatch_output=dispatch_output,
                quant_info=SimpleNamespace(layer=layer),
                runner_config=SimpleNamespace(),
                running_state={},
            )

        assert torch.equal(
            runner_input.hidden_states_scale, token_major_scales
        )
        """
    )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
