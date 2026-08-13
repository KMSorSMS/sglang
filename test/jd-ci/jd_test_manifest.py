#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence


INTERNAL_COMMITS: tuple[str, ...] = (
    "9a821d7c2ed1eba469c0b183c9df8cc74a0aaeaf",
    "e305ec4d75f4f908d38aa89582c614b30d7a771a",
    "0c9fc9febbad7ea2dbabd9c75b8d86b5fbb233fa",
    "0043612717f355eadd9e91aea58c8a902dc241f9",
    "fca4d330a4600e1030cd70b2a21d6cf4f0b8bc35",
    "5a434ce77a74d9d69e8e050f7b9249ef80b5d51d",
    "e500501af5df334f4929b2752e8f45d9f38e78c9",
    "bc9ea948f0b96224f8cbc8fb0b347e0c24c9ac0a",
    "bb1e18920251614ad1565e2d08bc141547bc9af5",
    "9ddc6f2e06a721a3eae10045966247bfb9b7be0a",
    "bbf427bc2f5d15232c656d8a678df5c8f0a472ad",
    "e146f63f35d23ae0d08fa8e4e95fa65b3d08999e",
    "d7dae680b3e7b78dfb838d83b96cd805d7c2a652",
    "9e8dda1cbe8f64cc908a89b332eadf7f4ea5d23b",
    "bbe3e3ef5d466cb1e55216d342ac0ebe3cb5519c",
    "4b4176639fdb6b4413dcbd3bb60e238397b918fc",
    "ae274515d9d59a53f03a9a2292e444f88d918f8e",
    "dc1d5dc6ec2ce5cdd7187c2a2da2d133ec94b46b",
    "e6d5015ee2dacd352b4126fa6339a1d716e39ca7",
    "15b68af531e856abb9b3c9cf7820f6df8fcce4d8",
    "c71de8350659c0ad05e533ffc8a179754ce7f479",
)

VALID_CATEGORIES = frozenset(
    {"cpu", "server", "operator_correctness", "operator_performance"}
)


@dataclass(frozen=True, slots=True)
class JDCase:
    case_id: str
    commits: tuple[str, ...]
    category: str
    command: tuple[str, ...]
    assertion: str
    min_gpus: int = 0
    timeout_seconds: int = 300
    operator: str | None = None
    tracks_ci_head: bool = False


CASES: tuple[JDCase, ...] = (
    JDCase(
        case_id="jd-openai-function-call",
        commits=(
            "e305ec4d75f4f908d38aa89582c614b30d7a771a",
            "0c9fc9febbad7ea2dbabd9c75b8d86b5fbb233fa",
            "e6d5015ee2dacd352b4126fa6339a1d716e39ca7",
        ),
        category="cpu",
        command=(
            "python3",
            "test/jd-ci/unit/server/test_openai_and_function_call.py",
            "-v",
        ),
        assertion="JD invalid-thinking, DeepSeek/GLM reasoning, usage, and Kimi/DSV4 parsing",
    ),
    JDCase(
        case_id="jd-runtime-model-fixes",
        commits=(
            "fca4d330a4600e1030cd70b2a21d6cf4f0b8bc35",
            "e500501af5df334f4929b2752e8f45d9f38e78c9",
            "bb1e18920251614ad1565e2d08bc141547bc9af5",
            "9ddc6f2e06a721a3eae10045966247bfb9b7be0a",
            "e146f63f35d23ae0d08fa8e4e95fa65b3d08999e",
            "9e8dda1cbe8f64cc908a89b332eadf7f4ea5d23b",
            "bbe3e3ef5d466cb1e55216d342ac0ebe3cb5519c",
            "ae274515d9d59a53f03a9a2292e444f88d918f8e",
        ),
        category="cpu",
        command=(
            "python3",
            "test/jd-ci/unit/server/test_runtime_and_model_fixes.py",
            "-v",
        ),
        assertion="JD runtime, multimodal, EPLB, CUDA-graph, CP, quantization, DeepEP dtype, and OCR branches",
    ),
    JDCase(
        case_id="jd-dsv4-inference-buffer-copy",
        commits=(
            "e146f63f35d23ae0d08fa8e4e95fa65b3d08999e",
            "15b68af531e856abb9b3c9cf7820f6df8fcce4d8",
        ),
        category="cpu",
        command=(
            "python3",
            "test/jd-ci/unit/server/test_dsv4_inference_buffer_copy.py",
            "-v",
        ),
        assertion="DSV4 decode and verify metadata support persistent inference tensors",
    ),
    JDCase(
        case_id="jd-metrics-cache",
        commits=("4b4176639fdb6b4413dcbd3bb60e238397b918fc",),
        category="cpu",
        command=(
            "python3",
            "test/jd-ci/unit/server/test_metrics_and_cache.py",
            "-v",
        ),
        assertion="JD L1/L2 cache metrics state and duration accounting",
    ),
    JDCase(
        case_id="jd-deploy-and-tma-configs",
        commits=(
            "0043612717f355eadd9e91aea58c8a902dc241f9",
            "5a434ce77a74d9d69e8e050f7b9249ef80b5d51d",
            "bc9ea948f0b96224f8cbc8fb0b347e0c24c9ac0a",
            "bbf427bc2f5d15232c656d8a678df5c8f0a472ad",
            "d7dae680b3e7b78dfb838d83b96cd805d7c2a652",
        ),
        category="cpu",
        command=(
            "python3",
            "test/jd-ci/unit/server/test_deploy_and_tma_configs.py",
            "-v",
        ),
        assertion="JD deploy model mapping and internal H20D/H200 TMA configuration files",
    ),
    JDCase(
        case_id="jd-ci-contract",
        commits=(),
        category="cpu",
        tracks_ci_head=True,
        command=(
            "python3",
            "test/jd-ci/unit/ci/test_internal_ci_contract.py",
            "-v",
        ),
        assertion="JD build, artifact, log-dump, and orchestration contracts",
    ),
    JDCase(
        case_id="jd-server-api-regressions",
        commits=(
            "e305ec4d75f4f908d38aa89582c614b30d7a771a",
            "e500501af5df334f4929b2752e8f45d9f38e78c9",
            "9ddc6f2e06a721a3eae10045966247bfb9b7be0a",
            "ae274515d9d59a53f03a9a2292e444f88d918f8e",
        ),
        category="server",
        command=(
            "python3",
            "test/jd-ci/pipeline/server_api_dummy_model.py",
            "--case",
            "jd-server-api-regressions",
            "--output",
            "{result}",
        ),
        assertion="One Qwen2.5-VL dummy Server covers all JD HTTP and request-lifecycle fixes",
        min_gpus=1,
        timeout_seconds=600,
    ),
    JDCase(
        case_id="jd-rmsnorm-correctness",
        commits=("9a821d7c2ed1eba469c0b183c9df8cc74a0aaeaf",),
        category="operator_correctness",
        command=("python3", "test/jd-ci/operators/test_optimized_rmsnorm.py"),
        assertion="Optimized RMSNorm matches the reference RMSNorm",
        min_gpus=1,
        operator="optimized_rmsnorm",
    ),
    JDCase(
        case_id="jd-rmsnorm-performance",
        commits=("9a821d7c2ed1eba469c0b183c9df8cc74a0aaeaf",),
        category="operator_performance",
        command=("python3", "test/jd-ci/operators/bench_optimized_rmsnorm.py"),
        assertion="Optimized RMSNorm is not slower than its reference path",
        min_gpus=1,
        operator="optimized_rmsnorm",
    ),
    JDCase(
        case_id="jd-dp-allgather-correctness",
        commits=(
            "fca4d330a4600e1030cd70b2a21d6cf4f0b8bc35",
            "dc1d5dc6ec2ce5cdd7187c2a2da2d133ec94b46b",
        ),
        category="operator_correctness",
        command=(
            "torchrun",
            "--standalone",
            "--nproc-per-node=2",
            "test/jd-ci/operators/test_dp_attention_allgather.py",
        ),
        assertion="Compressed DP-attention all-gather reconstructs legacy metadata",
        min_gpus=2,
        operator="dp_attention_allgather",
    ),
    JDCase(
        case_id="jd-dp-allgather-performance",
        commits=(
            "fca4d330a4600e1030cd70b2a21d6cf4f0b8bc35",
            "dc1d5dc6ec2ce5cdd7187c2a2da2d133ec94b46b",
        ),
        category="operator_performance",
        command=(
            "torchrun",
            "--standalone",
            "--nproc-per-node=2",
            "test/jd-ci/operators/bench_dp_attention_allgather.py",
        ),
        assertion="Compressed DP-attention metadata reduces transfer overhead",
        min_gpus=2,
        operator="dp_attention_allgather",
    ),
    JDCase(
        case_id="jd-dsv4-norm-rope-correctness",
        commits=(),
        category="operator_correctness",
        command=("python3", "test/jd-ci/operators/test_dsv4_norm_rope.py"),
        assertion="JD DSV4 norm-rope kernel matches its reference path",
        min_gpus=1,
        operator="dsv4_norm_rope",
        tracks_ci_head=True,
    ),
    JDCase(
        case_id="jd-dsv4-norm-rope-performance",
        commits=(),
        category="operator_performance",
        command=("python3", "test/jd-ci/operators/bench_dsv4_norm_rope.py"),
        assertion="JD DSV4 norm-rope kernel stays within its relative performance gate",
        min_gpus=1,
        operator="dsv4_norm_rope",
        tracks_ci_head=True,
    ),
    JDCase(
        case_id="jd-w4a8-correctness",
        commits=(
            "9e8dda1cbe8f64cc908a89b332eadf7f4ea5d23b",
            "c71de8350659c0ad05e533ffc8a179754ce7f479",
        ),
        category="operator_correctness",
        command=("python3", "test/jd-ci/operators/test_w4a8.py"),
        assertion="JD W4A8 group-size, scale packing, and dynamic quantization are correct",
        min_gpus=1,
        operator="w4a8",
    ),
    JDCase(
        case_id="jd-w4a8-performance",
        commits=(
            "9e8dda1cbe8f64cc908a89b332eadf7f4ea5d23b",
            "c71de8350659c0ad05e533ffc8a179754ce7f479",
        ),
        category="operator_performance",
        command=("python3", "test/jd-ci/operators/bench_w4a8.py"),
        assertion="JD W4A8 optimized path stays within its relative performance gate",
        min_gpus=1,
        operator="w4a8",
    ),
)


def all_cases(category: str | None = None) -> list[JDCase]:
    if category is None:
        return list(CASES)
    if category not in VALID_CATEGORIES:
        raise ValueError(f"unknown JD test category: {category}")
    return [case for case in CASES if case.category == category]


def _uses_upstream_test_command(case: JDCase) -> bool:
    command = " ".join(case.command)
    if "test/run_suite.py" in command:
        return True
    return "test/registered/" in command


def _command_paths(case: JDCase) -> list[str]:
    return [part for part in case.command if part.endswith((".py", ".sh"))]


def validate_cases(
    cases: Sequence[JDCase],
    expected_commits: Sequence[str],
    *,
    repo_root: str | Path | None = None,
    check_paths: bool = True,
) -> dict[str, list[str]]:
    ids = [case.case_id for case in cases]
    mapped_commits = {commit for case in cases for commit in case.commits}
    expected = set(expected_commits)
    duplicate_case_ids = sorted({case_id for case_id in ids if ids.count(case_id) > 1})
    invalid_categories = sorted(
        case.case_id for case in cases if case.category not in VALID_CATEGORIES
    )
    invalid_commits = sorted(
        {
            commit
            for case in cases
            for commit in case.commits
            if len(commit) != 40 or any(char not in "0123456789abcdef" for char in commit)
        }
    )
    upstream_test_commands = sorted(
        case.case_id for case in cases if _uses_upstream_test_command(case)
    )
    untracked_cases = sorted(
        case.case_id for case in cases if not case.commits and not case.tracks_ci_head
    )
    invalid_head_tracking = sorted(
        case.case_id for case in cases if case.commits and case.tracks_ci_head
    )
    missing_paths: list[str] = []
    if check_paths:
        root = Path(repo_root or Path(__file__).resolve().parents[2])
        for case in cases:
            for path in _command_paths(case):
                if not (root / path).is_file():
                    missing_paths.append(f"{case.case_id}:{path}")

    return {
        "missing_commits": sorted(expected - mapped_commits),
        "unexpected_commits": sorted(mapped_commits - expected),
        "duplicate_case_ids": duplicate_case_ids,
        "invalid_categories": invalid_categories,
        "invalid_commits": invalid_commits,
        "upstream_test_commands": upstream_test_commands,
        "untracked_cases": untracked_cases,
        "invalid_head_tracking": invalid_head_tracking,
        "missing_paths": sorted(missing_paths),
    }


def validate_manifest(repo_root: str | Path) -> dict[str, list[str]]:
    report = validate_cases(CASES, INTERNAL_COMMITS, repo_root=repo_root)
    failures = {name: values for name, values in report.items() if values}
    if failures:
        raise ValueError("invalid JD test manifest: " + json.dumps(failures, sort_keys=True))
    return report


def write_cases(path: str | Path, cases: Sequence[JDCase]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(
        json.dumps([asdict(case) for case in cases], indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output)


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Emit the fixed cumulative JD test inventory")
    parser.add_argument("--output", required=True)
    parser.add_argument("--category", choices=sorted(VALID_CATEGORIES))
    parser.add_argument("--source", default=str(Path(__file__).resolve().parents[2]))
    parser.add_argument("--skip-path-check", action="store_true")
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    report = validate_cases(
        CASES,
        INTERNAL_COMMITS,
        repo_root=args.source,
        check_paths=not args.skip_path_check,
    )
    failures = {name: values for name, values in report.items() if values}
    if failures:
        raise SystemExit("invalid JD test manifest: " + json.dumps(failures, sort_keys=True))
    write_cases(args.output, all_cases(args.category))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
