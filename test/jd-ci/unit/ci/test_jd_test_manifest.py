import sys
import unittest
from pathlib import Path

try:
    from sglang.test.ci.ci_register import register_cpu_ci
    from sglang.test.test_utils import CustomTestCase
except ModuleNotFoundError:
    def register_cpu_ci(*args, **kwargs):
        return None

    CustomTestCase = unittest.TestCase


REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT / "test/jd-ci"))

from jd_test_manifest import (  # noqa: E402
    INTERNAL_COMMITS,
    JDCase,
    all_cases,
    validate_cases,
)


register_cpu_ci(est_time=3, suite="base-a-test-cpu")


EXPECTED_V0517_COMMITS = {
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
    "4b4176639fdb6b4413dcbd3bb60e238397b918fc",
    "ae274515d9d59a53f03a9a2292e444f88d918f8e",
    "dc1d5dc6ec2ce5cdd7187c2a2da2d133ec94b46b",
    "e6d5015ee2dacd352b4126fa6339a1d716e39ca7",
    "15b68af531e856abb9b3c9cf7820f6df8fcce4d8",
    "c71de8350659c0ad05e533ffc8a179754ce7f479",
}


class TestJDTestManifest(CustomTestCase):
    def test_manifest_covers_every_v0517_production_commit(self):
        self.assertEqual(set(INTERNAL_COMMITS), EXPECTED_V0517_COMMITS)

        report = validate_cases(all_cases(), INTERNAL_COMMITS, check_paths=False)

        self.assertEqual(report["missing_commits"], [])
        self.assertEqual(report["duplicate_case_ids"], [])
        self.assertEqual(report["upstream_test_commands"], [])
        self.assertEqual(report["untracked_cases"], [])
        self.assertEqual(report["invalid_head_tracking"], [])

    def test_ci_owned_cases_track_the_single_ci_head_without_self_sha(self):
        tracked = {
            case.case_id for case in all_cases() if case.tracks_ci_head
        }
        self.assertEqual(
            tracked,
            {
                "jd-ci-contract",
                "jd-dsv4-norm-rope-correctness",
                "jd-dsv4-norm-rope-performance",
            },
        )
        for case in all_cases():
            if case.tracks_ci_head:
                self.assertEqual(case.commits, ())

    def test_every_operator_has_correctness_and_performance(self):
        cases = all_cases()
        correctness = {
            case.operator
            for case in cases
            if case.category == "operator_correctness"
        }
        performance = {
            case.operator
            for case in cases
            if case.category == "operator_performance"
        }

        self.assertTrue(correctness)
        self.assertEqual(correctness, performance)
        self.assertNotIn(None, correctness)

    def test_upstream_suite_command_is_rejected(self):
        bad_case = JDCase(
            case_id="bad-upstream-suite",
            commits=(next(iter(EXPECTED_V0517_COMMITS)),),
            category="cpu",
            command=("python3", "test/run_suite.py", "--suite", "base-a-test-cpu"),
            assertion="incorrectly runs an upstream suite",
        )

        report = validate_cases(
            [bad_case],
            [bad_case.commits[0]],
            check_paths=False,
        )

        self.assertEqual(report["upstream_test_commands"], ["bad-upstream-suite"])

    def test_unmapped_commit_is_rejected(self):
        report = validate_cases(all_cases(), [*INTERNAL_COMMITS, "f" * 40], check_paths=False)

        self.assertEqual(report["missing_commits"], ["f" * 40])

    def test_case_inventory_is_fixed_not_diff_selected(self):
        first = [case.case_id for case in all_cases()]
        second = [case.case_id for case in all_cases()]

        self.assertEqual(first, second)
        self.assertNotIn("changed_files", all_cases.__code__.co_varnames)

    def test_model_specific_protocol_fixtures_remain_explicit(self):
        test_source = (
            REPO_ROOT / "test/jd-ci/unit/server/test_openai_and_function_call.py"
        ).read_text(encoding="utf-8")
        required_methods = (
            "test_invalid_thinking_list_is_ignored",
            "test_invalid_thinking_dict_without_type_is_ignored",
            "test_invalid_thinking_unknown_string_is_ignored",
            "test_invalid_thinking_integer_is_ignored",
            "test_deepseek_v4_reasoning_switch",
            "test_glm45_non_stream_tool_interruption",
            "test_glm45_stream_tool_interruption",
            "test_reasoning_token_usage",
        )

        for method in required_methods:
            with self.subTest(method=method):
                self.assertIn(f"def {method}(", test_source)


if __name__ == "__main__":
    unittest.main()
