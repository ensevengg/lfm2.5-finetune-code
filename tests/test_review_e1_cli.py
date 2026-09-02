from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


WORKSPACE = Path(__file__).resolve().parents[1]
REVIEWER = WORKSPACE / "scripts" / "review_e1.py"
SUITE = WORKSPACE / "benchmarks" / "suites" / "e0-humanevalplus-smoke-v1.json"


class ReviewE1CliTests(unittest.TestCase):
    def run_reviewer(self, records: list[dict[str, object]], root: Path) -> tuple[subprocess.CompletedProcess[str], dict[str, object]]:
        generations = root / "generations.jsonl"
        generations.write_text(
            "".join(json.dumps(record) + "\n" for record in records),
            encoding="utf-8",
        )
        output = root / "behavior-review.json"
        completed = subprocess.run(
            [
                sys.executable,
                str(REVIEWER),
                "--suite",
                str(SUITE),
                "--generations",
                str(generations),
                "--output",
                str(output),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        return completed, json.loads(output.read_text(encoding="utf-8"))

    def test_bad_integrity_outputs_make_behavior_gate_fail(self) -> None:
        records = [
            {
                "case_id": "integrity-greeting",
                "case_kind": "prompt_integrity",
                "raw_completion": "I already responded to your greeting. Hello again!",
                "completion_token_count": 12,
                "finish_reason": "eos",
            },
            {
                "case_id": "integrity-exact",
                "case_kind": "prompt_integrity",
                "raw_completion": "<think>I should obey.</think>\nHello",
                "completion_token_count": 10,
                "finish_reason": "eos",
            },
            {
                "case_id": "integrity-state",
                "case_kind": "prompt_integrity",
                "raw_completion": "You previously asked my name. " * 200,
                "completion_token_count": 3072,
                "finish_reason": "limit",
            },
            {
                "case_id": "integrity-code-fence",
                "case_kind": "prompt_integrity",
                "raw_completion": "def add(a, b):\n    return a + b",
                "completion_token_count": 12,
                "finish_reason": "eos",
            },
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            completed, report = self.run_reviewer(records, root)

            self.assertEqual(completed.returncode, 4)
            self.assertFalse(report["passed"])
            self.assertEqual(report["failed_case_count"], 4)
            self.assertEqual(
                {case["case_id"] for case in report["cases"] if not case["passed"]},
                {
                    "integrity-greeting",
                    "integrity-exact",
                    "integrity-state",
                    "integrity-code-fence",
                },
            )

    def test_missing_bos_token_makes_prompt_encoding_gate_fail(self) -> None:
        records = self.compliant_records()
        for record in records:
            record["templated_prompt_token_ids"] = [6, 6423, 708]
        with tempfile.TemporaryDirectory() as temp_dir:
            completed, report = self.run_reviewer(records, Path(temp_dir))

            self.assertEqual(completed.returncode, 4)
            self.assertFalse(report["prompt_encoding"]["passed"])
            self.assertEqual(report["prompt_encoding"]["expected_bos_token_id"], 1)

    def test_compliant_integrity_outputs_and_bos_pass(self) -> None:
        records = self.compliant_records()
        for record in records:
            record["templated_prompt_token_ids"] = [1, 6, 6423, 708]
        with tempfile.TemporaryDirectory() as temp_dir:
            completed, report = self.run_reviewer(records, Path(temp_dir))

            self.assertEqual(completed.returncode, 0)
            self.assertTrue(report["passed"])
            self.assertEqual(report["failed_case_count"], 0)
            self.assertTrue(report["prompt_encoding"]["passed"])

    @staticmethod
    def compliant_records() -> list[dict[str, object]]:
        return [
            {
                "case_id": "integrity-greeting",
                "case_kind": "prompt_integrity",
                "raw_completion": "Hello! How can I help?",
                "completion_token_count": 7,
                "finish_reason": "eos",
            },
            {
                "case_id": "integrity-exact",
                "case_kind": "prompt_integrity",
                "raw_completion": "Hello",
                "completion_token_count": 1,
                "finish_reason": "eos",
            },
            {
                "case_id": "integrity-state",
                "case_kind": "prompt_integrity",
                "raw_completion": "Your only message was: What is the only message I have sent you in this conversation?",
                "completion_token_count": 20,
                "finish_reason": "eos",
            },
            {
                "case_id": "integrity-code-fence",
                "case_kind": "prompt_integrity",
                "raw_completion": "```python\ndef add(a, b):\n    return a + b\n```",
                "completion_token_count": 16,
                "finish_reason": "eos",
            },
        ]


if __name__ == "__main__":
    unittest.main()
