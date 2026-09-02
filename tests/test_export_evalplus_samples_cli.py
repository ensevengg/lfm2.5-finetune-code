from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


WORKSPACE = Path(__file__).resolve().parents[1]
EXPORTER = WORKSPACE / "scripts" / "export_evalplus_samples.py"
PROMPT_MANIFEST = WORKSPACE / "benchmarks" / "prompts" / "lfm-code-v1.json"
PROMPT = json.loads(PROMPT_MANIFEST.read_text(encoding="utf-8"))


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


class ExportEvalPlusSamplesCliTests(unittest.TestCase):
    def run_exporter(
        self, root: Path, records: list[dict[str, object]]
    ) -> tuple[subprocess.CompletedProcess[str], Path, Path]:
        generations = root / "generations.jsonl"
        generations.write_text(
            "".join(json.dumps(record) + "\n" for record in records),
            encoding="utf-8",
        )
        samples = root / "samples.jsonl"
        report = root / "extraction-report.json"
        completed = subprocess.run(
            [
                sys.executable,
                str(EXPORTER),
                "--generations",
                str(generations),
                "--prompt-manifest",
                str(PROMPT_MANIFEST),
                "--samples-output",
                str(samples),
                "--report-output",
                str(report),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        return completed, samples, report

    @staticmethod
    def benchmark_record(
        raw_completion: str,
        task_id: str = "HumanEval/0",
        entry_point: str = "has_close_elements",
    ) -> dict[str, object]:
        return {
            "schema_version": 1,
            "case_kind": "humanevalplus_development",
            "case_id": "humanevalplus-" + task_id.replace("/", "-"),
            "task_id": task_id,
            "entry_point": entry_point,
            "prompt_id": PROMPT["prompt_id"],
            "prompt_sha256": PROMPT["sha256"],
            "raw_completion": raw_completion,
            "generated_program_executed": False,
            "scored": False,
        }

    def test_plain_python_is_exported_without_modifying_raw_evidence(self) -> None:
        raw = (
            "from typing import List\n\n"
            "def has_close_elements(numbers: List[float], threshold: float) -> bool:\n"
            "    return False\n"
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            completed, samples_path, report_path = self.run_exporter(
                Path(temp_dir), [self.benchmark_record(raw)]
            )

            self.assertEqual(
                completed.returncode,
                0,
                msg=f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}",
            )
            samples = [
                json.loads(line)
                for line in samples_path.read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(samples, [{"task_id": "HumanEval/0", "solution": raw}])
            report = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertTrue(report["passed"])
            self.assertEqual(report["cases"][0]["strategy"], "plain_python")
            self.assertEqual(report["cases"][0]["raw_completion_sha256"], sha256_text(raw))
            self.assertEqual(report["cases"][0]["solution_sha256"], sha256_text(raw))
            self.assertFalse(report["generated_programs_executed"])

    def test_one_reasoning_prefix_and_python_fence_are_derived_separately(self) -> None:
        solution = "def parse_music(music_string: str):\n    return []"
        raw = (
            "<think>\nI will produce the requested function.\n</think>\n\n"
            "```python\n"
            + solution
            + "\n```"
        )
        record = self.benchmark_record(
            raw, task_id="HumanEval/17", entry_point="parse_music"
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            completed, samples_path, report_path = self.run_exporter(
                Path(temp_dir), [record]
            )

            self.assertEqual(completed.returncode, 0)
            sample = json.loads(samples_path.read_text(encoding="utf-8"))
            self.assertEqual(sample, {"task_id": "HumanEval/17", "solution": solution})
            report = json.loads(report_path.read_text(encoding="utf-8"))
            case = report["cases"][0]
            self.assertEqual(case["strategy"], "reasoning_prefix_then_python_fence")
            self.assertEqual(case["raw_completion_sha256"], sha256_text(raw))
            self.assertEqual(case["solution_sha256"], sha256_text(solution))
            self.assertEqual(case["warnings"], ["reasoning_tags_present", "markdown_fence_present"])

    def test_one_malformed_task_blocks_partial_sample_export(self) -> None:
        valid = self.benchmark_record(
            "def has_close_elements(numbers, threshold):\n    return False\n"
        )
        malformed = self.benchmark_record(
            "Here is the solution:\ndef parse_music(music_string):\n    return []\n",
            task_id="HumanEval/17",
            entry_point="parse_music",
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            completed, samples_path, report_path = self.run_exporter(
                Path(temp_dir), [valid, malformed]
            )

            self.assertEqual(completed.returncode, 4)
            self.assertFalse(samples_path.exists())
            report = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertFalse(report["passed"])
            self.assertEqual(report["selected_record_count"], 2)
            self.assertEqual(report["failed_extraction_count"], 1)
            self.assertIsNone(report["samples_output"])
            self.assertEqual(
                [(case["task_id"], case["status"]) for case in report["cases"]],
                [
                    ("HumanEval/0", "extracted"),
                    ("HumanEval/17", "extraction_failure"),
                ],
            )
            self.assertEqual(report["cases"][1]["error_type"], "SyntaxError")

    def test_duplicate_task_ids_block_export(self) -> None:
        first = self.benchmark_record(
            "def has_close_elements(numbers, threshold):\n    return False\n"
        )
        second = self.benchmark_record(
            "def has_close_elements(numbers, threshold):\n    return True\n"
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            completed, samples_path, report_path = self.run_exporter(
                Path(temp_dir), [first, second]
            )

            self.assertEqual(completed.returncode, 4)
            self.assertFalse(samples_path.exists())
            report = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertFalse(report["passed"])
            self.assertEqual(report["duplicate_task_ids"], ["HumanEval/0"])
            self.assertEqual(report["exported_sample_count"], 0)
            self.assertTrue(
                all(case["error_type"] == "DuplicateTaskId" for case in report["cases"])
            )

    def test_prompt_identity_mismatch_blocks_export(self) -> None:
        record = self.benchmark_record(
            "def has_close_elements(numbers, threshold):\n    return False\n"
        )
        record["prompt_sha256"] = "0" * 64
        with tempfile.TemporaryDirectory() as temp_dir:
            completed, samples_path, report_path = self.run_exporter(
                Path(temp_dir), [record]
            )

            self.assertEqual(completed.returncode, 4)
            self.assertFalse(samples_path.exists())
            report = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertEqual(report["failed_extraction_count"], 1)
            self.assertEqual(report["cases"][0]["error_type"], "ValueError")
            self.assertIn("prompt_sha256 mismatch", report["cases"][0]["error_message"])


if __name__ == "__main__":
    unittest.main()
