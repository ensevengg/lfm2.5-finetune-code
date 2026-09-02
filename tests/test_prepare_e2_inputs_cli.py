from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


WORKSPACE = Path(__file__).resolve().parents[1]
PREPARER = WORKSPACE / "scripts" / "prepare_e2_inputs.py"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def identity(path: Path, **extra: object) -> dict[str, object]:
    return {
        "path": str(path.name),
        "sha256": sha256(path),
        "size_bytes": path.stat().st_size,
        **extra,
    }


class PrepareE2InputsCliTests(unittest.TestCase):
    def make_fixture(self, root: Path) -> Path:
        task_ids = ["HumanEval/0", "HumanEval/17", "HumanEval/119"]
        generations = root / "generations.jsonl"
        generations.write_text("{}\n", encoding="utf-8")
        extraction = root / "extraction-report.json"
        extraction.write_text(
            json.dumps(
                {
                    "passed": True,
                    "selected_record_count": 3,
                    "exported_sample_count": 3,
                    "generated_programs_executed": False,
                }
            ),
            encoding="utf-8",
        )
        samples = root / "samples.jsonl"
        samples.write_text(
            "".join(
                json.dumps({"task_id": task_id, "solution": "def f():\n    pass"})
                + "\n"
                for task_id in task_ids
            ),
            encoding="utf-8",
        )
        dataset = root / "dataset.jsonl"
        dataset.write_text(
            "".join(
                json.dumps(
                    {
                        "task_id": task_id,
                        "prompt": "def f():\n",
                        "canonical_solution": "    pass\n",
                        "base_input": [],
                        "plus_input": [],
                        "entry_point": "f",
                        "contract": "",
                        "atol": 0,
                    }
                )
                + "\n"
                for task_id in task_ids
            ),
            encoding="utf-8",
        )
        suite = {
            "schema_version": 1,
            "suite_id": "e2-humanevalplus-dev-v1",
            "default_manual_task_id": "HumanEval/0",
            "development_task_ids": task_ids,
            "source": {
                "generations": identity(generations),
                "extraction_report": identity(
                    extraction,
                    selected_record_count=3,
                    exported_sample_count=3,
                ),
                "samples": identity(
                    samples,
                    record_count=3,
                    exact_fields=["task_id", "solution"],
                ),
            },
            "dataset": identity(
                dataset,
                record_count=3,
                unique_task_id_count=3,
            ),
        }
        suite_path = root / "suite.json"
        suite_path.write_text(json.dumps(suite), encoding="utf-8")
        return suite_path

    def run_preparer(
        self, root: Path, suite: Path, task_id: str
    ) -> tuple[subprocess.CompletedProcess[str], Path]:
        output = root / "prepared"
        completed = subprocess.run(
            [
                sys.executable,
                str(PREPARER),
                "--suite",
                str(suite),
                "--workspace",
                str(root),
                "--output-dir",
                str(output),
                "--task-id",
                task_id,
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        return completed, output

    def test_prepares_one_matching_task_without_execution(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            suite = self.make_fixture(root)
            completed, output = self.run_preparer(root, suite, "HumanEval/17")

            self.assertEqual(completed.returncode, 0, completed.stderr)
            sample = json.loads((output / "samples.jsonl").read_text(encoding="utf-8"))
            dataset = json.loads(
                (output / "HumanEvalPlus-subset.jsonl").read_text(encoding="utf-8")
            )
            self.assertEqual(sample["task_id"], "HumanEval/17")
            self.assertEqual(dataset["task_id"], "HumanEval/17")
            report = json.loads(
                (output / "preparation-report.json").read_text(encoding="utf-8")
            )
            self.assertEqual(report["selected_task_ids"], ["HumanEval/17"])
            self.assertFalse(report["generated_programs_executed"])
            self.assertFalse(report["hidden_tests_printed"])

    def test_rejects_task_outside_frozen_development_set(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            suite = self.make_fixture(root)
            completed, output = self.run_preparer(root, suite, "HumanEval/1")

            self.assertNotEqual(completed.returncode, 0)
            self.assertFalse(output.exists())
            self.assertIn("outside the frozen development set", completed.stderr)


if __name__ == "__main__":
    unittest.main()
