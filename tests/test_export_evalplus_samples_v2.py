from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parents[1]
V1 = WORKSPACE / "scripts" / "export_evalplus_samples.py"
V1_MANIFEST = WORKSPACE / "benchmarks" / "extractors" / "python-solution-v1.json"
V2 = WORKSPACE / "scripts" / "export_evalplus_samples_v2.py"
V2_MANIFEST = WORKSPACE / "benchmarks" / "extractors" / "python-solution-v2.json"
PROMPT_MANIFEST = WORKSPACE / "benchmarks" / "prompts" / "lfm-code-v1.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def generation_record(
    task_id: str,
    raw_completion: str,
    entry_point: str = "foo",
    case_kind: str = "humanevalplus_pilot",
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "suite_id": "e3-humanevalplus-pilot-v1",
        "case_kind": case_kind,
        "task_id": task_id,
        "entry_point": entry_point,
        "prompt_id": "lfm-code-v1",
        "prompt_sha256": "f3d4b17279edb68261ae2a7094cfce64bd8c236397b3d0f548c7e01c15398cf6",
        "raw_completion": raw_completion,
    }


class FreezeDisciplineTests(unittest.TestCase):
    def test_v1_files_remain_byte_identical_to_their_frozen_hashes(self) -> None:
        manifest = json.loads(V1_MANIFEST.read_text(encoding="utf-8"))
        implementation = manifest["implementation"]
        self.assertEqual(sha256(V1), implementation["sha256"])
        self.assertEqual(V1.stat().st_size, implementation["size_bytes"])

    def test_v2_manifest_matches_the_implementation(self) -> None:
        manifest = json.loads(V2_MANIFEST.read_text(encoding="utf-8"))
        implementation = manifest["implementation"]
        self.assertEqual(sha256(V2), implementation["sha256"])
        self.assertEqual(V2.stat().st_size, implementation["size_bytes"])
        self.assertEqual(
            implementation["v1_implementation_sha256"],
            json.loads(V1_MANIFEST.read_text(encoding="utf-8"))["implementation"]["sha256"],
        )


class V2ExtractorCliTests(unittest.TestCase):
    def run_v2(
        self, records: list[dict[str, object]], root: Path
    ) -> subprocess.CompletedProcess[str]:
        generations = root / "generations.jsonl"
        generations.write_text(
            "".join(json.dumps(record) + "\n" for record in records),
            encoding="utf-8",
        )
        return subprocess.run(
            [
                sys.executable, str(V2),
                "--generations", str(generations),
                "--prompt-manifest", str(PROMPT_MANIFEST),
                "--samples-output", str(root / "samples.jsonl"),
                "--report-output", str(root / "extraction-report.json"),
            ],
            capture_output=True, text=True, check=False,
        )

    def test_failures_continue_the_batch_and_are_recorded(self) -> None:
        root = Path(tempfile.mkdtemp(prefix="e3-extract-"))
        try:
            records = [
                generation_record("HumanEval/1", "def foo():\n    return 1\n"),
                generation_record(
                    "HumanEval/6", "def foo(:\n  return 1\n"
                ),
                generation_record(
                    "HumanEval/11",
                    "<think>reasoning</think>\n```python\ndef foo():\n    return 2\n```",
                ),
                generation_record(
                    "HumanEval/16", "def bar():\n    return 3\n"
                ),
            ]
            completed = self.run_v2(records, root)
            self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
            report = json.loads(
                (root / "extraction-report.json").read_text(encoding="utf-8")
            )
            self.assertTrue(report["passed"])
            self.assertEqual(report["selected_record_count"], 4)
            self.assertEqual(report["successful_extraction_count"], 2)
            self.assertEqual(report["failed_extraction_count"], 2)
            self.assertEqual(report["failed_task_ids"], ["HumanEval/16", "HumanEval/6"])
            self.assertEqual(report["extraction_yield"], 0.5)
            self.assertFalse(report["generated_programs_executed"])

            samples = [
                json.loads(line)
                for line in (root / "samples.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(
                [sample["task_id"] for sample in samples], ["HumanEval/1", "HumanEval/11"]
            )
            for sample in samples:
                self.assertEqual(set(sample.keys()), {"task_id", "solution"})
        finally:
            shutil.rmtree(root, ignore_errors=True)

    def test_duplicate_task_ids_are_batch_fatal(self) -> None:
        root = Path(tempfile.mkdtemp(prefix="e3-extract-"))
        try:
            records = [
                generation_record("HumanEval/1", "def foo():\n    return 1\n"),
                generation_record("HumanEval/1", "def foo():\n    return 2\n"),
            ]
            completed = self.run_v2(records, root)
            self.assertNotEqual(completed.returncode, 0)
            self.assertFalse((root / "samples.jsonl").exists())
            report = json.loads(
                (root / "extraction-report.json").read_text(encoding="utf-8")
            )
            self.assertFalse(report["passed"])
            self.assertIn("Duplicate task IDs", report["error"])
        finally:
            shutil.rmtree(root, ignore_errors=True)

    def test_prompt_identity_mismatch_is_per_task_not_fatal(self) -> None:
        root = Path(tempfile.mkdtemp(prefix="e3-extract-"))
        try:
            good = generation_record("HumanEval/1", "def foo():\n    return 1\n")
            bad = generation_record("HumanEval/6", "def foo():\n    return 2\n")
            bad["prompt_sha256"] = "0" * 64
            completed = self.run_v2([good, bad], root)
            self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
            report = json.loads(
                (root / "extraction-report.json").read_text(encoding="utf-8")
            )
            self.assertTrue(report["passed"])
            self.assertEqual(report["failed_task_ids"], ["HumanEval/6"])
        finally:
            shutil.rmtree(root, ignore_errors=True)

    def test_refuses_to_overwrite_existing_evidence(self) -> None:
        root = Path(tempfile.mkdtemp(prefix="e3-extract-"))
        try:
            (root / "samples.jsonl").write_text("", encoding="utf-8")
            completed = self.run_v2(
                [generation_record("HumanEval/1", "def foo():\n    return 1\n")], root
            )
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("Refusing to overwrite", completed.stdout)
        finally:
            shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
