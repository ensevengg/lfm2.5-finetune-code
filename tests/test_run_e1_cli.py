from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


WORKSPACE = Path(__file__).resolve().parents[1]
RUNNER = WORKSPACE / "scripts" / "run_e1.py"
REAL_SUITE = WORKSPACE / "benchmarks" / "suites" / "e0-humanevalplus-smoke-v1.json"
V2_SUITE = WORKSPACE / "benchmarks" / "suites" / "e1-humanevalplus-smoke-v2.json"
sys.path.insert(0, str(WORKSPACE / "scripts"))
import run_e1  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class RunE1CliTests(unittest.TestCase):
    def make_fixture(
        self, root: Path, source_suite: Path = REAL_SUITE
    ) -> tuple[Path, Path, Path, Path, Path]:
        prompts = root / "prompts.jsonl"
        records = [
            {
                "task_id": "HumanEval/0",
                "entry_point": "has_close_elements",
                "prompt": "def has_close_elements():\n    pass\n",
            },
            {
                "task_id": "HumanEval/17",
                "entry_point": "parse_music",
                "prompt": "def parse_music():\n    pass\n",
            },
            {
                "task_id": "HumanEval/119",
                "entry_point": "match_parens",
                "prompt": "def match_parens():\n    pass\n",
            },
        ]
        prompts.write_text(
            "".join(json.dumps(record, separators=(",", ":")) + "\n" for record in records),
            encoding="utf-8",
        )

        prompt_template = root / "prompt.txt"
        prompt_template.write_text(
            "Complete this function.\n{{BENCHMARK_PROMPT}}",
            encoding="utf-8",
        )

        model = root / "model.gguf"
        model.write_bytes(b"not-a-real-model")

        suite = json.loads(source_suite.read_text(encoding="utf-8"))
        suite["benchmark"]["generation_asset"] = {
            "path": "evals/E1_smoketest.jsonl",
            "sha256": sha256(prompts),
            "size_bytes": prompts.stat().st_size,
            "record_count": 3,
            "allowed_fields": ["task_id", "entry_point", "prompt"],
        }
        suite["prompt"]["sha256"] = sha256(prompt_template)
        suite["model"]["sha256"] = sha256(model)
        suite["model"]["size_bytes"] = model.stat().st_size
        suite_path = root / "suite.json"
        suite_path.write_text(json.dumps(suite), encoding="utf-8")

        results = root / "results"
        return suite_path, prompts, prompt_template, model, results

    def test_validate_only_accepts_frozen_assets(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            suite, prompts, prompt_template, model, results = self.make_fixture(
                Path(temp_dir)
            )

            completed = subprocess.run(
                [
                    sys.executable,
                    str(RUNNER),
                    "--suite",
                    str(suite),
                    "--prompts",
                    str(prompts),
                    "--prompt-template",
                    str(prompt_template),
                    "--model",
                    str(model),
                    "--results-dir",
                    str(results),
                    "--run-id",
                    "test-validation",
                    "--validate-only",
                ],
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(
                completed.returncode,
                0,
                msg=f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}",
            )
            report = json.loads((results / "validation.json").read_text(encoding="utf-8"))
            self.assertTrue(report["passed"])
            self.assertEqual(report["summary"]["prompt_record_count"], 3)

    def test_validate_only_accepts_prompt_v1_suite(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            suite, prompts, prompt_template, model, results = self.make_fixture(
                Path(temp_dir), V2_SUITE
            )

            completed = subprocess.run(
                [
                    sys.executable,
                    str(RUNNER),
                    "--suite",
                    str(suite),
                    "--prompts",
                    str(prompts),
                    "--prompt-template",
                    str(prompt_template),
                    "--model",
                    str(model),
                    "--results-dir",
                    str(results),
                    "--run-id",
                    "test-v2-validation",
                    "--validate-only",
                ],
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(
                completed.returncode,
                0,
                msg=f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}",
            )
            report = json.loads((results / "validation.json").read_text(encoding="utf-8"))
            self.assertEqual(report["summary"]["prompt_id"], "lfm-code-v1")
            self.assertEqual(report["summary"]["extractor_id"], "python-solution-v1")

    def test_generation_record_carries_prompt_identity_without_executing_code(self) -> None:
        suite = {
            "_run_id": "test-run",
            "_mode": "primary",
            "prompt": {
                "prompt_id": "lfm-code-v1",
                "sha256": "f3d4b17279edb68261ae2a7094cfce64bd8c236397b3d0f548c7e01c15398cf6",
                "bos_token_id": 1,
            },
            "generation": {
                "max_output_tokens": 32,
                "temperature": 0.0,
                "seed": 42,
                "stream": False,
            },
            "runtime": {"context_size": 128},
        }
        case = {
            "case_id": "humanevalplus-HumanEval-0",
            "case_kind": "humanevalplus_development",
            "task_id": "HumanEval/0",
            "entry_point": "has_close_elements",
            "messages": [{"role": "user", "content": "Return Python."}],
            "benchmark_prompt": "def has_close_elements():\n    pass\n",
        }
        responses = [
            {"prompt": "rendered"},
            {"tokens": [1, 2, 3]},
            {
                "content": "def has_close_elements():\n    return False\n",
                "tokens": [4, 5],
                "stop_type": "eos",
                "truncated": False,
            },
        ]

        with patch.object(run_e1, "http_json", side_effect=responses):
            record = run_e1.generate_case("http://unused", case, suite)

        self.assertEqual(record["prompt_id"], "lfm-code-v1")
        self.assertEqual(record["prompt_sha256"], suite["prompt"]["sha256"])
        self.assertFalse(record["generated_program_executed"])
        self.assertFalse(record["scored"])


if __name__ == "__main__":
    unittest.main()
