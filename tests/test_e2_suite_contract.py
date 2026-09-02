from __future__ import annotations

import hashlib
import json
import subprocess
import unittest
from pathlib import Path


WORKSPACE = Path(__file__).resolve().parents[1]
SUITE_PATH = WORKSPACE / "benchmarks" / "suites" / "e2-humanevalplus-dev-v1.json"
RUNNER = WORKSPACE / "scripts" / "run_e2.ps1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def assert_identity(
    case: unittest.TestCase, workspace: Path, contract: dict[str, object]
) -> None:
    path = workspace / str(contract["path"])
    case.assertEqual(path.stat().st_size, contract["size_bytes"])
    case.assertEqual(sha256(path), contract["sha256"])


class E2SuiteContractTests(unittest.TestCase):
    def test_source_and_scorer_files_are_frozen(self) -> None:
        suite = json.loads(SUITE_PATH.read_text(encoding="utf-8"))
        self.assertEqual(suite["suite_id"], "e2-humanevalplus-dev-v1")
        assert_identity(self, WORKSPACE, suite["source"]["generations"])
        assert_identity(self, WORKSPACE, suite["source"]["extraction_report"])
        assert_identity(self, WORKSPACE, suite["source"]["samples"])
        assert_identity(self, WORKSPACE, suite["dataset"])
        assert_identity(self, WORKSPACE, suite["scorer"]["dockerfile"])

        preparer = suite["preparation"]
        preparer_path = WORKSPACE / preparer["implementation_path"]
        self.assertEqual(preparer_path.stat().st_size, preparer["implementation_size_bytes"])
        self.assertEqual(sha256(preparer_path), preparer["implementation_sha256"])
        self.assertFalse(preparer["executes_generated_programs"])
        self.assertFalse(suite["safety"]["validation_executes_generated_programs"])
        self.assertTrue(suite["safety"]["scoring_executes_generated_programs"])

    def test_runner_requires_explicit_execution_confirmation(self) -> None:
        completed = subprocess.run(
            ["pwsh", "-NoProfile", "-File", str(RUNNER), "-TaskId", "HumanEval/0"],
            cwd=WORKSPACE,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("-ConfirmExecution", completed.stderr)


if __name__ == "__main__":
    unittest.main()
