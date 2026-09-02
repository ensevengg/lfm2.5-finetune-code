from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path


WORKSPACE = Path(__file__).resolve().parents[1]
SUITE_PATH = WORKSPACE / "benchmarks" / "suites" / "e1-humanevalplus-smoke-v2.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class SuiteV2ContractTests(unittest.TestCase):
    def test_prompt_and_extractor_identities_are_frozen(self) -> None:
        suite = json.loads(SUITE_PATH.read_text(encoding="utf-8"))
        self.assertEqual(suite["suite_id"], "e1-humanevalplus-smoke-v2")

        prompt = suite["prompt"]
        prompt_template = WORKSPACE / prompt["path"]
        prompt_manifest = WORKSPACE / prompt["manifest_path"]
        self.assertEqual(prompt["prompt_id"], "lfm-code-v1")
        self.assertEqual(sha256(prompt_template), prompt["sha256"])
        self.assertEqual(prompt_template.stat().st_size, prompt["size_bytes"])
        self.assertEqual(sha256(prompt_manifest), prompt["manifest_sha256"])
        self.assertEqual(prompt_manifest.stat().st_size, prompt["manifest_size_bytes"])

        extractor = suite["extractor"]
        extractor_manifest = WORKSPACE / extractor["manifest_path"]
        implementation = WORKSPACE / extractor["implementation_path"]
        self.assertEqual(extractor["extractor_id"], "python-solution-v1")
        self.assertEqual(sha256(extractor_manifest), extractor["manifest_sha256"])
        self.assertEqual(
            extractor_manifest.stat().st_size, extractor["manifest_size_bytes"]
        )
        self.assertEqual(sha256(implementation), extractor["implementation_sha256"])
        self.assertEqual(
            implementation.stat().st_size, extractor["implementation_size_bytes"]
        )
        self.assertFalse(suite["execution"]["generated_code_execution_allowed"])
        self.assertFalse(suite["execution"]["scoring_allowed"])


if __name__ == "__main__":
    unittest.main()
