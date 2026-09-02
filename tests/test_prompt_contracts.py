from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path


WORKSPACE = Path(__file__).resolve().parents[1]
V0 = WORKSPACE / "benchmarks" / "prompts" / "lfm-code-v0.txt"
V1_MANIFEST = WORKSPACE / "benchmarks" / "prompts" / "lfm-code-v1.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class PromptContractTests(unittest.TestCase):
    def test_v1_plain_python_contract_is_frozen_and_v0_is_unchanged(self) -> None:
        manifest = json.loads(V1_MANIFEST.read_text(encoding="utf-8"))
        prompt = WORKSPACE / manifest["path"]
        content = prompt.read_text(encoding="utf-8")

        self.assertEqual(
            sha256(V0),
            "bd97a6ce8e0a574a4426815d9e73ecea2a42beaea9f82ebdfcbbf04a88a8a788",
        )
        self.assertEqual(prompt.stat().st_size, manifest["size_bytes"])
        self.assertEqual(sha256(prompt), manifest["sha256"])
        self.assertEqual(content.count(manifest["placeholder"]), 1)
        self.assertIn("Return only valid Python source code.", content)
        self.assertIn("Do not use Markdown code fences.", content)
        self.assertIn("Do not include explanations, reasoning, analysis, or <think> tags.", content)
        self.assertFalse(manifest["output_contract"]["markdown_code_fences_allowed"])
        self.assertTrue(manifest["output_contract"]["raw_completion_must_remain_unchanged"])


if __name__ == "__main__":
    unittest.main()
