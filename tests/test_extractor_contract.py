from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path


WORKSPACE = Path(__file__).resolve().parents[1]
MANIFEST = WORKSPACE / "benchmarks" / "extractors" / "python-solution-v1.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class ExtractorContractTests(unittest.TestCase):
    def test_python_solution_v1_implementation_is_frozen(self) -> None:
        manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
        implementation = WORKSPACE / manifest["implementation"]["path"]

        self.assertEqual(
            implementation.stat().st_size,
            manifest["implementation"]["size_bytes"],
        )
        self.assertEqual(sha256(implementation), manifest["implementation"]["sha256"])
        self.assertFalse(manifest["safety"]["executes_generated_programs"])
        self.assertTrue(manifest["batch_policy"]["all_selected_records_must_extract"])
        self.assertEqual(
            manifest["output_schema"]["exact_fields"], ["task_id", "solution"]
        )


if __name__ == "__main__":
    unittest.main()
