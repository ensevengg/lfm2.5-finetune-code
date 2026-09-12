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
BUILDER = WORKSPACE / "scripts" / "build_e3_e4_suites.py"
sys.path.insert(0, str(WORKSPACE / "scripts"))

from build_e3_e4_suites import (  # noqa: E402
    BURNED_TASK_IDS,
    EXPECTED_PILOT_LIST_SHA256,
    canonical_task_list_sha256,
    numeric_task_key,
    select_slices,
)


def canonical_hash(task_ids: list[str]) -> str:
    """Independent reimplementation of the canonical frozen-list form."""
    canonical = "".join(
        f"{task_id}\n" for task_id in sorted(task_ids, key=numeric_task_key)
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class SelectSliceTests(unittest.TestCase):
    def test_real_universe_matches_the_frozen_plan(self) -> None:
        burned = {0, 17, 119}
        eligible = [index for index in range(164) if index not in burned]
        expected_pilot = [f"HumanEval/{index}" for index in eligible[::5]]
        expected_sealed = [
            f"HumanEval/{index}" for index in eligible if index not in set(eligible[::5])
        ]
        expected_hash = hashlib.sha256(
            "".join(f"HumanEval/{index}\n" for index in eligible[::5]).encode("utf-8")
        ).hexdigest()

        pilot, sealed = select_slices(
            [f"HumanEval/{index}" for index in range(164)], list(BURNED_TASK_IDS)
        )
        self.assertEqual(len(pilot), 33)
        self.assertEqual(len(sealed), 128)
        self.assertEqual(pilot, expected_pilot)
        self.assertEqual(sealed, expected_sealed)
        self.assertEqual(canonical_task_list_sha256(pilot), expected_hash)
        # The hash frozen in the Step 5 plan must equal the independent derivation.
        self.assertEqual(expected_hash, EXPECTED_PILOT_LIST_SHA256)
        self.assertEqual(pilot[0], "HumanEval/1")
        self.assertEqual(pilot[4], "HumanEval/22")
        self.assertEqual(pilot[-1], "HumanEval/163")

    def test_slices_partition_the_eligible_pool(self) -> None:
        universe = [f"HumanEval/{index}" for index in range(164)]
        pilot, sealed = select_slices(universe, list(BURNED_TASK_IDS))
        self.assertEqual(len(set(pilot) & set(sealed)), 0)
        self.assertEqual(
            set(pilot) | set(sealed) | set(BURNED_TASK_IDS), set(universe)
        )

    def test_selection_is_order_independent(self) -> None:
        universe = [f"HumanEval/{index}" for index in range(164)]
        pilot_a, sealed_a = select_slices(universe, list(BURNED_TASK_IDS))
        pilot_b, sealed_b = select_slices(list(reversed(universe)), list(BURNED_TASK_IDS))
        self.assertEqual(pilot_a, pilot_b)
        self.assertEqual(sealed_a, sealed_b)

    def test_stride_must_leave_a_sealed_slice(self) -> None:
        universe = [f"HumanEval/{index}" for index in range(164)]
        with self.assertRaises(ValueError):
            select_slices(universe, list(BURNED_TASK_IDS), stride=1)

    def test_non_numeric_task_ids_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            select_slices(["HumanEval/abc"], [])


class BuilderCliTests(unittest.TestCase):
    # Fake universe: HumanEval/0..29. Burned IDs present in it: 0 and 17.
    # Eligible = 28 IDs; stride 5 pilot = [1, 6, 11, 16, 21, 26]; sealed = 22.
    UNIVERSE_SIZE = 30
    EXPECTED_PILOT_COUNT = 6
    EXPECTED_SEALED_COUNT = 22

    def build_workspace(self) -> Path:
        root = Path(tempfile.mkdtemp(prefix="e3-builder-"))
        dataset_dir = root / "HumanEvalPlus.jsonl"
        dataset_dir.mkdir()
        dataset = dataset_dir / "HumanEvalPlus.jsonl"
        lines = [
            json.dumps({"task_id": f"HumanEval/{index}", "prompt": "x"})
            for index in range(self.UNIVERSE_SIZE)
        ]
        dataset.write_text("\n".join(lines) + "\n", encoding="utf-8")

        prompts = root / "benchmarks" / "prompts"
        prompts.mkdir(parents=True)
        template = prompts / "lfm-code-v1.txt"
        template.write_text("Task: {{BENCHMARK_PROMPT}}\n", encoding="utf-8")
        manifest = prompts / "lfm-code-v1.json"
        manifest.write_text(
            json.dumps({"prompt_id": "lfm-code-v1", "placeholder": "{{BENCHMARK_PROMPT}}"})
            + "\n",
            encoding="utf-8",
        )
        return root

    def run_builder(self, root: Path, extra: list[str]) -> subprocess.CompletedProcess[str]:
        dataset = root / "HumanEvalPlus.jsonl" / "HumanEvalPlus.jsonl"
        template = root / "benchmarks" / "prompts" / "lfm-code-v1.txt"
        manifest = root / "benchmarks" / "prompts" / "lfm-code-v1.json"
        command = [
            sys.executable,
            str(BUILDER),
            "--workspace", str(root),
            "--dataset", str(dataset),
            "--prompt-template", str(template),
            "--prompt-manifest", str(manifest),
            "--skip-model-rows",
            "--dataset-sha256", hashlib.sha256(dataset.read_bytes()).hexdigest(),
            "--dataset-size", str(dataset.stat().st_size),
            "--dataset-records", str(self.UNIVERSE_SIZE),
            "--expect-burned-present", "HumanEval/0,HumanEval/17",
            "--prompt-template-sha256", hashlib.sha256(template.read_bytes()).hexdigest(),
            "--prompt-manifest-sha256", hashlib.sha256(manifest.read_bytes()).hexdigest(),
        ] + extra
        return subprocess.run(command, capture_output=True, text=True, check=False)

    def expected_pilot_hash(self) -> str:
        universe = [f"HumanEval/{index}" for index in range(self.UNIVERSE_SIZE)]
        pilot, _ = select_slices(universe, list(BURNED_TASK_IDS))
        return canonical_hash(pilot)

    def test_builder_writes_deterministic_suites(self) -> None:
        root = self.build_workspace()
        try:
            completed = self.run_builder(root, ["--pilot-hash", self.expected_pilot_hash()])
            self.assertEqual(completed.returncode, 0, completed.stderr)
            pilot_suite = json.loads(
                (root / "benchmarks" / "suites" / "e3-humanevalplus-pilot-v1.json")
                .read_text(encoding="utf-8")
            )
            sealed_suite = json.loads(
                (root / "benchmarks" / "suites" / "e4-humanevalplus-sealed-v1.json")
                .read_text(encoding="utf-8")
            )
            self.assertEqual(pilot_suite["suite_id"], "e3-humanevalplus-pilot-v1")
            self.assertEqual(pilot_suite["pilot_task_count"], self.EXPECTED_PILOT_COUNT)
            self.assertEqual(pilot_suite["pilot_task_ids"][0], "HumanEval/1")
            self.assertEqual(pilot_suite["pilot_task_ids"][-1], "HumanEval/27")
            self.assertNotIn("HumanEval/1", sealed_suite["sealed_task_ids"])
            self.assertNotIn("HumanEval/17", sealed_suite["sealed_task_ids"])
            self.assertEqual(
                sealed_suite["sealed_task_count"], self.EXPECTED_SEALED_COUNT
            )
            self.assertEqual(
                sealed_suite["burned_task_ids"],
                sorted(
                    set(BURNED_TASK_IDS) | set(pilot_suite["pilot_task_ids"]),
                    key=numeric_task_key,
                ),
            )
            self.assertEqual(
                pilot_suite["selection_rule"]["canonical_list_sha256"],
                self.expected_pilot_hash(),
            )
        finally:
            shutil.rmtree(root, ignore_errors=True)

    def test_builder_rejects_a_drifted_dataset(self) -> None:
        root = self.build_workspace()
        try:
            completed = self.run_builder(
                root,
                ["--dataset-sha256", "0" * 64, "--pilot-hash", self.expected_pilot_hash()],
            )
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("identity mismatch", completed.stderr)
        finally:
            shutil.rmtree(root, ignore_errors=True)

    def test_builder_rejects_a_drifted_pilot_hash(self) -> None:
        root = self.build_workspace()
        try:
            completed = self.run_builder(root, ["--pilot-hash", "1" * 64])
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("drifted", completed.stderr)
        finally:
            shutil.rmtree(root, ignore_errors=True)

    def test_builder_refuses_to_overwrite_frozen_suites(self) -> None:
        root = self.build_workspace()
        try:
            first = self.run_builder(root, ["--pilot-hash", self.expected_pilot_hash()])
            self.assertEqual(first.returncode, 0, first.stderr)
            second = self.run_builder(root, ["--pilot-hash", self.expected_pilot_hash()])
            self.assertNotEqual(second.returncode, 0)
            self.assertIn("Refusing to overwrite", second.stderr)
            third = self.run_builder(
                root, ["--pilot-hash", self.expected_pilot_hash(), "--force"]
            )
            self.assertEqual(third.returncode, 0, third.stderr)
        finally:
            shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
