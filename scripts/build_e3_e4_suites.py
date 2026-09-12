#!/usr/bin/env python3
"""Freeze the E3 pilot and E4 sealed HumanEval+ suites from the Step 5 plan.

The selection rule is the one frozen in
docs/evaluation/step5-pilot-and-sealed-benchmark-plan.md: from the dataset
task IDs, remove the permanently burned development IDs, sort ascending, and
take every Nth (0-based) as the pilot slice. Pilot IDs become burned after E3,
so the sealed slice is the remainder. The builder verifies dataset, prompt,
extractor, and model-row identities before writing anything and refuses to
overwrite an existing suite unless --force is given twice-verified intent.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

BURNED_TASK_IDS = ("HumanEval/0", "HumanEval/17", "HumanEval/119")
PILOT_STRIDE = 5
EXPECTED_PILOT_LIST_SHA256 = (
    "d22bef9489c7643b6f61a4b8e0404cdf3499e9be57f0ac7d0dd4c39d7a4119c2"
)
EXPECTED_DATASET_SHA256 = (
    "42526ec0e7d5f3ee0b06d6ced98f8c8bae3d76519151bfb3d36f79010645bd7f"
)
EXPECTED_DATASET_SIZE = 7714666
EXPECTED_PROMPT_TEMPLATE_SHA256 = (
    "f3d4b17279edb68261ae2a7094cfce64bd8c236397b3d0f548c7e01c15398cf6"
)
EXPECTED_PROMPT_TEMPLATE_SIZE = 290
EXPECTED_PROMPT_MANIFEST_SHA256 = (
    "35acbdda228acc373e802345a98cdf4ee2616274a6be73db9108db7c7ba314ed"
)
EXPECTED_PROMPT_MANIFEST_SIZE = 939
EXPECTED_QUANT_SHA256 = {
    "q8_0": "3e2223fa4baf49896f1a58d0b73e02f5143b745d8deb71a01a6038b7994ce394",
    "q6_k": "d1a14fa33fba75118c8262827bfcadc89d0c549d628ac243014df81a14d8f331",
}
EXPECTED_MERGED_SHA256 = (
    "3e39c292be1f2b2740c3e05a2cb9081260439033e1b05f601117c72c8a73b8e6"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def numeric_task_key(task_id: str) -> int:
    """Sort key for 'HumanEval/<n>' IDs: ascending numeric, per the frozen rule."""
    prefix, _, suffix = task_id.rpartition("/")
    if not suffix.isdigit():
        raise ValueError(f"Task ID is not in 'prefix/<number>' form: {task_id!r}")
    return int(suffix)


def select_slices(
    task_ids: list[str], burned_task_ids: list[str], stride: int = PILOT_STRIDE
) -> tuple[list[str], list[str]]:
    """Return (pilot, sealed) per the frozen stride rule."""
    if stride < 2:
        raise ValueError("Stride must be at least 2 to leave a sealed slice")
    eligible = sorted(set(task_ids) - set(burned_task_ids), key=numeric_task_key)
    pilot = eligible[::stride]
    pilot_set = set(pilot)
    sealed = [task_id for task_id in eligible if task_id not in pilot_set]
    return pilot, sealed


def canonical_task_list_sha256(task_ids: list[str]) -> str:
    canonical = "".join(
        f"{task_id}\n" for task_id in sorted(task_ids, key=numeric_task_key)
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def verify_dataset(
    path: Path,
    expected_sha256: str,
    expected_size: int,
    expected_records: int = 164,
) -> list[dict[str, Any]]:
    actual_hash = sha256_file(path)
    actual_size = path.stat().st_size
    if actual_hash != expected_sha256 or actual_size != expected_size:
        raise SystemExit(
            f"Dataset identity mismatch: {path} is {actual_size} bytes,"
            f" SHA-256 {actual_hash}"
        )
    records = []
    task_ids = []
    with path.open("r", encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                raise SystemExit(f"Blank dataset line at {path}:{number}")
            record = json.loads(line)
            task_id = record.get("task_id")
            if not task_id:
                raise SystemExit(f"Dataset line {number} has no task_id")
            task_ids.append(task_id)
            records.append(record)
    if len(task_ids) != expected_records or len(set(task_ids)) != expected_records:
        raise SystemExit(
            f"Expected {expected_records} unique dataset records; found"
            f" {len(task_ids)} lines with {len(set(task_ids))} unique IDs"
        )
    return records


def verify_prompt_identities(
    template_path: Path,
    manifest_path: Path,
    expected_template_sha256: str = EXPECTED_PROMPT_TEMPLATE_SHA256,
    expected_manifest_sha256: str = EXPECTED_PROMPT_MANIFEST_SHA256,
) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("prompt_id") != "lfm-code-v1":
        raise SystemExit(f"Unexpected prompt id in {manifest_path}")
    template_hash = sha256_file(template_path)
    if template_hash != expected_template_sha256:
        raise SystemExit(f"Prompt template identity mismatch: {template_hash}")
    manifest_hash = sha256_file(manifest_path)
    if manifest_hash != expected_manifest_sha256:
        raise SystemExit(f"Prompt manifest identity mismatch: {manifest_hash}")
    return {
        "prompt_id": "lfm-code-v1",
        "template_path": template_path.relative_to(template_path.parents[2]).as_posix(),
        "template_sha256": template_hash,
        "template_size_bytes": template_path.stat().st_size,
        "manifest_path": manifest_path.relative_to(manifest_path.parents[2]).as_posix(),
        "manifest_sha256": manifest_hash,
        "manifest_size_bytes": manifest_path.stat().st_size,
        "placeholder": "{{BENCHMARK_PROMPT}}",
        "bos_token_id": 1,
        "bos_policy": "Every tokenized request begins with exactly one BOS token.",
    }


def model_row(
    alias: str,
    path: Path,
    workspace: Path,
    expected_sha256: str,
    evidence: str,
    role: str,
) -> dict[str, Any]:
    actual_hash = sha256_file(path)
    if actual_hash != expected_sha256:
        raise SystemExit(
            f"Model row {alias} drifted: expected {expected_sha256}, got {actual_hash}"
        )
    return {
        "alias": alias,
        "role": role,
        "path": path.relative_to(workspace).as_posix(),
        "sha256": actual_hash,
        "size_bytes": path.stat().st_size,
        "evidence_report": evidence,
    }


def build_pilot_suite(
    pilot_ids: list[str],
    sealed_count: int,
    rows: list[dict[str, Any]],
    prompt_identity: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "suite_id": "e3-humanevalplus-pilot-v1",
        "status": "frozen-ready-for-runner-implementation",
        "purpose": (
            "Validate the E3 harness at scale, implement the frozen retry policy,"
            " measure extraction yield, and size the predeclared sealed gates."
            " Scores from this suite are tuning evidence and are never headline"
            " results; pilot task IDs become burned after E3."
        ),
        "development_only": True,
        "created_at": "2026-09-12",
        "plan": "docs/evaluation/step5-pilot-and-sealed-benchmark-plan.md",
        "burned_task_ids": list(BURNED_TASK_IDS),
        "pilot_task_ids": pilot_ids,
        "pilot_task_count": len(pilot_ids),
        "sealed_task_count_removable": sealed_count,
        "selection_rule": {
            "rule": "stride",
            "stride": PILOT_STRIDE,
            "offset": 0,
            "ordering": "ascending numeric HumanEval task id",
            "canonical_list_sha256": canonical_task_list_sha256(pilot_ids),
        },
        "dataset": {
            "name": "HumanEvalPlus",
            "release": "v0.1.10",
            "path": "HumanEvalPlus.jsonl/HumanEvalPlus.jsonl",
            "sha256": EXPECTED_DATASET_SHA256,
            "size_bytes": EXPECTED_DATASET_SIZE,
            "record_count": 164,
        },
        "prompt": prompt_identity,
        "extractor": {
            "extractor_id": "python-solution-v1",
            "manifest_path": "benchmarks/extractors/python-solution-v1.json",
            "implementation_path": "scripts/export_evalplus_samples.py",
        },
        "model_rows": rows,
        "runtime": {
            "toolchain_image": "lfm25/llama-cpp:3018a11e-cuda128-sm120",
            "toolchain_image_id": "sha256:550d53ae97c4d83a6c0fba17cd8f532aa6f775e9498d2452fa6f04e33707b4b0",
            "llama_cpp_commit": "3018a11e79e489b657dbb77c95694889ccff92df",
            "context_tokens": 4096,
            "max_output_tokens": 3072,
            "gpu_layers": "all",
            "flash_attention": True,
            "cache_type_k": "f16",
            "cache_type_v": "f16",
            "fit": "off",
            "parallel_slots": 1,
        },
        "generation": {
            "sampling": "greedy",
            "temperature": 0,
            "seed": 42,
            "samples_per_task": 1,
            "server_per_row": True,
        },
        "retry_policy": {
            "infrastructure_retry_allowed": 1,
            "infrastructure_retry_requires": "fresh server process for the row",
            "model_outcome_retry_allowed": False,
            "second_consecutive_infra_failure": "abort the row",
            "duplicate_generation_key": "reject unless explicit resume",
            "model_identity_mismatch": "abort before first request",
        },
        "extraction_policy": {
            "per_task_failure_record": "extraction_failure",
            "failure_scores_as": "incorrect",
            "batch_continues_after_task_failure": True,
            "repair_allowed": False,
            "raw_output_preserved": True,
            "executes_generated_programs": False,
        },
        "scoring": {
            "scorer_suite": "e2-humanevalplus-dev-v1 (same restricted container boundary)",
            "executes_generated_programs": True,
            "host_execution_allowed": False,
        },
        "acceptance_gate": {
            "terminal_record_per_task_per_row": True,
            "infra_failure_rate_max": 0.02,
            "extraction_yield_min": 0.95,
            "report_required": "pilot report with per-row yield, failures, timing",
            "thresholds_frozen_after": "pilot report, before any sealed request",
        },
    }


def build_sealed_suite(
    pilot_ids: list[str],
    sealed_ids: list[str],
    prompt_identity: dict[str, Any],
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "suite_id": "e4-humanevalplus-sealed-v1",
        "status": "frozen-ready-for-sealed-run",
        "purpose": (
            "Headline paired quality comparison on tasks untouched by any"
            " development, smoke, or pilot request. Gates below were frozen from"
            " the E3 pilot report before this suite is executed."
        ),
        "development_only": False,
        "created_at": "2026-09-12",
        "frozen_at": "2026-09-12",
        "plan": "docs/evaluation/step5-pilot-and-sealed-benchmark-plan.md",
        "pilot_report": "reports/e3/pilot-report.json",
        "burned_task_ids": sorted(
            list(BURNED_TASK_IDS) + list(pilot_ids), key=numeric_task_key
        ),
        "sealed_task_ids": sealed_ids,
        "sealed_task_count": len(sealed_ids),
        "selection_rule": {
            "rule": "eligible remainder",
            "ordering": "ascending numeric HumanEval task id",
            "canonical_list_sha256": canonical_task_list_sha256(sealed_ids),
        },
        "dataset": {
            "name": "HumanEvalPlus",
            "release": "v0.1.10",
            "path": "HumanEvalPlus.jsonl/HumanEvalPlus.jsonl",
            "sha256": EXPECTED_DATASET_SHA256,
            "size_bytes": EXPECTED_DATASET_SIZE,
            "record_count": 164,
        },
        "prompt": prompt_identity,
        "extractor": {
            "extractor_id": "python-solution-v2",
            "manifest_path": "benchmarks/extractors/python-solution-v2.json",
            "implementation_path": "scripts/export_evalplus_samples_v2.py",
        },
        "model_rows": rows,
        "runtime": {
            "toolchain_image": "lfm25/llama-cpp:3018a11e-cuda128-sm120",
            "toolchain_image_id": "sha256:550d53ae97c4d83a6c0fba17cd8f532aa6f775e9498d2452fa6f04e33707b4b0",
            "llama_cpp_commit": "3018a11e79e489b657dbb77c95694889ccff92df",
            "context_tokens": 4096,
            "max_output_tokens": 3072,
            "gpu_layers": "all",
            "flash_attention": True,
            "cache_type_k": "f16",
            "cache_type_v": "f16",
            "fit": "off",
            "parallel_slots": 1,
        },
        "generation": {
            "sampling": "greedy",
            "temperature": 0,
            "seed": 42,
            "samples_per_task": 1,
            "server_per_row": True,
        },
        "retry_policy": {
            "infrastructure_retry_allowed": 1,
            "infrastructure_retry_requires": "fresh server process for the row",
            "model_outcome_retry_allowed": False,
            "second_consecutive_infra_failure": "abort the row",
            "duplicate_generation_key": "reject unless explicit resume",
            "model_identity_mismatch": "abort before first request",
        },
        "extraction_policy": {
            "per_task_failure_record": "extraction_failure",
            "failure_scores_as": "incorrect",
            "batch_continues_after_task_failure": True,
            "repair_allowed": False,
            "raw_output_preserved": True,
            "executes_generated_programs": False,
        },
        "frozen_gates": {
            "deployment_quant": "q6_k",
            "deployment_quant_gate": (
                "Q6_K deployable iff sealed pass@1(plus) is within 2 percentage"
                " points of merged-bf16 AND no paired loss pattern absent from"
                " merged-bf16"
            ),
            "harness_infra_failure_rate_max": 0.02,
            "harness_extraction_yield": "reported per row as model evidence, not gated",
        },
        "scoring": {
            "scorer_suite": "e2-humanevalplus-dev-v1 (same restricted container boundary)",
            "executes_generated_programs": True,
            "host_execution_allowed": False,
        },
        "safety": {
            "scoring_executes_generated_programs": True,
            "host_execution_allowed": False,
            "automatic_retry_allowed": False,
        },
    }


def write_json(path: Path, value: dict[str, Any], force: bool) -> None:
    if path.exists() and not force:
        raise SystemExit(f"Refusing to overwrite frozen suite: {path} (use --force)")
    path.parent.mkdir(parents=True, exist_ok=True)
    canonical = json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    path.write_text(canonical, encoding="utf-8")
    print(f"WROTE {path} (sha256 {hashlib.sha256(canonical.encode()).hexdigest()[:16]}…)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--dataset", type=Path, default=None,
                        help="override dataset path (tests)")
    parser.add_argument("--prompt-template", type=Path, default=None,
                        help="override prompt template path (tests)")
    parser.add_argument("--prompt-manifest", type=Path, default=None,
                        help="override prompt manifest path (tests)")
    parser.add_argument("--baseline-report", type=Path, default=None,
                        help="override phase5 conversion report path (tests)")
    parser.add_argument("--pilot-hash", default=EXPECTED_PILOT_LIST_SHA256,
                        help="override expected canonical pilot hash (tests)")
    parser.add_argument("--prompt-template-sha256", default=EXPECTED_PROMPT_TEMPLATE_SHA256,
                        help="override expected prompt template hash (tests)")
    parser.add_argument("--prompt-manifest-sha256", default=EXPECTED_PROMPT_MANIFEST_SHA256,
                        help="override expected prompt manifest hash (tests)")
    parser.add_argument("--dataset-sha256", default=EXPECTED_DATASET_SHA256,
                        help="override expected dataset hash (tests)")
    parser.add_argument("--dataset-size", type=int, default=EXPECTED_DATASET_SIZE,
                        help="override expected dataset size (tests)")
    parser.add_argument("--dataset-records", type=int, default=164,
                        help="override expected unique dataset record count (tests)")
    parser.add_argument("--skip-model-rows", action="store_true",
                        help="write suites without verifying local GGUF rows (tests)")
    parser.add_argument("--expect-burned-present", default=",".join(BURNED_TASK_IDS),
                        help="burned IDs that must exist in the dataset (tests)")
    parser.add_argument("--force", action="store_true",
                        help="overwrite existing suite files")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    workspace: Path = args.workspace
    dataset_path = args.dataset or (workspace / "HumanEvalPlus.jsonl" / "HumanEvalPlus.jsonl")
    prompt_template_path = args.prompt_template or (
        workspace / "benchmarks" / "prompts" / "lfm-code-v1.txt"
    )
    prompt_manifest_path = args.prompt_manifest or (
        workspace / "benchmarks" / "prompts" / "lfm-code-v1.json"
    )
    baseline_report_path = args.baseline_report or (
        workspace / "reports" / "phase5" / "original-baseline-conversion.json"
    )

    records = verify_dataset(
        dataset_path, args.dataset_sha256, args.dataset_size, args.dataset_records
    )
    prompt_identity = verify_prompt_identities(
        prompt_template_path,
        prompt_manifest_path,
        args.prompt_template_sha256,
        args.prompt_manifest_sha256,
    )

    task_ids = [record["task_id"] for record in records]
    burned_present = [item for item in args.expect_burned_present.split(",") if item]
    missing = [task_id for task_id in burned_present if task_id not in set(task_ids)]
    if missing:
        raise SystemExit(f"Burned task IDs absent from dataset: {missing}")
    pilot, sealed = select_slices(task_ids, list(BURNED_TASK_IDS))
    pilot_hash = canonical_task_list_sha256(pilot)
    if pilot_hash != args.pilot_hash:
        raise SystemExit(
            "Canonical pilot list hash drifted from the frozen plan:"
            f" expected {args.pilot_hash}, computed {pilot_hash}"
        )
    print(f"Pilot slice: {len(pilot)} tasks; sealed slice: {len(sealed)} tasks")

    rows: list[dict[str, Any]] = []
    if not args.skip_model_rows:
        baseline = json.loads(baseline_report_path.read_text(encoding="utf-8"))
        rows.append(
            model_row(
                "original-bf16",
                Path(baseline["output"]["path"]),
                workspace,
                baseline["output"]["sha256"],
                "reports/phase5/original-baseline-conversion.json",
                "upstream baseline",
            )
        )
        rows.append(
            model_row(
                "merged-bf16",
                workspace / "artifacts" / "gguf" / "lfm2.5-1.2b-thinking-kodcode-bf16.gguf",
                workspace,
                EXPECTED_MERGED_SHA256,
                "reports/phase1/bf16-conversion.json",
                "fine-tuned reference",
            )
        )
        for alias in ("q8_0", "q6_k"):
            rows.append(
                model_row(
                    alias,
                    workspace / "artifacts" / "gguf" / "phase2"
                    / f"lfm2.5-1.2b-thinking-kodcode-{alias}.gguf",
                    workspace,
                    EXPECTED_QUANT_SHA256[alias],
                    "reports/phase2/quantization.json",
                    "pilot candidate",
                )
            )

    pilot_suite = build_pilot_suite(pilot, len(sealed), rows, prompt_identity)
    sealed_suite = build_sealed_suite(pilot, sealed, prompt_identity, rows)
    write_json(
        workspace / "benchmarks" / "suites" / "e3-humanevalplus-pilot-v1.json",
        pilot_suite,
        args.force,
    )
    write_json(
        workspace / "benchmarks" / "suites" / "e4-humanevalplus-sealed-v1.json",
        sealed_suite,
        args.force,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
