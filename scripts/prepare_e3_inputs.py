#!/usr/bin/env python3
"""Prepare matched E3 pilot sample/dataset subsets without executing code."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected one JSON object in {path}")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            raise ValueError(f"Blank JSONL line at {path}:{line_number}")
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"Expected a JSON object at {path}:{line_number}")
        records.append(value)
    return records


def atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(content, encoding="utf-8", newline="\n")
    os.replace(temporary, path)


def file_identity(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def jsonl_text(records: list[dict[str, Any]]) -> str:
    return "".join(
        json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
        for record in records
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--extraction-report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.output_dir.exists():
        print(f"Refusing to overwrite E3 inputs: {args.output_dir}", flush=True)
        return 3

    suite = read_json(args.suite)
    suite_id = suite.get("suite_id", "")
    if suite_id.startswith("e3-"):
        allowed_task_ids = suite["pilot_task_ids"]
    elif suite_id.startswith("e4-"):
        allowed_task_ids = suite["sealed_task_ids"]
    else:
        raise ValueError(f"Unsupported generation suite identity: {suite_id}")
    if suite.get("schema_version") != 1:
        raise ValueError("Unsupported generation suite identity")

    dataset_path = args.workspace / suite["dataset"]["path"]
    actual_hash = sha256_file(dataset_path)
    actual_size = dataset_path.stat().st_size
    if (
        actual_hash != suite["dataset"]["sha256"]
        or actual_size != suite["dataset"]["size_bytes"]
    ):
        raise ValueError("Dataset identity mismatch against the frozen E3 suite")

    extraction_report = read_json(args.extraction_report)
    if (
        extraction_report.get("schema_version") != 2
        or extraction_report.get("extractor_id") != "python-solution-v2"
        or extraction_report.get("passed") is not True
        or extraction_report.get("generated_programs_executed") is not False
    ):
        raise ValueError("Extraction report does not satisfy the E3 extraction contract")
    if extraction_report.get("prompt_sha256") != suite["prompt"]["template_sha256"]:
        raise ValueError("Extraction report prompt identity does not match the suite")

    samples = read_jsonl(args.samples)
    if not samples:
        raise ValueError("Samples file is empty")
    if any(set(sample) != {"task_id", "solution"} for sample in samples):
        raise ValueError("Samples contain fields outside the frozen schema")
    sample_ids = [sample.get("task_id") for sample in samples]
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("Duplicate task IDs in samples file")
    outside = [task_id for task_id in sample_ids if task_id not in allowed_task_ids]
    if outside:
        raise ValueError(f"Sample task IDs outside the frozen suite slice: {outside}")

    dataset = read_jsonl(dataset_path)
    dataset_by_id = {record.get("task_id"): record for record in dataset}
    if len(dataset) != suite["dataset"]["record_count"] or len(dataset_by_id) != len(
        dataset
    ):
        raise ValueError("Dataset record or unique-task count mismatch")
    missing = [task_id for task_id in sample_ids if task_id not in dataset_by_id]
    if missing:
        raise ValueError(f"Sampled tasks missing from the dataset: {missing}")
    # Extraction failures score zero by absence; the subset carries exactly the
    # tasks that produced an extractable solution.
    selected_dataset = [dataset_by_id[task_id] for task_id in sample_ids]

    args.output_dir.mkdir(parents=True)
    samples_out = args.output_dir / "samples.jsonl"
    dataset_out = args.output_dir / "HumanEvalPlus-subset.jsonl"
    atomic_write(samples_out, jsonl_text(samples))
    atomic_write(dataset_out, jsonl_text(selected_dataset))

    report = {
        "schema_version": 1,
        "generated_at_utc": utc_now(),
        "suite_id": suite["suite_id"],
        "passed": True,
        "sampled_task_ids": sample_ids,
        "sampled_task_count": len(sample_ids),
        "suite_task_count": len(allowed_task_ids),
        "extraction_failures_scored_zero": sorted(
            set(allowed_task_ids) - set(sample_ids)
        ),
        "source_samples": file_identity(args.samples),
        "source_extraction_report": file_identity(args.extraction_report),
        "source_dataset": file_identity(dataset_path),
        "prepared_samples": file_identity(samples_out),
        "prepared_dataset": file_identity(dataset_out),
        "generated_programs_executed": False,
        "hidden_tests_printed": False,
    }
    atomic_write(
        args.output_dir / "preparation-report.json",
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    print(
        "E3/E4 input preparation: PASS"
        f" ({len(sample_ids)} sample(s),"
        f" {len(allowed_task_ids) - len(sample_ids)}"
        " extraction failure(s) score zero)",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
