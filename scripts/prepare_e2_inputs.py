#!/usr/bin/env python3
"""Prepare matched E2 sample/dataset subsets without executing generated code."""

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


def verify_asset(workspace: Path, contract: dict[str, Any], label: str) -> Path:
    path = (workspace / contract["path"]).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")
    actual_hash = sha256_file(path)
    actual_size = path.stat().st_size
    if actual_hash != contract["sha256"] or actual_size != contract["size_bytes"]:
        raise ValueError(
            f"{label} identity mismatch: got {actual_size} bytes, SHA-256 {actual_hash}"
        )
    return path


def jsonl_text(records: list[dict[str, Any]]) -> str:
    return "".join(
        json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
        for record in records
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--task-id", action="append", dest="task_ids")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.output_dir.exists():
        print(f"Refusing to overwrite E2 inputs: {args.output_dir}", flush=True)
        return 3

    suite = read_json(args.suite)
    if suite.get("schema_version") != 1 or suite.get("suite_id") != "e2-humanevalplus-dev-v1":
        raise ValueError("Unsupported E2 suite identity")

    allowed_task_ids = suite["development_task_ids"]
    requested = args.task_ids or [suite["default_manual_task_id"]]
    if len(requested) != len(set(requested)):
        raise ValueError("Duplicate --task-id values are not allowed")
    unknown = [task_id for task_id in requested if task_id not in allowed_task_ids]
    if unknown:
        raise ValueError(f"Task IDs are outside the frozen development set: {unknown}")
    selected_task_ids = [task_id for task_id in allowed_task_ids if task_id in requested]

    source = suite["source"]
    generations_path = verify_asset(args.workspace, source["generations"], "generations")
    extraction_path = verify_asset(
        args.workspace, source["extraction_report"], "extraction report"
    )
    samples_path = verify_asset(args.workspace, source["samples"], "samples")
    dataset_path = verify_asset(args.workspace, suite["dataset"], "dataset")

    extraction_report = read_json(extraction_path)
    if (
        extraction_report.get("passed") is not True
        or extraction_report.get("selected_record_count")
        != source["extraction_report"]["selected_record_count"]
        or extraction_report.get("exported_sample_count")
        != source["extraction_report"]["exported_sample_count"]
        or extraction_report.get("generated_programs_executed") is not False
    ):
        raise ValueError("Extraction report does not satisfy the frozen E2 source contract")

    samples = read_jsonl(samples_path)
    if len(samples) != source["samples"]["record_count"]:
        raise ValueError("Source sample count mismatch")
    expected_sample_fields = set(source["samples"]["exact_fields"])
    if any(set(sample) != expected_sample_fields for sample in samples):
        raise ValueError("Source samples contain fields outside the frozen schema")
    sample_by_id = {sample.get("task_id"): sample for sample in samples}
    if len(sample_by_id) != len(samples) or set(sample_by_id) != set(allowed_task_ids):
        raise ValueError("Source samples do not contain exactly one record per development task")

    dataset = read_jsonl(dataset_path)
    dataset_ids = [record.get("task_id") for record in dataset]
    if (
        len(dataset) != suite["dataset"]["record_count"]
        or len(set(dataset_ids)) != suite["dataset"]["unique_task_id_count"]
    ):
        raise ValueError("Dataset record or unique-task count mismatch")
    dataset_by_id = {record.get("task_id"): record for record in dataset}
    missing = [task_id for task_id in selected_task_ids if task_id not in dataset_by_id]
    if missing:
        raise ValueError(f"Selected tasks are missing from the dataset: {missing}")

    selected_samples = [sample_by_id[task_id] for task_id in selected_task_ids]
    selected_dataset = [dataset_by_id[task_id] for task_id in selected_task_ids]
    args.output_dir.mkdir(parents=True)
    selected_samples_path = args.output_dir / "samples.jsonl"
    selected_dataset_path = args.output_dir / "HumanEvalPlus-subset.jsonl"
    atomic_write(selected_samples_path, jsonl_text(selected_samples))
    atomic_write(selected_dataset_path, jsonl_text(selected_dataset))

    report = {
        "schema_version": 1,
        "generated_at_utc": utc_now(),
        "suite_id": suite["suite_id"],
        "passed": True,
        "selected_task_ids": selected_task_ids,
        "selected_task_count": len(selected_task_ids),
        "source_generations": file_identity(generations_path),
        "source_extraction_report": file_identity(extraction_path),
        "source_samples": file_identity(samples_path),
        "source_dataset": file_identity(dataset_path),
        "prepared_samples": file_identity(selected_samples_path),
        "prepared_dataset": file_identity(selected_dataset_path),
        "generated_programs_executed": False,
        "hidden_tests_printed": False,
    }
    atomic_write(
        args.output_dir / "preparation-report.json",
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    print(
        f"E2 input preparation: PASS ({len(selected_task_ids)} task(s), no execution)",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
