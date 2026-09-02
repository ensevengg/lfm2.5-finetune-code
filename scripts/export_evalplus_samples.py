#!/usr/bin/env python3
"""Export scorer-ready EvalPlus samples without executing generated Python."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PYTHON_FENCE = re.compile(
    r"\A```python[ \t]*\r?\n(?P<solution>[\s\S]*?)\r?\n```[ \t]*\Z"
)


class DuplicateTaskId(ValueError):
    """Raised when one batch contains more than one record for a benchmark task."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


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


def write_json(path: Path, value: Any) -> None:
    atomic_write(
        path,
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )


def top_level_functions(tree: ast.Module) -> set[str]:
    return {
        node.name
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }


def extract_python(
    record: dict[str, Any], prompt_manifest: dict[str, Any]
) -> tuple[dict[str, str], dict[str, Any]]:
    task_id = record.get("task_id")
    entry_point = record.get("entry_point")
    raw = record.get("raw_completion")
    if not isinstance(task_id, str) or not task_id:
        raise ValueError("missing task_id")
    if not isinstance(entry_point, str) or not entry_point:
        raise ValueError(f"{task_id}: missing entry_point")
    if record.get("prompt_id") != prompt_manifest.get("prompt_id"):
        raise ValueError(f"{task_id}: prompt_id mismatch")
    if record.get("prompt_sha256") != prompt_manifest.get("sha256"):
        raise ValueError(f"{task_id}: prompt_sha256 mismatch")
    if not isinstance(raw, str) or not raw:
        raise ValueError(f"{task_id}: empty or non-string raw_completion")

    candidate = raw
    strategies: list[str] = []
    warnings: list[str] = []
    if "<think>" in candidate or "</think>" in candidate:
        if not candidate.startswith("<think>"):
            raise ValueError(f"{task_id}: reasoning tags must form one leading prefix")
        if candidate.count("<think>") != 1 or candidate.count("</think>") != 1:
            raise ValueError(f"{task_id}: expected exactly one closed reasoning prefix")
        closing = candidate.find("</think>")
        candidate = candidate[closing + len("</think>") :].lstrip()
        strategies.append("reasoning_prefix")
        warnings.append("reasoning_tags_present")

    fence = PYTHON_FENCE.fullmatch(candidate)
    if fence:
        candidate = fence.group("solution")
        strategies.append("python_fence")
        warnings.append("markdown_fence_present")
    elif "```" in candidate:
        raise ValueError(f"{task_id}: Markdown fence is incomplete, multiple, or has outside text")
    else:
        strategies.append("plain_python")

    tree = ast.parse(candidate, filename=f"<{task_id}-derived-solution>", mode="exec")
    functions = top_level_functions(tree)
    if entry_point not in functions:
        raise ValueError(f"{task_id}: expected top-level function {entry_point!r} not found")

    raw_hash = sha256_bytes(raw.encode("utf-8"))
    solution_hash = sha256_bytes(candidate.encode("utf-8"))
    case = {
        "task_id": task_id,
        "entry_point": entry_point,
        "status": "extracted",
        "strategy": "_then_".join(strategies),
        "raw_completion_sha256": raw_hash,
        "solution_sha256": solution_hash,
        "raw_completion_size_bytes": len(raw.encode("utf-8")),
        "solution_size_bytes": len(candidate.encode("utf-8")),
        "syntax_valid": True,
        "entry_point_found": True,
        "warnings": warnings,
        "generated_program_executed": False,
    }
    return {"task_id": task_id, "solution": candidate}, case


def extraction_failure_case(
    record: dict[str, Any], exc: Exception
) -> dict[str, Any]:
    raw = record.get("raw_completion")
    raw_bytes = raw.encode("utf-8") if isinstance(raw, str) else None
    return {
        "task_id": record.get("task_id"),
        "entry_point": record.get("entry_point"),
        "status": "extraction_failure",
        "error_type": type(exc).__name__,
        "error_message": str(exc),
        "raw_completion_sha256": (
            sha256_bytes(raw_bytes) if raw_bytes is not None else None
        ),
        "raw_completion_size_bytes": len(raw_bytes) if raw_bytes is not None else None,
        "generated_program_executed": False,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generations", type=Path, required=True)
    parser.add_argument("--prompt-manifest", type=Path, required=True)
    parser.add_argument("--samples-output", type=Path, required=True)
    parser.add_argument("--report-output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.samples_output.exists() or args.report_output.exists():
        print("Refusing to overwrite existing extraction evidence.", flush=True)
        return 3
    started = utc_now()
    try:
        prompt_manifest = read_json(args.prompt_manifest)
        records = read_jsonl(args.generations)
        selected = [
            record
            for record in records
            if isinstance(record.get("task_id"), str)
            and str(record.get("case_kind", "")).startswith("humanevalplus")
        ]
        if not selected:
            raise ValueError("No HumanEval+ benchmark records were found")
        task_id_counts = Counter(record["task_id"] for record in selected)
        duplicate_task_ids = sorted(
            task_id for task_id, count in task_id_counts.items() if count > 1
        )
        samples: list[dict[str, str]] = []
        cases: list[dict[str, Any]] = []
        for record in selected:
            try:
                if record["task_id"] in duplicate_task_ids:
                    raise DuplicateTaskId(
                        f"{record['task_id']}: duplicate task_id in selected records"
                    )
                sample, case = extract_python(record, prompt_manifest)
            except Exception as exc:
                cases.append(extraction_failure_case(record, exc))
            else:
                samples.append(sample)
                cases.append(case)

        failed_extraction_count = sum(
            case["status"] == "extraction_failure" for case in cases
        )
        report = {
            "schema_version": 1,
            "generated_at_utc": utc_now(),
            "started_at_utc": started,
            "passed": failed_extraction_count == 0,
            "prompt_id": prompt_manifest["prompt_id"],
            "prompt_sha256": prompt_manifest["sha256"],
            "source_generations": {
                "path": str(args.generations),
                "sha256": sha256_file(args.generations),
                "size_bytes": args.generations.stat().st_size,
                "record_count": len(records),
            },
            "selected_record_count": len(selected),
            "ignored_record_count": len(records) - len(selected),
            "duplicate_task_ids": duplicate_task_ids,
            "successful_extraction_count": len(samples),
            "failed_extraction_count": failed_extraction_count,
            "exported_sample_count": 0,
            "samples_output": None,
            "cases": cases,
            "generated_programs_executed": False,
        }
        if report["passed"]:
            samples_text = "".join(
                json.dumps(sample, ensure_ascii=False, separators=(",", ":")) + "\n"
                for sample in samples
            )
            atomic_write(args.samples_output, samples_text)
            report["exported_sample_count"] = len(samples)
            report["samples_output"] = {
                "path": str(args.samples_output),
                "sha256": sha256_bytes(samples_text.encode("utf-8")),
                "size_bytes": len(samples_text.encode("utf-8")),
            }
    except Exception as exc:
        report = {
            "schema_version": 1,
            "generated_at_utc": utc_now(),
            "started_at_utc": started,
            "passed": False,
            "error": f"{type(exc).__name__}: {exc}",
            "generated_programs_executed": False,
        }
    write_json(args.report_output, report)
    print(
        "EvalPlus extraction: " + ("PASS" if report["passed"] else "FAIL"),
        flush=True,
    )
    return 0 if report["passed"] else 4


if __name__ == "__main__":
    raise SystemExit(main())
