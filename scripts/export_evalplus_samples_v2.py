#!/usr/bin/env python3
"""Export EvalPlus samples with per-task failure tolerance (E3 policy).

Thin variant of the frozen v1 extractor: it reuses v1's extraction logic and
report shapes byte-for-byte and changes only the blast radius of a failure.
The frozen v1 contract is all-or-nothing per batch; the frozen E3 suite
(e3-humanevalplus-pilot-v1) requires that one unextractable output at 33-128
task scale records a per-task failure and scores zero, without voiding the
row. Nothing is repaired, no generated program is executed, and raw outputs
are preserved unchanged.

Fatal (batch-level) conditions remain fatal: duplicate task IDs, a missing
prompt identity, or no benchmark records at all — identity ambiguity can
never be scored.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from export_evalplus_samples import (  # noqa: E402
    DuplicateTaskId,
    atomic_write,
    extraction_failure_case,
    extract_python,
    read_json,
    read_jsonl,
    sha256_bytes,
    sha256_file,
    utc_now,
    write_json,
)


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
        if duplicate_task_ids:
            raise DuplicateTaskId(
                "Duplicate task IDs make identity ambiguous; the batch is fatal:"
                f" {duplicate_task_ids}"
            )

        samples: list[dict[str, str]] = []
        cases: list[dict[str, Any]] = []
        for record in selected:
            try:
                sample, case = extract_python(record, prompt_manifest)
            except Exception as exc:
                cases.append(extraction_failure_case(record, exc))
            else:
                samples.append(sample)
                cases.append(case)

        failed_extraction_count = sum(
            case["status"] == "extraction_failure" for case in cases
        )
        failed_task_ids = sorted(
            case["task_id"] for case in cases if case["status"] == "extraction_failure"
        )
        # Process correctness, not model quality: extraction failures are E3
        # evidence, so the batch passes when identity was never ambiguous and
        # every selected record produced a terminal case.
        passed = len(cases) == len(selected)
        extraction_yield = (
            len(samples) / len(selected) if selected else 0.0
        )
        report = {
            "schema_version": 2,
            "generated_at_utc": utc_now(),
            "started_at_utc": started,
            "passed": passed,
            "extractor_id": "python-solution-v2",
            "policy": "per_task_failure_records_score_zero",
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
            "failed_task_ids": failed_task_ids,
            "extraction_yield": round(extraction_yield, 6),
            "exported_sample_count": 0,
            "samples_output": None,
            "cases": cases,
            "generated_programs_executed": False,
        }
        if passed and samples:
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
            "schema_version": 2,
            "generated_at_utc": utc_now(),
            "started_at_utc": started,
            "passed": False,
            "extractor_id": "python-solution-v2",
            "error": f"{type(exc).__name__}: {exc}",
            "generated_programs_executed": False,
        }
    write_json(args.report_output, report)
    print(
        "EvalPlus extraction v2: " + ("PASS" if report["passed"] else "FAIL")
        + f" (yield {report.get('extraction_yield', 0.0):.3f})",
        flush=True,
    )
    return 0 if report["passed"] else 4


if __name__ == "__main__":
    raise SystemExit(main())
