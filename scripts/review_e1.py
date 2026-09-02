#!/usr/bin/env python3
"""Apply frozen, non-executing behavior checks to E1 generation evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PYTHON_FENCE = re.compile(r"\A```python\r?\n.+\r?\n```\s*\Z", re.DOTALL)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


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


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def check_case(record: dict[str, Any], expectation: dict[str, Any]) -> dict[str, Any]:
    completion = record.get("raw_completion")
    if not isinstance(completion, str):
        completion = ""
    folded = completion.casefold()
    checks: dict[str, dict[str, Any]] = {}

    if "allowed_finish_reasons" in expectation:
        allowed = expectation["allowed_finish_reasons"]
        checks["finish_reason"] = {
            "passed": record.get("finish_reason") in allowed,
            "actual": record.get("finish_reason"),
            "expected": allowed,
        }
    if "max_completion_tokens" in expectation:
        maximum = expectation["max_completion_tokens"]
        actual = record.get("completion_token_count")
        checks["max_completion_tokens"] = {
            "passed": isinstance(actual, int) and actual <= maximum,
            "actual": actual,
            "expected_maximum": maximum,
        }
    if "exact_completion" in expectation:
        expected = expectation["exact_completion"]
        checks["exact_completion"] = {
            "passed": completion == expected,
            "actual": completion,
            "expected": expected,
        }
    forbidden = expectation.get("forbidden_substrings_case_insensitive", [])
    if forbidden:
        found = [item for item in forbidden if item.casefold() in folded]
        checks["forbidden_substrings"] = {
            "passed": not found,
            "found": found,
            "forbidden": forbidden,
        }
    required = expectation.get("required_substrings_case_insensitive", [])
    if required:
        missing = [item for item in required if item.casefold() not in folded]
        checks["required_substrings"] = {
            "passed": not missing,
            "missing": missing,
            "required": required,
        }
    if expectation.get("single_python_fence"):
        checks["single_python_fence"] = {
            "passed": bool(PYTHON_FENCE.fullmatch(completion)),
            "expected": "exactly one complete ```python fenced block and no outside text",
        }

    return {
        "case_id": record.get("case_id"),
        "passed": bool(checks) and all(check["passed"] for check in checks.values()),
        "checks": checks,
        "completion_sha256": hashlib.sha256(completion.encode("utf-8")).hexdigest(),
        "completion_preview": completion[:500],
    }


def review(suite: dict[str, Any], records: list[dict[str, Any]]) -> dict[str, Any]:
    expected_cases = {
        case["case_id"]: case.get("behavior_expectation", {})
        for case in suite.get("prompt_integrity_cases", [])
    }
    actual_cases = {
        record.get("case_id"): record
        for record in records
        if record.get("case_kind") == "prompt_integrity"
    }
    duplicate_ids = sorted(
        {
            record.get("case_id")
            for record in records
            if record.get("case_kind") == "prompt_integrity"
            and sum(
                other.get("case_id") == record.get("case_id")
                for other in records
                if other.get("case_kind") == "prompt_integrity"
            )
            > 1
        }
    )
    case_reports = []
    for case_id, expectation in expected_cases.items():
        if case_id not in actual_cases:
            case_reports.append(
                {
                    "case_id": case_id,
                    "passed": False,
                    "checks": {"record_present": {"passed": False}},
                }
            )
            continue
        case_reports.append(check_case(actual_cases[case_id], expectation))

    unexpected_ids = sorted(set(actual_cases) - set(expected_cases))
    expected_bos = suite.get("prompt", {}).get("bos_token_id")
    prompt_encoding_failures = []
    for record in records:
        token_ids = record.get("templated_prompt_token_ids")
        first_token = token_ids[0] if isinstance(token_ids, list) and token_ids else None
        bos_count = token_ids.count(expected_bos) if isinstance(token_ids, list) else 0
        if first_token != expected_bos or bos_count != 1:
            prompt_encoding_failures.append(
                {
                    "case_id": record.get("case_id"),
                    "actual_first_token_id": first_token,
                    "actual_bos_count": bos_count,
                }
            )
    prompt_encoding = {
        "passed": isinstance(expected_bos, int)
        and bool(records)
        and not prompt_encoding_failures,
        "expected_bos_token_id": expected_bos,
        "expected_bos_count_per_prompt": 1,
        "failures": prompt_encoding_failures,
    }
    passed = (
        len(case_reports) == len(expected_cases)
        and not duplicate_ids
        and not unexpected_ids
        and all(case["passed"] for case in case_reports)
        and prompt_encoding["passed"]
    )
    return {
        "schema_version": 1,
        "generated_at_utc": utc_now(),
        "suite_id": suite.get("suite_id"),
        "passed": passed,
        "expected_case_count": len(expected_cases),
        "actual_case_count": len(actual_cases),
        "failed_case_count": sum(not case["passed"] for case in case_reports),
        "duplicate_case_ids": duplicate_ids,
        "unexpected_case_ids": unexpected_ids,
        "prompt_encoding": prompt_encoding,
        "cases": case_reports,
        "generated_programs_executed": 0,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--generations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        report = review(read_json(args.suite), read_jsonl(args.generations))
    except Exception as exc:
        report = {
            "schema_version": 1,
            "generated_at_utc": utc_now(),
            "passed": False,
            "error": f"{type(exc).__name__}: {exc}",
        }
    write_json(args.output, report)
    print(
        "E1 behavior review: " + ("PASS" if report.get("passed") else "FAIL"),
        flush=True,
    )
    return 0 if report.get("passed") else 4


if __name__ == "__main__":
    raise SystemExit(main())
