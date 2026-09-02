#!/usr/bin/env python3
"""Probe the restricted E2 container without executing generated programs."""

from __future__ import annotations

import importlib.metadata
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path


def read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def main() -> int:
    samples_path = Path("/inputs/samples.jsonl")
    dataset_path = Path("/inputs/HumanEvalPlus-subset.jsonl")
    output_path = Path("/results/container-validation.json")
    samples = read_jsonl(samples_path)
    dataset = read_jsonl(dataset_path)
    sample_ids = [record.get("task_id") for record in samples]
    dataset_ids = [record.get("task_id") for record in dataset]

    root_write_blocked = False
    root_probe = Path("/e2-root-write-probe")
    try:
        root_probe.write_text("probe", encoding="utf-8")
    except OSError:
        root_write_blocked = True
    else:
        root_probe.unlink(missing_ok=True)

    route_text = Path("/proc/net/route").read_text(encoding="utf-8")
    non_loopback_routes = [
        line for line in route_text.splitlines()[1:] if line and not line.startswith("lo\t")
    ]
    checks = {
        "non_root_uid": os.getuid() == 65534,
        "root_write_blocked": root_write_blocked,
        "sample_and_dataset_task_ids_match": sample_ids == dataset_ids,
        "selected_task_count_nonzero": len(sample_ids) > 0,
        "no_duplicate_sample_task_ids": len(sample_ids) == len(set(sample_ids)),
        "no_non_loopback_network_routes": not non_loopback_routes,
    }
    report = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "passed": all(checks.values()),
        "checks": checks,
        "selected_task_ids": sample_ids,
        "python_version": sys.version.split()[0],
        "evalplus_version": importlib.metadata.version("evalplus"),
        "uid": os.getuid(),
        "generated_programs_executed": False,
        "hidden_tests_printed": False,
    }
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        "Restricted E2 container validation: " + ("PASS" if report["passed"] else "FAIL"),
        flush=True,
    )
    return 0 if report["passed"] else 4


if __name__ == "__main__":
    raise SystemExit(main())
