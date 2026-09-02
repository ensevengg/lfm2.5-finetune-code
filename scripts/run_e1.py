#!/usr/bin/env python3
"""Run the frozen E1 generation-only smoke test through pinned llama.cpp."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from review_e1 import review as review_behavior


PLACEHOLDER = "{{BENCHMARK_PROMPT}}"
SUPPORTED_SUITE_IDS = {
    "e0-humanevalplus-smoke-v1",
    "e1-humanevalplus-smoke-v2",
}
REQUIRED_SERVER_FLAGS = (
    "--model",
    "--ctx-size",
    "--n-gpu-layers",
    "--flash-attn",
    "--cache-type-k",
    "--cache-type-v",
    "--fit",
    "--parallel",
    "--offline",
)


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


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def append_jsonl(path: Path, value: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def add_gate(gates: dict[str, dict[str, Any]], name: str, passed: bool, detail: Any) -> None:
    gates[name] = {"passed": bool(passed), "detail": detail}


def validate_assets(
    args: argparse.Namespace,
) -> tuple[dict[str, Any], list[dict[str, Any]], str, dict[str, Any]]:
    suite = read_json(args.suite)
    prompts = read_jsonl(args.prompts)
    prompt_template = args.prompt_template.read_text(encoding="utf-8")
    gates: dict[str, dict[str, Any]] = {}

    benchmark = suite.get("benchmark", {})
    generation_asset = benchmark.get("generation_asset", {})
    expected_fields = generation_asset.get(
        "allowed_fields", ["task_id", "entry_point", "prompt"]
    )
    expected_ids = benchmark.get("task_ids", [])
    expected_entry_points = {
        item.get("task_id"): item.get("entry_point")
        for item in benchmark.get("verified_tasks", [])
    }
    actual_ids = [record.get("task_id") for record in prompts]
    duplicate_ids = sorted(
        {task_id for task_id in actual_ids if actual_ids.count(task_id) > 1}
    )
    field_errors = [
        {"line": index, "actual": list(record.keys()), "expected": expected_fields}
        for index, record in enumerate(prompts, 1)
        if set(record.keys()) != set(expected_fields)
    ]
    entry_point_errors = [
        {
            "task_id": record.get("task_id"),
            "actual": record.get("entry_point"),
            "expected": expected_entry_points.get(record.get("task_id")),
        }
        for record in prompts
        if record.get("entry_point")
        != expected_entry_points.get(record.get("task_id"))
    ]
    empty_prompts = [
        record.get("task_id")
        for record in prompts
        if not isinstance(record.get("prompt"), str) or not record["prompt"]
    ]

    prompt_hash = sha256_file(args.prompts)
    template_hash = sha256_file(args.prompt_template)
    model_hash = sha256_file(args.model)
    prompt_contract = suite.get("prompt", {})
    prompt_id = prompt_contract.get("prompt_id", prompt_contract.get("version"))
    extractor_contract = suite.get("extractor")
    add_gate(
        gates,
        "suite_identity",
        suite.get("schema_version") == 1
        and suite.get("suite_id") in SUPPORTED_SUITE_IDS,
        {"schema_version": suite.get("schema_version"), "suite_id": suite.get("suite_id")},
    )
    add_gate(
        gates,
        "prompt_identity",
        isinstance(prompt_id, str)
        and bool(prompt_id)
        and prompt_contract.get("version") == prompt_id,
        {"prompt_id": prompt_id, "version": prompt_contract.get("version")},
    )
    add_gate(
        gates,
        "extractor_identity",
        extractor_contract is None
        or (
            suite.get("suite_id") == "e1-humanevalplus-smoke-v2"
            and extractor_contract.get("extractor_id") == "python-solution-v1"
            and isinstance(extractor_contract.get("manifest_sha256"), str)
            and isinstance(extractor_contract.get("implementation_sha256"), str)
        ),
        extractor_contract,
    )
    add_gate(
        gates,
        "prompt_asset_hash",
        bool(generation_asset)
        and prompt_hash == generation_asset.get("sha256")
        and args.prompts.stat().st_size == generation_asset.get("size_bytes"),
        {
            "actual_sha256": prompt_hash,
            "expected_sha256": generation_asset.get("sha256"),
            "actual_size_bytes": args.prompts.stat().st_size,
            "expected_size_bytes": generation_asset.get("size_bytes"),
        },
    )
    add_gate(
        gates,
        "prompt_record_count",
        len(prompts) == generation_asset.get("record_count") == len(expected_ids),
        {
            "actual": len(prompts),
            "asset_expected": generation_asset.get("record_count"),
            "suite_expected": len(expected_ids),
        },
    )
    add_gate(
        gates,
        "prompt_task_ids",
        actual_ids == expected_ids and not duplicate_ids,
        {"actual": actual_ids, "expected": expected_ids, "duplicates": duplicate_ids},
    )
    add_gate(gates, "prompt_fields", not field_errors, field_errors)
    add_gate(gates, "prompt_entry_points", not entry_point_errors, entry_point_errors)
    add_gate(gates, "prompt_text_nonempty", not empty_prompts, empty_prompts)
    add_gate(
        gates,
        "prompt_template",
        template_hash == prompt_contract.get("sha256")
        and prompt_template.count(PLACEHOLDER) == 1,
        {
            "actual_sha256": template_hash,
            "expected_sha256": prompt_contract.get("sha256"),
            "placeholder_count": prompt_template.count(PLACEHOLDER),
        },
    )
    add_gate(
        gates,
        "model_identity",
        model_hash == suite.get("model", {}).get("sha256")
        and args.model.stat().st_size == suite.get("model", {}).get("size_bytes"),
        {
            "actual_sha256": model_hash,
            "expected_sha256": suite.get("model", {}).get("sha256"),
            "actual_size_bytes": args.model.stat().st_size,
            "expected_size_bytes": suite.get("model", {}).get("size_bytes"),
        },
    )
    add_gate(
        gates,
        "generation_only_policy",
        suite.get("execution", {}).get("generated_code_execution_allowed") is False
        and suite.get("execution", {}).get("scoring_allowed") is False,
        suite.get("execution", {}),
    )

    report = {
        "schema_version": 1,
        "generated_at_utc": utc_now(),
        "run_id": args.run_id,
        "mode": args.mode,
        "passed": all(gate["passed"] for gate in gates.values()),
        "gates": gates,
        "summary": {
            "prompt_record_count": len(prompts),
            "prompt_integrity_case_count": len(suite.get("prompt_integrity_cases", [])),
            "prompt_asset_sha256": prompt_hash,
            "prompt_template_sha256": template_hash,
            "prompt_id": prompt_id,
            "extractor_id": (
                extractor_contract.get("extractor_id")
                if isinstance(extractor_contract, dict)
                else None
            ),
            "model_sha256": model_hash,
        },
    }
    return suite, prompts, prompt_template, report


def http_json(
    method: str,
    url: str,
    payload: dict[str, Any] | None = None,
    timeout: float = 30.0,
) -> dict[str, Any]:
    body = None
    headers: dict[str, str] = {}
    if payload is not None:
        body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(url, data=body, headers=headers, method=method)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            value = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} from {url}: {detail}") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"Expected JSON object from {url}")
    return value


def server_command(args: argparse.Namespace, suite: dict[str, Any]) -> list[str]:
    runtime = suite["runtime"]
    return [
        str(args.server_binary),
        "--model",
        str(args.model),
        "--alias",
        suite["model"]["alias"],
        "--ctx-size",
        str(runtime["context_size"]),
        "--n-gpu-layers",
        str(runtime["gpu_layers"]),
        "--flash-attn",
        "on" if runtime["flash_attention"] else "off",
        "--cache-type-k",
        runtime["cache_type_k"],
        "--cache-type-v",
        runtime["cache_type_v"],
        "--fit",
        "on" if runtime["automatic_fit"] else "off",
        "--parallel",
        str(runtime["parallel_slots"]),
        "--host",
        args.server_host,
        "--port",
        str(args.server_port),
        "--offline" if runtime["offline"] else "--no-offline",
        "--no-warmup",
    ]


def check_server_cli(args: argparse.Namespace, command: list[str]) -> dict[str, Any]:
    completed = subprocess.run(
        [str(args.server_binary), "--help"],
        capture_output=True,
        text=True,
        errors="replace",
        check=False,
        timeout=30,
    )
    help_text = completed.stdout + completed.stderr
    missing = [flag for flag in REQUIRED_SERVER_FLAGS if flag not in help_text]
    report = {
        "schema_version": 1,
        "generated_at_utc": utc_now(),
        "executable": str(args.server_binary),
        "command": command,
        "help_exit_code": completed.returncode,
        "required_flags": list(REQUIRED_SERVER_FLAGS),
        "missing_flags": missing,
        "passed": completed.returncode == 0 and not missing,
    }
    write_json(args.results_dir / "server-command.json", report)
    return report


def wait_for_server(base_url: str, process: subprocess.Popen[Any], timeout: float) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    last_error = "not attempted"
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"llama-server exited early with code {process.returncode}")
        try:
            health = http_json("GET", base_url + "/health", timeout=2.0)
            if health.get("status") == "ok":
                return health
            last_error = repr(health)
        except Exception as exc:
            last_error = str(exc)
        time.sleep(0.5)
    raise TimeoutError(f"llama-server was not healthy after {timeout}s: {last_error}")


def build_cases(
    suite: dict[str, Any],
    prompts: list[dict[str, Any]],
    prompt_template: str,
    mode: str,
    replay_task_id: str,
) -> list[dict[str, Any]]:
    integrity = [
        {
            "case_id": case["case_id"],
            "case_kind": "prompt_integrity",
            "task_id": None,
            "entry_point": None,
            "messages": case["messages"],
            "diagnostic": case["diagnostic"],
        }
        for case in suite["prompt_integrity_cases"]
    ]
    benchmark = [
        {
            "case_id": "humanevalplus-" + record["task_id"].replace("/", "-"),
            "case_kind": "humanevalplus_development",
            "task_id": record["task_id"],
            "entry_point": record["entry_point"],
            "messages": [
                {
                    "role": "user",
                    "content": prompt_template.replace(PLACEHOLDER, record["prompt"]),
                }
            ],
            "benchmark_prompt": record["prompt"],
        }
        for record in prompts
    ]
    if mode == "replay":
        selected = [case for case in benchmark if case["task_id"] == replay_task_id]
        if len(selected) != 1:
            raise ValueError(f"Replay task ID must select exactly one frozen task: {replay_task_id}")
        selected[0]["case_kind"] = "determinism_replay"
        return selected
    return integrity + benchmark


def generate_case(
    base_url: str,
    case: dict[str, Any],
    suite: dict[str, Any],
) -> dict[str, Any]:
    generation = suite["generation"]
    started = utc_now()
    applied = http_json(
        "POST", base_url + "/apply-template", {"messages": case["messages"]}, timeout=30.0
    )
    rendered_prompt = applied.get("prompt")
    if not isinstance(rendered_prompt, str):
        raise RuntimeError("/apply-template response did not contain a string prompt")

    tokenized = http_json(
        "POST",
        base_url + "/tokenize",
        {
            "content": rendered_prompt,
            "add_special": True,
            "parse_special": True,
            "with_pieces": False,
        },
        timeout=30.0,
    )
    prompt_tokens = tokenized.get("tokens")
    if not isinstance(prompt_tokens, list) or not all(
        isinstance(token, int) for token in prompt_tokens
    ):
        raise RuntimeError("/tokenize response did not contain integer token IDs")
    if len(prompt_tokens) + generation["max_output_tokens"] > suite["runtime"]["context_size"]:
        raise RuntimeError(
            "Frozen prompt plus max output exceeds the frozen context window: "
            f"{len(prompt_tokens)} + {generation['max_output_tokens']} > "
            f"{suite['runtime']['context_size']}"
        )

    completion_request = {
        "prompt": prompt_tokens,
        "n_predict": generation["max_output_tokens"],
        "temperature": generation["temperature"],
        "seed": generation["seed"],
        "stream": generation["stream"],
        "cache_prompt": False,
        "return_tokens": True,
    }
    response = http_json(
        "POST", base_url + "/completion", completion_request, timeout=900.0
    )
    completion = response.get("content")
    completion_tokens = response.get("tokens")
    if not isinstance(completion, str):
        raise RuntimeError("/completion response did not contain string content")
    if not isinstance(completion_tokens, list) or not all(
        isinstance(token, int) for token in completion_tokens
    ):
        raise RuntimeError("/completion response did not contain integer token IDs")

    return {
        "schema_version": 1,
        "run_id": suite.get("_run_id"),
        "mode": suite.get("_mode"),
        "case_id": case["case_id"],
        "case_kind": case["case_kind"],
        "task_id": case.get("task_id"),
        "entry_point": case.get("entry_point"),
        "prompt_id": suite["prompt"].get(
            "prompt_id", suite["prompt"].get("version")
        ),
        "prompt_sha256": suite["prompt"]["sha256"],
        "started_at_utc": started,
        "finished_at_utc": utc_now(),
        "status": "terminal",
        "messages": case["messages"],
        "benchmark_prompt": case.get("benchmark_prompt"),
        "templated_prompt": rendered_prompt,
        "templated_prompt_token_ids": prompt_tokens,
        "templated_prompt_token_count": len(prompt_tokens),
        "prompt_tokenization": {
            "add_special": True,
            "parse_special": True,
            "expected_bos_token_id": suite["prompt"]["bos_token_id"],
        },
        "request": completion_request,
        "raw_completion": completion,
        "completion_token_ids": completion_tokens,
        "completion_token_count": len(completion_tokens),
        "finish_reason": response.get("stop_type"),
        "prompt_truncated": bool(response.get("truncated", False)),
        "timings": response.get("timings"),
        "raw_response": response,
        "generated_program_executed": False,
        "scored": False,
    }


def compare_replay(
    reference_path: Path,
    replay_record: dict[str, Any],
    replay_task_id: str,
) -> dict[str, Any]:
    matches = [
        record
        for record in read_jsonl(reference_path)
        if record.get("task_id") == replay_task_id
        and record.get("case_kind") == "humanevalplus_development"
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Reference generations must contain one development record for {replay_task_id}"
        )
    reference = matches[0]
    fields = (
        "raw_completion",
        "completion_token_ids",
        "templated_prompt_token_ids",
        "templated_prompt_token_count",
        "completion_token_count",
        "finish_reason",
        "prompt_id",
        "prompt_sha256",
    )
    comparisons = {
        field: {
            "matched": reference.get(field) == replay_record.get(field),
            "reference": reference.get(field),
            "replay": replay_record.get(field),
        }
        for field in fields
    }
    return {
        "schema_version": 1,
        "generated_at_utc": utc_now(),
        "task_id": replay_task_id,
        "reference_run_id": reference.get("run_id"),
        "replay_run_id": replay_record.get("run_id"),
        "passed": all(item["matched"] for item in comparisons.values()),
        "comparisons": comparisons,
    }


def environment_report(
    args: argparse.Namespace,
    suite: dict[str, Any],
    health: dict[str, Any],
    props: dict[str, Any],
) -> dict[str, Any]:
    host_environment = read_json(args.host_environment) if args.host_environment else None
    return {
        "schema_version": 1,
        "generated_at_utc": utc_now(),
        "run_id": args.run_id,
        "mode": args.mode,
        "container": {
            "python": sys.version,
            "platform": platform.platform(),
            "machine": platform.machine(),
            "uid": os.getuid() if hasattr(os, "getuid") else None,
            "image_tag": args.image_tag,
            "image_id": args.image_id,
        },
        "host": host_environment,
        "runtime_expected": suite["runtime"],
        "server_health": health,
        "server_props": props,
    }


def runtime_gates(
    suite: dict[str, Any], props: dict[str, Any], args: argparse.Namespace
) -> dict[str, dict[str, Any]]:
    settings = props.get("default_generation_settings", {})
    build_info = str(props.get("build_info", ""))
    expected_revision = suite["runtime"]["revision"]
    gates: dict[str, dict[str, Any]] = {}
    add_gate(
        gates,
        "container_image",
        args.image_tag == suite["runtime"]["toolchain_image"]
        and args.image_id == suite["runtime"].get("toolchain_image_id"),
        {
            "actual_tag": args.image_tag,
            "expected_tag": suite["runtime"]["toolchain_image"],
            "actual_id": args.image_id,
            "expected_id": suite["runtime"].get("toolchain_image_id"),
        },
    )
    add_gate(
        gates,
        "runtime_revision",
        expected_revision[:8] in build_info or expected_revision in build_info,
        {"expected_revision": expected_revision, "build_info": build_info},
    )
    add_gate(
        gates,
        "context_size",
        settings.get("n_ctx") == suite["runtime"]["context_size"],
        {"actual": settings.get("n_ctx"), "expected": suite["runtime"]["context_size"]},
    )
    add_gate(
        gates,
        "parallel_slots",
        props.get("total_slots") == suite["runtime"]["parallel_slots"],
        {"actual": props.get("total_slots"), "expected": suite["runtime"]["parallel_slots"]},
    )
    return gates


def run_generation(
    args: argparse.Namespace,
    suite: dict[str, Any],
    prompts: list[dict[str, Any]],
    prompt_template: str,
    validation: dict[str, Any],
) -> int:
    output_path = args.results_dir / "generations.jsonl"
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing evidence: {output_path}")
    logs_dir = args.results_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    command = server_command(args, suite)
    cli_report = check_server_cli(args, command)
    if not cli_report["passed"]:
        raise RuntimeError(f"llama-server CLI preflight failed: {cli_report['missing_flags']}")

    suite["_run_id"] = args.run_id
    suite["_mode"] = args.mode
    cases = build_cases(suite, prompts, prompt_template, args.mode, args.replay_task_id)
    base_url = f"http://{args.server_host}:{args.server_port}"
    records: list[dict[str, Any]] = []
    runtime_checks: dict[str, dict[str, Any]] = {}
    replay_report: dict[str, Any] | None = None
    behavior_report: dict[str, Any] | None = None
    failure: str | None = None
    process: subprocess.Popen[Any] | None = None
    started_at = utc_now()

    print("Starting pinned llama.cpp server inside this container...", flush=True)
    with (logs_dir / "server.stdout.log").open("wb") as stdout_log, (
        logs_dir / "server.stderr.log"
    ).open("wb") as stderr_log:
        try:
            process = subprocess.Popen(command, stdout=stdout_log, stderr=stderr_log)
            health = wait_for_server(base_url, process, args.startup_timeout)
            props = http_json("GET", base_url + "/props", timeout=30.0)
            runtime_checks = runtime_gates(suite, props, args)
            write_json(
                args.results_dir / "environment.json",
                environment_report(args, suite, health, props),
            )
            failed_runtime = [
                name for name, gate in runtime_checks.items() if not gate["passed"]
            ]
            if failed_runtime:
                raise RuntimeError(f"Runtime identity/settings gates failed: {failed_runtime}")
            print(
                f"Server is healthy; frozen context={suite['runtime']['context_size']}, slots=1.",
                flush=True,
            )

            for index, case in enumerate(cases, 1):
                label = case.get("task_id") or case["case_id"]
                print(
                    f"[{index}/{len(cases)}] Generating {label} (no execution, no scoring)...",
                    flush=True,
                )
                record = generate_case(base_url, case, suite)
                append_jsonl(output_path, record)
                records.append(record)
                print(
                    f"[{index}/{len(cases)}] Saved terminal record: "
                    f"prompt={record['templated_prompt_token_count']} tokens, "
                    f"completion={record['completion_token_count']} tokens, "
                    f"stop={record['finish_reason']}",
                    flush=True,
                )

            if args.mode == "replay":
                if args.reference_generations is None:
                    raise ValueError("--reference-generations is required in replay mode")
                replay_report = compare_replay(
                    args.reference_generations, records[0], args.replay_task_id
                )
                write_json(args.results_dir / "replay-comparison.json", replay_report)
                print(
                    "Replay comparison: " + ("PASS" if replay_report["passed"] else "FAIL"),
                    flush=True,
                )
            else:
                behavior_report = review_behavior(suite, records)
                write_json(args.results_dir / "behavior-review.json", behavior_report)
                print(
                    "Prompt-integrity behavior: "
                    + ("PASS" if behavior_report["passed"] else "FAIL"),
                    flush=True,
                )
        except Exception as exc:
            failure = f"{type(exc).__name__}: {exc}"
            print(f"E1 run stopped: {failure}", file=sys.stderr, flush=True)
        finally:
            if process is not None and process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=10)

    expected_count = (
        1
        if args.mode == "replay"
        else len(suite["prompt_integrity_cases"]) + len(prompts)
    )
    integrity_count = sum(record["case_kind"] == "prompt_integrity" for record in records)
    benchmark_count = sum(
        record["case_kind"] == "humanevalplus_development" for record in records
    )
    gates: dict[str, dict[str, Any]] = {
        "asset_validation": {
            "passed": validation["passed"],
            "detail": str(args.results_dir / "validation.json"),
        },
        "server_cli": {
            "passed": cli_report["passed"],
            "detail": str(args.results_dir / "server-command.json"),
        },
        **runtime_checks,
    }
    add_gate(
        gates,
        "request_count",
        len(records) == expected_count,
        {"actual": len(records), "expected": expected_count},
    )
    add_gate(
        gates,
        "terminal_records",
        all(record.get("status") == "terminal" for record in records)
        and len(records) == expected_count,
        len(records),
    )
    add_gate(
        gates,
        "silent_truncation",
        not any(record.get("prompt_truncated") for record in records),
        [record["case_id"] for record in records if record.get("prompt_truncated")],
    )
    add_gate(
        gates,
        "no_execution_or_scoring",
        all(
            not record.get("generated_program_executed") and not record.get("scored")
            for record in records
        ),
        {"executed": 0, "scored": 0},
    )
    if args.mode == "primary":
        add_gate(
            gates,
            "prompt_integrity_count",
            integrity_count == 4,
            {"actual": integrity_count, "expected": 4},
        )
        add_gate(
            gates,
            "humanevalplus_count",
            benchmark_count == 3,
            {"actual": benchmark_count, "expected": 3},
        )
        add_gate(
            gates,
            "prompt_integrity_behavior",
            bool(behavior_report and behavior_report["passed"]),
            str(args.results_dir / "behavior-review.json"),
        )
    else:
        add_gate(
            gates,
            "determinism_replay",
            bool(replay_report and replay_report["passed"]),
            replay_report,
        )
    add_gate(gates, "no_infrastructure_failure", failure is None, failure)

    manifest = {
        "schema_version": 1,
        "run_id": args.run_id,
        "mode": args.mode,
        "suite_id": suite["suite_id"],
        "started_at_utc": started_at,
        "finished_at_utc": utc_now(),
        "passed": all(gate["passed"] for gate in gates.values()),
        "gates": gates,
        "record_count": len(records),
        "expected_record_count": expected_count,
        "failure": failure,
        "generated_programs_executed": 0,
        "scores_computed": 0,
        "artifacts": [
            "validation.json",
            "run-manifest.json",
            "server-command.json",
            "environment.json",
            "generations.jsonl",
            "logs/server.stdout.log",
            "logs/server.stderr.log",
        ]
        + (["behavior-review.json"] if args.mode == "primary" else [])
        + (["replay-comparison.json"] if args.mode == "replay" else []),
    }
    write_json(args.results_dir / "run-manifest.json", manifest)
    print("E1 run gate: " + ("PASS" if manifest["passed"] else "FAIL"), flush=True)
    return 0 if manifest["passed"] else 4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--prompt-template", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--mode", choices=("primary", "replay"), default="primary")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument(
        "--server-binary", type=Path, default=Path("/opt/llama/bin/llama-server")
    )
    parser.add_argument("--server-host", default="127.0.0.1")
    parser.add_argument("--server-port", type=int, default=8080)
    parser.add_argument("--startup-timeout", type=float, default=180.0)
    parser.add_argument("--reference-generations", type=Path)
    parser.add_argument("--replay-task-id", default="HumanEval/0")
    parser.add_argument("--host-environment", type=Path)
    parser.add_argument("--image-tag")
    parser.add_argument("--image-id")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.results_dir.mkdir(parents=True, exist_ok=True)
    try:
        suite, prompts, prompt_template, report = validate_assets(args)
    except Exception as exc:
        report = {
            "schema_version": 1,
            "generated_at_utc": utc_now(),
            "run_id": args.run_id,
            "mode": args.mode,
            "passed": False,
            "error": f"{type(exc).__name__}: {exc}",
        }
        suite = {}
        prompts = []
        prompt_template = ""
    write_json(args.results_dir / "validation.json", report)
    print(f"E1 asset validation: {'PASS' if report.get('passed') else 'FAIL'}", flush=True)
    if not report.get("passed"):
        return 2
    if args.validate_only:
        return 0
    try:
        return run_generation(args, suite, prompts, prompt_template, report)
    except Exception as exc:
        print(
            f"E1 harness failure: {type(exc).__name__}: {exc}",
            file=sys.stderr,
            flush=True,
        )
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
