#!/usr/bin/env python3
"""E3 pilot generation runner: one model row, the frozen pilot slice.

Runs inside the pinned llama.cpp toolchain container (GPU, no network,
read-only root). One llama-server process serves the whole row; the frozen
retry policy from suite e3-humanevalplus-pilot-v1 is applied through
e3_runner_lib.decide_retry. No generated program is ever executed here.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from e3_runner_lib import (
    ABORT_ROW,
    RETRY_FRESH_SERVER,
    build_prompt,
    classify_finish_reason,
    decide_retry,
    ensure_single_bos,
    generation_record,
    row_summary,
)

REQUIRED_SERVER_FLAGS = (
    "--ctx-size", "--n-gpu-layers", "--flash-attn", "--cache-type-k",
    "--cache-type-v", "--fit", "--parallel", "--offline",
)
EXIT_OK = 0
EXIT_ROW_ABORTED = 3
EXIT_GATES_RED = 4


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def http_json(method: str, url: str, payload: dict[str, Any] | None = None,
              timeout: float = 30.0) -> Any:
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    request = urllib.request.Request(
        url, data=data, method=method,
        headers={"Content-Type": "application/json"} if data else {},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


class InfrastructureFailure(Exception):
    def __init__(self, failure_class: str, detail: str) -> None:
        super().__init__(detail)
        self.failure_class = failure_class
        self.detail = detail


class ContractViolation(Exception):
    """Non-retryable harness/contract failure: aborts the row."""


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            raise ContractViolation(f"Blank JSONL line at {path}:{number}")
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ContractViolation(f"Expected an object at {path}:{number}")
        records.append(value)
    return records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--prompt-template", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--model-alias", required=True)
    parser.add_argument("--model-sha256", required=True,
                        help="artifact hash the host wrapper verified pre-launch")
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--server-binary", type=Path,
                        default=Path("/opt/llama/bin/llama-server"))
    parser.add_argument("--server-host", default="127.0.0.1")
    parser.add_argument("--server-port", type=int, default=8080)
    parser.add_argument("--health-timeout", type=float, default=180.0)
    return parser.parse_args()


def build_server_command(args: argparse.Namespace, suite: dict[str, Any]) -> list[str]:
    runtime = suite["runtime"]
    return [
        str(args.server_binary),
        "--model", str(args.model),
        "--alias", args.model_alias,
        "--ctx-size", str(runtime["context_tokens"]),
        "--n-gpu-layers", str(runtime["gpu_layers"]),
        "--flash-attn", "on" if runtime["flash_attention"] else "off",
        "--cache-type-k", runtime["cache_type_k"],
        "--cache-type-v", runtime["cache_type_v"],
        "--fit", runtime["fit"],
        "--parallel", str(runtime["parallel_slots"]),
        "--host", args.server_host,
        "--port", str(args.server_port),
        "--offline",
        "--no-warmup",
    ]


def check_server_cli(command: list[str], binary: Path) -> None:
    completed = subprocess.run(
        [str(binary), "--help"], capture_output=True, text=True,
        errors="replace", check=False, timeout=30,
    )
    help_text = completed.stdout + completed.stderr
    missing = [flag for flag in REQUIRED_SERVER_FLAGS if flag not in help_text]
    if completed.returncode != 0 or missing:
        raise ContractViolation(
            f"Server CLI preflight failed; missing flags: {missing}"
        )


class ServerHandle:
    def __init__(self, command: list[str], log_path: Path,
                 base_url: str, health_timeout: float) -> None:
        self.command = command
        self.log_path = log_path
        self.base_url = base_url
        self.health_timeout = health_timeout
        self.process: subprocess.Popen[Any] | None = None
        self.restarts = 0

    def start(self) -> None:
        self.log_handle = self.log_path.open("a", encoding="utf-8")
        self.process = subprocess.Popen(
            self.command, stdout=self.log_handle, stderr=subprocess.STDOUT, text=True,
        )
        deadline = time.monotonic() + self.health_timeout
        last_error = "not attempted"
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                raise InfrastructureFailure(
                    "server_death",
                    f"llama-server exited early with code {self.process.returncode}",
                )
            try:
                health = http_json("GET", self.base_url + "/health", timeout=2.0)
                if health.get("status") == "ok":
                    self.log_handle.flush()
                    return
                last_error = repr(health)
            except (OSError, urllib.error.URLError, json.JSONDecodeError) as exc:
                last_error = str(exc)
                if isinstance(exc, urllib.error.HTTPError) and exc.code < 500:
                    raise ContractViolation(f"Health check returned {exc.code}")
            time.sleep(0.5)
        raise InfrastructureFailure(
            "health_check_timeout",
            f"llama-server was not healthy after {self.health_timeout}s: {last_error}",
        )

    def stop(self) -> None:
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=15)
        if hasattr(self, "log_handle") and not self.log_handle.closed:
            self.log_handle.close()
        self.process = None


def generate_one(
    server: ServerHandle,
    *,
    messages: list[dict[str, str]],
    generation: dict[str, Any],
) -> dict[str, Any]:
    """One generation request; raises InfrastructureFailure/ContractViolation."""
    try:
        rendered = http_json(
            "POST", server.base_url + "/apply-template",
            {"messages": messages}, timeout=30.0,
        )
        if not isinstance(rendered, dict) or not isinstance(rendered.get("prompt"), str):
            raise ContractViolation("/apply-template did not return a prompt string")
        tokenized = http_json(
            "POST", server.base_url + "/tokenize",
            {"content": rendered["prompt"], "add_special": True,
             "parse_special": True, "with_pieces": False},
            timeout=30.0,
        )
        prompt_ids = tokenized.get("tokens")
        if not isinstance(prompt_ids, list):
            raise ContractViolation("/tokenize did not return token IDs")
        prompt_ids = [int(token_id) for token_id in prompt_ids]
        if not ensure_single_bos(prompt_ids):
            raise ContractViolation(
                "Templated prompt must begin with exactly one BOS token"
            )
        completion = http_json(
            "POST", server.base_url + "/completion",
            {
                "prompt": prompt_ids,
                "n_predict": generation["max_output_tokens"],
                "temperature": generation["temperature"],
                "seed": generation["seed"],
                "cache_prompt": False,
                "return_tokens": True,
            },
            timeout=900.0,
        )
    except urllib.error.HTTPError as exc:
        if exc.code >= 500:
            raise InfrastructureFailure("http_5xx", f"/completion HTTP {exc.code}")
        raise ContractViolation(f"Gateway rejected the request: HTTP {exc.code}")
    except (urllib.error.URLError, OSError, json.JSONDecodeError) as exc:
        raise InfrastructureFailure("connection_error", str(exc))

    content = completion.get("content")
    tokens = completion.get("tokens")
    if not isinstance(content, str):
        raise ContractViolation("/completion did not return string content")
    if not isinstance(tokens, list):
        raise ContractViolation("/completion did not return token IDs")
    stop_type = completion.get("stop_type")
    if stop_type not in ("eos", "length"):
        if completion.get("stopped_eos"):
            stop_type = "eos"
        elif completion.get("stopped_word") or completion.get("stopped_glu"):
            stop_type = "eos"
        else:
            stop_type = "length"
    return {
        "templated_prompt": rendered["prompt"],
        "templated_prompt_token_ids": prompt_ids,
        "raw_completion": content,
        "completion_token_ids": [int(token_id) for token_id in tokens],
        "finish_reason": stop_type,
        "timings": completion.get("timings") or {},
    }


def main() -> int:
    args = parse_args()
    args.results_dir.mkdir(parents=True, exist_ok=True)
    suite = json.loads(args.suite.read_text(encoding="utf-8"))
    suite_id: str = suite["suite_id"]
    if suite_id.startswith("e3-"):
        case_kind = "humanevalplus_pilot"
        mode_name = "pilot-generation"
        task_ids = suite["pilot_task_ids"]
    elif suite_id.startswith("e4-"):
        case_kind = "humanevalplus_sealed"
        mode_name = "sealed-generation"
        task_ids = suite["sealed_task_ids"]
    else:
        raise ContractViolation(f"Suite is not an E3/E4 suite: {suite_id}")

    row = next(
        (candidate for candidate in suite["model_rows"]
         if candidate["alias"] == args.model_alias),
        None,
    )
    if row is None:
        raise ContractViolation(f"Alias is not in the suite: {args.model_alias}")
    if args.model_sha256.lower() != row["sha256"].lower():
        # Frozen retry policy: identity mismatch aborts before the first request.
        raise ContractViolation("Model artifact hash does not match the suite row")

    dataset = {record["task_id"]: record for record in read_jsonl(args.dataset)}
    missing = [t for t in task_ids if t not in dataset]
    if missing:
        raise ContractViolation(f"Pilot task IDs absent from dataset: {missing}")
    template_text = args.prompt_template.read_text(encoding="utf-8")
    generation = dict(suite["generation"])
    generation["max_output_tokens"] = suite["runtime"]["max_output_tokens"]

    server_command = build_server_command(args, suite)
    check_server_cli(server_command, args.server_binary)
    (args.results_dir / "server-command.json").write_text(
        json.dumps({"schema_version": 1, "command": server_command,
                    "row_alias": args.model_alias}, indent=2) + "\n",
        encoding="utf-8",
    )

    base_url = f"http://{args.server_host}:{args.server_port}"
    server = ServerHandle(
        server_command, args.results_dir / "server.log", base_url, args.health_timeout
    )
    generations_path = args.results_dir / "generations.jsonl"

    server.start()
    record_count = 0
    aborted_reason: str | None = None
    try:
        with generations_path.open("w", encoding="utf-8", newline="\n") as evidence:
            for task_id in task_ids:
                messages = [{"role": "user", "content": build_prompt(
                    template_text, dataset[task_id]["prompt"])}]
                attempt = 1
                while True:
                    started = utc_now()
                    started_clock = time.perf_counter()
                    try:
                        result = generate_one(server, messages=messages,
                                              generation=generation)
                    except ContractViolation as exc:
                        aborted_reason = f"contract_violation: {exc}"
                        raise
                    except InfrastructureFailure as exc:
                        record = generation_record(
                            schema_version=1,
                            suite_id=suite["suite_id"],
                            run_id=args.run_id,
                            row_alias=args.model_alias,
                            task_id=task_id,
                            entry_point=dataset[task_id].get("entry_point", ""),
                            case_kind=case_kind,
                            attempt=attempt,
                            model_path=str(args.model),
                            model_sha256=row["sha256"],
                            prompt_id=suite["prompt"]["prompt_id"],
                            prompt_sha256=suite["prompt"]["template_sha256"],
                            templated_prompt=messages[0]["content"],
                            templated_prompt_token_ids=[],
                            raw_completion="",
                            completion_token_ids=[],
                            finish_reason="none",
                            timings={},
                            started_at_utc=started,
                            finished_at_utc=utc_now(),
                            status=f"infrastructure_{exc.failure_class}",
                        )
                        evidence.write(json.dumps(record, ensure_ascii=False) + "\n")
                        evidence.flush()
                        record_count += 1
                        decision = decide_retry(attempt, exc.failure_class)
                        if decision == RETRY_FRESH_SERVER:
                            server.stop()
                            server.restarts += 1
                            server.start()
                            attempt += 1
                            continue
                        aborted_reason = (
                            f"second infrastructure failure ({exc.failure_class})"
                            " on the same task"
                        )
                        raise ContractViolation(aborted_reason)

                    status = classify_finish_reason(
                        result["finish_reason"], result["raw_completion"]
                    )
                    record = generation_record(
                        schema_version=1,
                        suite_id=suite["suite_id"],
                        run_id=args.run_id,
                        row_alias=args.model_alias,
                        task_id=task_id,
                        entry_point=dataset[task_id].get("entry_point", ""),
                        case_kind=case_kind,
                        attempt=attempt,
                        model_path=str(args.model),
                        model_sha256=row["sha256"],
                        prompt_id=suite["prompt"]["prompt_id"],
                        prompt_sha256=suite["prompt"]["template_sha256"],
                        templated_prompt=result["templated_prompt"],
                        templated_prompt_token_ids=result["templated_prompt_token_ids"],
                        raw_completion=result["raw_completion"],
                        completion_token_ids=result["completion_token_ids"],
                        finish_reason=result["finish_reason"],
                        timings={
                            **result["timings"],
                            "wall_seconds": round(time.perf_counter() - started_clock, 4),
                        },
                        started_at_utc=started,
                        finished_at_utc=utc_now(),
                        status=status,
                    )
                    evidence.write(json.dumps(record, ensure_ascii=False) + "\n")
                    evidence.flush()
                    record_count += 1
                    break
    except ContractViolation:
        pass
    finally:
        server.stop()

    if record_count == 0:
        raise ContractViolation("Row produced no generation records")

    generations = read_jsonl(generations_path)
    summary = row_summary(generations)
    task_ids_with_terminal = {
        record["task_id"] for record in generations
        if record["status"] in ("completed", "empty_completion", "length_limit")
    }
    gates = {
        "all_pilot_tasks_have_terminal_record": (
            task_ids_with_terminal == set(task_ids)
        ),
        "no_duplicate_generation_keys": summary["duplicate_generation_keys"] == 0,
        "generated_programs_executed": summary["generated_programs_executed"] == 0,
        "row_not_aborted": aborted_reason is None,
    }
    manifest = {
        "schema_version": 1,
        "run_id": args.run_id,
        "suite_id": suite["suite_id"],
        "mode": mode_name,
        "row_alias": args.model_alias,
        "model_sha256": row["sha256"],
        "generated_at_utc": utc_now(),
        "server_restarts": server.restarts,
        "aborted_reason": aborted_reason,
        "summary": summary,
        "gates": gates,
        "passed": all(gates.values()) and aborted_reason is None,
        "results_dir": str(args.results_dir),
    }
    (args.results_dir / "run-manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, sort_keys=True))
    print(f"E3 row {args.model_alias}: "
          f"{'PASS' if manifest['passed'] else 'RED'}")
    if aborted_reason is not None:
        return EXIT_ROW_ABORTED
    return EXIT_OK if manifest["passed"] else EXIT_GATES_RED


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ContractViolation as exc:
        print(f"E3 row aborted: {exc}", file=__import__("sys").stderr)
        raise SystemExit(EXIT_ROW_ABORTED)
