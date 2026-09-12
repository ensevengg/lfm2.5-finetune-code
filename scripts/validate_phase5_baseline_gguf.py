#!/usr/bin/env python3
"""Validate the original-model BF16 GGUF against frozen original-HF references.

Parity bar and completion parameters are identical to Phase 1's
validate_phase1_gguf.py. The HF side of every comparison is the frozen
greedy record in the E1 original-versus-merged differential report, so no
new host-side Transformers inference is required.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def post_json(url: str, payload: dict[str, Any], timeout: float = 120) -> Any:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def get_json(url: str, timeout: float = 10) -> Any:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def common_prefix_length(left: list[int], right: list[int]) -> int:
    count = 0
    for left_id, right_id in zip(left, right):
        if left_id != right_id:
            break
        count += 1
    return count


def strip_terminal_eos(ids: list[int], eos_id: int) -> list[int]:
    result = list(ids)
    while result and result[-1] == eos_id:
        result.pop()
    return result


def wait_for_server(process: subprocess.Popen[str], timeout: float = 120) -> None:
    deadline = time.monotonic() + timeout
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"llama-server exited early with {process.returncode}")
        try:
            health = get_json("http://127.0.0.1:8080/health", timeout=2)
            if health.get("status") in {"ok", "no slot available"}:
                return
        except (OSError, urllib.error.URLError, json.JSONDecodeError) as exc:
            last_error = exc
        time.sleep(0.5)
    raise TimeoutError(f"llama-server did not become healthy: {last_error}")


def load_references(path: Path, expected_model: str) -> list[dict[str, Any]]:
    references: list[dict[str, Any]] = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            raise ValueError(f"Blank JSONL line at {path}:{number}")
        record = json.loads(line)
        if record.get("logical_model") != expected_model:
            raise ValueError(
                f"Reference {path}:{number} is for {record.get('logical_model')!r},"
                f" expected {expected_model!r}"
            )
        prompt_ids = [int(token_id) for token_id in record["templated_prompt_token_ids"]]
        bos_count = sum(1 for token_id in prompt_ids if token_id == 1)
        if bos_count != 1 or prompt_ids[0] != 1:
            raise ValueError(
                f"Reference {path}:{number} must start with exactly one BOS token;"
                f" found {bos_count}"
            )
        if not record.get("templated_prompt"):
            raise ValueError(f"Reference {path}:{number} is missing templated_prompt")
        references.append(record)
    if not references:
        raise ValueError(f"No reference records found in {path}")
    return references


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--references", type=Path, required=True,
                        help="original-hf generations.jsonl from the E1 differential run")
    parser.add_argument("--expected-model", default="original_thinking")
    parser.add_argument("--hf-model", type=Path, required=True,
                        help="original checkpoint directory (tokenizer identity + eos id)")
    parser.add_argument("--gguf", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--server-log", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    references = load_references(args.references, args.expected_model)
    tokenizer = AutoTokenizer.from_pretrained(
        args.hf_model,
        local_files_only=True,
        trust_remote_code=False,
    )
    eos_id = int(tokenizer.eos_token_id)

    server_command = [
        "/opt/llama/bin/llama-server",
        "--model",
        str(args.gguf),
        "--ctx-size",
        "4096",
        "--n-gpu-layers",
        "all",
        "--flash-attn",
        "on",
        "--cache-type-k",
        "f16",
        "--cache-type-v",
        "f16",
        "--fit",
        "off",
        "--parallel",
        "1",
        "--host",
        "127.0.0.1",
        "--port",
        "8080",
        "--log-verbosity",
        "4",
        "--offline",
        "--no-warmup",
    ]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.server_log.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    cases: list[dict[str, Any]] = []
    server_error: str | None = None

    with args.server_log.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            server_command,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )
        try:
            wait_for_server(process)
            for reference in references:
                prompt_ids = [
                    int(token_id)
                    for token_id in reference["templated_prompt_token_ids"]
                ]
                llama_tokens = post_json(
                    "http://127.0.0.1:8080/tokenize",
                    {
                        "content": reference["templated_prompt"],
                        "add_special": False,
                        "parse_special": True,
                        "with_pieces": False,
                    },
                )["tokens"]

                response = post_json(
                    "http://127.0.0.1:8080/completion",
                    {
                        "prompt": prompt_ids,
                        "n_predict": args.max_new_tokens,
                        "temperature": -1.0,
                        "cache_prompt": False,
                        "return_tokens": True,
                        "seed": 42,
                    },
                )
                gguf_ids = [int(token_id) for token_id in response["tokens"]]
                hf_reference_ids = [
                    int(token_id) for token_id in reference["completion_token_ids"]
                ][: args.max_new_tokens]
                gguf_without_eos = strip_terminal_eos(gguf_ids, eos_id)
                hf_without_eos = strip_terminal_eos(hf_reference_ids, eos_id)
                prefix_length = common_prefix_length(hf_without_eos, gguf_without_eos)
                first_token_equal = bool(
                    hf_without_eos
                    and gguf_without_eos
                    and hf_without_eos[0] == gguf_without_eos[0]
                )
                cases.append(
                    {
                        "case_id": reference["case_id"],
                        "case_kind": reference["case_kind"],
                        "prompt_token_ids_equal": prompt_ids
                        == [int(token_id) for token_id in llama_tokens],
                        "prompt_token_count": len(prompt_ids),
                        "hf_reference_completion_tokens": len(hf_reference_ids),
                        "gguf_completion_token_ids": gguf_ids,
                        "first_completion_token_equal": first_token_equal,
                        "full_completion_equal_ignoring_terminal_eos": (
                            hf_without_eos == gguf_without_eos
                        ),
                        "common_completion_prefix_tokens": prefix_length,
                        "stop_type": response.get("stop_type"),
                        "timings": response.get("timings"),
                    }
                )
        except Exception as exc:
            server_error = f"{type(exc).__name__}: {exc}"
        finally:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10)

    server_log = args.server_log.read_text(encoding="utf-8", errors="replace")
    all_tokenizers_equal = bool(cases) and all(
        case["prompt_token_ids_equal"] for case in cases
    )
    all_first_tokens_equal = bool(cases) and all(
        case["first_completion_token_equal"] for case in cases
    )
    full_equal_count = sum(
        case["full_completion_equal_ignoring_terminal_eos"] for case in cases
    )
    cuda_detected = (
        "CUDA0" in server_log
        and ("offloaded" in server_log or "offloading" in server_log)
        and "GPU" in server_log
    )
    gates = {
        "server_started_without_error": server_error is None,
        "all_reference_prompts_tested": len(cases) == len(references),
        "hf_and_gguf_prompt_token_ids_match": all_tokenizers_equal,
        "hf_and_gguf_first_greedy_tokens_match": all_first_tokens_equal,
        "cuda_model_offload_detected": cuda_detected,
    }
    report = {
        "schema_version": 1,
        "generated_at_utc": utc_now(),
        "passed": all(gates.values()),
        "duration_seconds": round(time.perf_counter() - started, 3),
        "hf_tokenizer_runtime_class": type(tokenizer).__name__,
        "references_path": str(args.references),
        "expected_logical_model": args.expected_model,
        "eos_token_id": eos_id,
        "server_command": server_command,
        "server_error": server_error,
        "gates": gates,
        "summary": {
            "case_count": len(cases),
            "tokenizer_equal_count": sum(
                case["prompt_token_ids_equal"] for case in cases
            ),
            "first_greedy_token_equal_count": sum(
                case["first_completion_token_equal"] for case in cases
            ),
            "full_greedy_completion_equal_count": full_equal_count,
        },
        "cases": cases,
    }
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report["summary"], sort_keys=True))
    print(f"Phase 5 baseline GGUF validation: {'PASS' if report['passed'] else 'FAIL'}")
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
