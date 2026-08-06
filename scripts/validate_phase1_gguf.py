#!/usr/bin/env python3
"""Validate BF16 GGUF tokenization and greedy behavior against Phase 0 HF."""

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


def normalize_token_ids(encoded: Any) -> list[int]:
    if isinstance(encoded, dict) or hasattr(encoded, "keys"):
        encoded = encoded["input_ids"]
    if hasattr(encoded, "tolist"):
        encoded = encoded.tolist()
    if encoded and isinstance(encoded[0], list):
        if len(encoded) != 1:
            raise ValueError(f"Expected one prompt, received {len(encoded)}")
        encoded = encoded[0]
    return [int(token_id) for token_id in encoded]


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hf-model", type=Path, required=True)
    parser.add_argument("--gguf", type=Path, required=True)
    parser.add_argument("--phase0-validation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--server-log", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    phase0 = json.loads(args.phase0_validation.read_text(encoding="utf-8"))
    references = phase0["merged_inference"]["outputs"]
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
                prompt = reference["prompt"]
                messages = [{"role": "user", "content": prompt}]
                rendered = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
                hf_prompt_ids = normalize_token_ids(
                    tokenizer.apply_chat_template(
                        messages,
                        tokenize=True,
                        add_generation_prompt=True,
                    )
                )
                llama_tokens = post_json(
                    "http://127.0.0.1:8080/tokenize",
                    {
                        "content": rendered,
                        "add_special": False,
                        "parse_special": True,
                        "with_pieces": False,
                    },
                )["tokens"]

                response = post_json(
                    "http://127.0.0.1:8080/completion",
                    {
                        "prompt": hf_prompt_ids,
                        "n_predict": args.max_new_tokens,
                        "temperature": -1.0,
                        "cache_prompt": False,
                        "return_tokens": True,
                        "seed": 42,
                    },
                )
                gguf_ids = [int(token_id) for token_id in response["tokens"]]
                hf_completion_ids = [
                    int(token_id) for token_id in reference["completion_token_ids"]
                ]
                gguf_without_eos = strip_terminal_eos(gguf_ids, eos_id)
                hf_without_eos = strip_terminal_eos(hf_completion_ids, eos_id)
                prefix_length = common_prefix_length(
                    hf_without_eos, gguf_without_eos
                )
                first_token_equal = bool(
                    hf_without_eos
                    and gguf_without_eos
                    and hf_without_eos[0] == gguf_without_eos[0]
                )
                cases.append(
                    {
                        "prompt": prompt,
                        "prompt_token_ids_equal": hf_prompt_ids
                        == [int(token_id) for token_id in llama_tokens],
                        "hf_prompt_token_count": len(hf_prompt_ids),
                        "gguf_prompt_token_count": len(llama_tokens),
                        "hf_completion_token_ids": hf_completion_ids,
                        "gguf_completion_token_ids": gguf_ids,
                        "first_completion_token_equal": first_token_equal,
                        "full_completion_equal_ignoring_terminal_eos": (
                            hf_without_eos == gguf_without_eos
                        ),
                        "common_completion_prefix_tokens": prefix_length,
                        "hf_completion_tokens_without_eos": len(hf_without_eos),
                        "gguf_completion_tokens_without_eos": len(gguf_without_eos),
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
        "all_phase0_prompts_tested": len(cases) == len(references),
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
    print(f"Phase 1 GGUF validation: {'PASS' if report['passed'] else 'FAIL'}")
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
