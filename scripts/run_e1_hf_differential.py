#!/usr/bin/env python3
"""Compare frozen E1 prompts on original and merged Hugging Face checkpoints."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import platform
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from review_e1 import review as review_behavior


PLACEHOLDER = "{{BENCHMARK_PROMPT}}"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected one JSON object in {path}")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            raise ValueError(f"Blank JSONL line at {path}:{number}")
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"Expected an object at {path}:{number}")
        records.append(value)
    return records


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def expected_model(manifest: dict[str, Any]) -> dict[str, Any]:
    matches = [item for item in manifest["files"] if item["path"] == "model.safetensors"]
    if len(matches) != 1:
        raise ValueError("Checkpoint manifest must contain exactly one model.safetensors")
    return matches[0]


def validate_checkpoint(path: Path, manifest_path: Path) -> dict[str, Any]:
    manifest = read_json(manifest_path)
    expected = expected_model(manifest)
    weights = path / "model.safetensors"
    actual_hash = sha256_file(weights)
    actual_size = weights.stat().st_size
    return {
        "logical_name": manifest.get("logical_name"),
        "checkpoint": str(path.resolve()),
        "manifest": str(manifest_path.resolve()),
        "expected_sha256": expected["sha256"],
        "actual_sha256": actual_hash,
        "expected_size_bytes": expected["size_bytes"],
        "actual_size_bytes": actual_size,
        "passed": actual_hash == expected["sha256"] and actual_size == expected["size_bytes"],
    }


def build_cases(
    suite: dict[str, Any], prompts: list[dict[str, Any]], template: str
) -> list[dict[str, Any]]:
    cases = [
        {
            "case_id": case["case_id"],
            "case_kind": "prompt_integrity",
            "task_id": None,
            "entry_point": None,
            "messages": case["messages"],
        }
        for case in suite["prompt_integrity_cases"]
    ]
    cases.extend(
        {
            "case_id": "humanevalplus-" + record["task_id"].replace("/", "-"),
            "case_kind": "humanevalplus_development",
            "task_id": record["task_id"],
            "entry_point": record["entry_point"],
            "messages": [
                {
                    "role": "user",
                    "content": template.replace(PLACEHOLDER, record["prompt"]),
                }
            ],
        }
        for record in prompts
    )
    return cases


def run_checkpoint(
    checkpoint: Path,
    logical_name: str,
    cases: list[dict[str, Any]],
    suite: dict[str, Any],
    max_new_tokens: int,
    seed: int,
    output_dir: Path,
) -> dict[str, Any]:
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this BF16 diagnostic")
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    print(f"Loading {logical_name} from frozen local files...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(
        checkpoint, local_files_only=True, trust_remote_code=False
    )
    loaded_at = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint,
        local_files_only=True,
        trust_remote_code=False,
        dtype=torch.bfloat16,
        device_map={"": "cuda"},
    )
    model.eval()
    load_seconds = time.perf_counter() - loaded_at
    records: list[dict[str, Any]] = []
    eos_ids = model.generation_config.eos_token_id
    if isinstance(eos_ids, int):
        eos_ids = [eos_ids]
    eos_ids = set(eos_ids or [])

    for index, case in enumerate(cases, 1):
        label = case.get("task_id") or case["case_id"]
        print(f"[{logical_name} {index}/{len(cases)}] {label}", flush=True)
        rendered = tokenizer.apply_chat_template(
            case["messages"], add_generation_prompt=True, tokenize=False
        )
        inputs = tokenizer.apply_chat_template(
            case["messages"],
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(model.device)
        prompt_ids = inputs["input_ids"][0].detach().cpu().tolist()
        if len(prompt_ids) + max_new_tokens > 4096:
            raise RuntimeError(
                f"Context overflow for {case['case_id']}: {len(prompt_ids)} + {max_new_tokens}"
            )
        started = time.perf_counter()
        with torch.inference_mode():
            generated = model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=max_new_tokens,
                use_cache=True,
                pad_token_id=tokenizer.pad_token_id,
            )
        torch.cuda.synchronize()
        generated_ids = generated[0, len(prompt_ids) :].detach().cpu().tolist()
        stopped_eos = bool(generated_ids and generated_ids[-1] in eos_ids)
        content_ids = generated_ids[:-1] if stopped_eos else generated_ids
        completion = tokenizer.decode(content_ids, skip_special_tokens=True)
        record = {
            "schema_version": 1,
            "runtime": "transformers-bf16",
            "logical_model": logical_name,
            "case_id": case["case_id"],
            "case_kind": case["case_kind"],
            "task_id": case.get("task_id"),
            "entry_point": case.get("entry_point"),
            "messages": case["messages"],
            "templated_prompt": rendered,
            "templated_prompt_token_ids": prompt_ids,
            "templated_prompt_token_count": len(prompt_ids),
            "raw_completion": completion,
            "completion_token_ids": content_ids,
            "completion_token_count": len(content_ids),
            "finish_reason": "eos" if stopped_eos else "limit",
            "duration_seconds": round(time.perf_counter() - started, 3),
            "generated_program_executed": False,
            "scored": False,
        }
        records.append(record)
        write_jsonl(output_dir / "generations.jsonl", records)
        print(
            f"  saved {len(content_ids)} tokens, stop={record['finish_reason']}",
            flush=True,
        )

    review = review_behavior(suite, records)
    write_json(output_dir / "behavior-review.json", review)
    summary = {
        "schema_version": 1,
        "generated_at_utc": utc_now(),
        "logical_model": logical_name,
        "checkpoint": str(checkpoint.resolve()),
        "load_seconds": round(load_seconds, 3),
        "generation_count": len(records),
        "behavior_review_passed": review["passed"],
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(),
        "generated_programs_executed": 0,
        "scores_computed": 0,
    }
    write_json(output_dir / "run-summary.json", summary)
    del model
    del tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    return {"records": records, "review": review, "summary": summary}


def compare_runs(
    base: dict[str, Any],
    merged: dict[str, Any],
    gguf_records: list[dict[str, Any]] | None,
) -> dict[str, Any]:
    base_by_id = {record["case_id"]: record for record in base["records"]}
    merged_by_id = {record["case_id"]: record for record in merged["records"]}
    gguf_by_id = {record["case_id"]: record for record in gguf_records or []}
    cases = []
    for case_id in base_by_id:
        base_record = base_by_id[case_id]
        merged_record = merged_by_id[case_id]
        gguf_record = gguf_by_id.get(case_id)
        cases.append(
            {
                "case_id": case_id,
                "prompt_token_ids_equal_base_vs_merged_hf": (
                    base_record["templated_prompt_token_ids"]
                    == merged_record["templated_prompt_token_ids"]
                ),
                "completion_equal_base_vs_merged_hf": (
                    base_record["raw_completion"] == merged_record["raw_completion"]
                ),
                "completion_equal_merged_hf_vs_merged_gguf": (
                    merged_record["raw_completion"] == gguf_record["raw_completion"]
                    if gguf_record
                    else None
                ),
                "base": {
                    "tokens": base_record["completion_token_count"],
                    "finish_reason": base_record["finish_reason"],
                    "completion_sha256": hashlib.sha256(
                        base_record["raw_completion"].encode("utf-8")
                    ).hexdigest(),
                },
                "merged_hf": {
                    "tokens": merged_record["completion_token_count"],
                    "finish_reason": merged_record["finish_reason"],
                    "completion_sha256": hashlib.sha256(
                        merged_record["raw_completion"].encode("utf-8")
                    ).hexdigest(),
                },
                "merged_gguf": {
                    "tokens": gguf_record["completion_token_count"],
                    "finish_reason": gguf_record["finish_reason"],
                    "completion_sha256": hashlib.sha256(
                        gguf_record["raw_completion"].encode("utf-8")
                    ).hexdigest(),
                }
                if gguf_record
                else None,
            }
        )
    return {
        "schema_version": 1,
        "generated_at_utc": utc_now(),
        "base_behavior_review_passed": base["review"]["passed"],
        "merged_hf_behavior_review_passed": merged["review"]["passed"],
        "cases": cases,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--prompt-template", type=Path, required=True)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--base-manifest", type=Path, required=True)
    parser.add_argument("--merged", type=Path, required=True)
    parser.add_argument("--merged-manifest", type=Path, required=True)
    parser.add_argument("--gguf-generations", type=Path)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.results_dir.mkdir(parents=True, exist_ok=False)
    suite = read_json(args.suite)
    prompts = read_jsonl(args.prompts)
    template = args.prompt_template.read_text(encoding="utf-8")
    max_new_tokens = args.max_new_tokens or suite["generation"]["max_output_tokens"]
    validation = {
        "schema_version": 1,
        "generated_at_utc": utc_now(),
        "offline": True,
        "base": validate_checkpoint(args.base, args.base_manifest),
        "merged": validate_checkpoint(args.merged, args.merged_manifest),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
    }
    validation["passed"] = validation["base"]["passed"] and validation["merged"]["passed"]
    write_json(args.results_dir / "validation.json", validation)
    if not validation["passed"]:
        print("Checkpoint validation: FAIL", flush=True)
        return 2
    print("Checkpoint validation: PASS", flush=True)
    cases = build_cases(suite, prompts, template)
    base = run_checkpoint(
        args.base,
        "original_thinking",
        cases,
        suite,
        max_new_tokens,
        suite["generation"]["seed"],
        args.results_dir / "original-hf",
    )
    merged = run_checkpoint(
        args.merged,
        "merged_finetune",
        cases,
        suite,
        max_new_tokens,
        suite["generation"]["seed"],
        args.results_dir / "merged-hf",
    )
    gguf = read_jsonl(args.gguf_generations) if args.gguf_generations else None
    comparison = compare_runs(base, merged, gguf)
    write_json(args.results_dir / "comparison.json", comparison)
    print("Differential evidence saved. Generated programs executed: 0", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
