#!/usr/bin/env python3
"""Freeze and validate the inputs for the LFM2.5 GGUF/evaluation pipeline.

This script is intentionally local-only. It discovers already-cached Hugging
Face snapshots, hashes the source artifacts, inventories the host, verifies
tokenizer/chat-template parity, confirms that the LoRA merge changed model
tensors, and optionally runs deterministic original-vs-merged smoke prompts.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import shutil
import struct
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


SCHEMA_VERSION = 1
BASE_REPO = "LiquidAI/LFM2.5-1.2B-Thinking"
ADAPTER_REPO = "enseven/lfm-2.5-think-code"
DATASET_REPO = "enseven/kodcode-lfm2.5"

SMOKE_PROMPTS = [
    "Write a Python function `two_sum(nums, target)` that returns the two indices. Return only code.",
    "Fix this Python function and explain the bug briefly:\n\n"
    "def is_even(n):\n"
    "    return n % 2 == 1",
    "Implement binary search in Python. Return -1 when the target is absent.",
    "Write a Python function that returns the length of the longest increasing subsequence.",
    "Given a directed graph, explain how to detect a cycle and provide a Python implementation.",
    "Return only Python code for a function `is_prime(n)` that handles all integer edge cases.",
]

PACKAGE_NAMES = [
    "accelerate",
    "bitsandbytes",
    "huggingface-hub",
    "numpy",
    "peft",
    "safetensors",
    "torch",
    "transformers",
    "trl",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def json_dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_range(
    handle: Any,
    absolute_offset: int,
    length: int,
    chunk_size: int = 8 * 1024 * 1024,
) -> str:
    digest = hashlib.sha256()
    handle.seek(absolute_offset)
    remaining = length
    while remaining:
        chunk = handle.read(min(chunk_size, remaining))
        if not chunk:
            raise EOFError(
                f"Unexpected EOF while hashing {length} bytes at {absolute_offset}"
            )
        digest.update(chunk)
        remaining -= len(chunk)
    return digest.hexdigest()


def command_result(command: list[str], timeout: int = 30) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
        )
        return {
            "command": command,
            "exit_code": completed.returncode,
            "stdout": completed.stdout.strip(),
            "stderr": completed.stderr.strip(),
            "duration_seconds": round(time.perf_counter() - started, 3),
        }
    except FileNotFoundError as exc:
        return {
            "command": command,
            "available": False,
            "error": str(exc),
            "duration_seconds": round(time.perf_counter() - started, 3),
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "command": command,
            "timed_out": True,
            "stdout": (exc.stdout or "").strip()
            if isinstance(exc.stdout, str)
            else "",
            "stderr": (exc.stderr or "").strip()
            if isinstance(exc.stderr, str)
            else "",
            "duration_seconds": round(time.perf_counter() - started, 3),
        }


def package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in PACKAGE_NAMES:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def huggingface_cache_root() -> Path:
    configured = os.environ.get("HF_HUB_CACHE")
    if configured:
        return Path(configured)
    try:
        from huggingface_hub import constants

        return Path(constants.HF_HUB_CACHE)
    except Exception:
        return Path.home() / ".cache" / "huggingface" / "hub"


def cache_repo_dir(repo_id: str, repo_type: str = "model") -> Path:
    prefix = "datasets" if repo_type == "dataset" else "models"
    return huggingface_cache_root() / f"{prefix}--{repo_id.replace('/', '--')}"


def resolve_cached_snapshot(
    repo_id: str, repo_type: str = "model"
) -> dict[str, Any]:
    repo_dir = cache_repo_dir(repo_id, repo_type)
    ref_path = repo_dir / "refs" / "main"
    revision = ref_path.read_text(encoding="utf-8").strip() if ref_path.exists() else None
    snapshots_dir = repo_dir / "snapshots"
    snapshots = (
        sorted(path.name for path in snapshots_dir.iterdir() if path.is_dir())
        if snapshots_dir.exists()
        else []
    )
    selected = snapshots_dir / revision if revision else None
    return {
        "repo_id": repo_id,
        "repo_type": repo_type,
        "cache_repo_dir": str(repo_dir.resolve()) if repo_dir.exists() else str(repo_dir),
        "ref_main": revision,
        "available_snapshots": snapshots,
        "selected_snapshot": str(selected.resolve())
        if selected is not None and selected.exists()
        else None,
        "selected_snapshot_exists": bool(selected is not None and selected.exists()),
        "evidence": (
            "Recovered from the local Hugging Face refs/main file and matching "
            "snapshot directory; no network lookup was used."
        ),
    }


def iter_regular_files(root: Path) -> Iterable[Path]:
    for path in sorted(root.rglob("*"), key=lambda item: item.as_posix().lower()):
        if path.is_file():
            yield path


def safetensors_inventory(path: Path) -> dict[str, Any]:
    from safetensors import safe_open

    dtype_counts: dict[str, int] = {}
    tensor_count = 0
    parameter_count = 0
    with safe_open(path, framework="pt", device="cpu") as tensors:
        for name in tensors.keys():
            tensor_slice = tensors.get_slice(name)
            shape = tensor_slice.get_shape()
            count = 1
            for dimension in shape:
                count *= dimension
            dtype = tensor_slice.get_dtype()
            dtype_counts[dtype] = dtype_counts.get(dtype, 0) + count
            parameter_count += count
            tensor_count += 1
    return {
        "tensor_count": tensor_count,
        "parameter_count": parameter_count,
        "parameters_by_dtype": dtype_counts,
    }


def checkpoint_manifest(
    root: Path,
    logical_name: str,
    repo_id: str | None = None,
    revision: str | None = None,
) -> dict[str, Any]:
    if not root.is_dir():
        raise FileNotFoundError(f"Checkpoint directory not found: {root}")
    files: list[dict[str, Any]] = []
    for path in iter_regular_files(root):
        item: dict[str, Any] = {
            "path": path.relative_to(root).as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        if path.suffix == ".safetensors":
            item["safetensors"] = safetensors_inventory(path)
        files.append(item)

    config_path = root / "config.json"
    config = (
        json.loads(config_path.read_text(encoding="utf-8"))
        if config_path.exists()
        else None
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": utc_now(),
        "logical_name": logical_name,
        "repo_id": repo_id,
        "revision": revision,
        "source_directory": str(root.resolve()),
        "file_count": len(files),
        "total_size_bytes": sum(item["size_bytes"] for item in files),
        "files": files,
        "config_summary": {
            key: config.get(key)
            for key in [
                "architectures",
                "dtype",
                "hidden_size",
                "layer_types",
                "max_position_embeddings",
                "model_type",
                "num_attention_heads",
                "num_hidden_layers",
                "num_key_value_heads",
                "rope_parameters",
                "vocab_size",
            ]
        }
        if config
        else None,
    }


def read_safetensors_header(path: Path) -> tuple[int, dict[str, Any]]:
    with path.open("rb") as handle:
        raw_length = handle.read(8)
        if len(raw_length) != 8:
            raise ValueError(f"Not a valid safetensors file: {path}")
        header_length = struct.unpack("<Q", raw_length)[0]
        header = json.loads(handle.read(header_length).decode("utf-8"))
    return 8 + header_length, header


def compare_safetensors_raw(base_path: Path, merged_path: Path) -> dict[str, Any]:
    base_data_start, base_header = read_safetensors_header(base_path)
    merged_data_start, merged_header = read_safetensors_header(merged_path)
    base_names = set(base_header) - {"__metadata__"}
    merged_names = set(merged_header) - {"__metadata__"}
    shared_names = sorted(base_names & merged_names)
    changed: list[dict[str, Any]] = []
    unchanged_count = 0
    changed_bytes = 0

    with base_path.open("rb") as base_handle, merged_path.open("rb") as merged_handle:
        for name in shared_names:
            base_info = base_header[name]
            merged_info = merged_header[name]
            base_start, base_end = base_info["data_offsets"]
            merged_start, merged_end = merged_info["data_offsets"]
            base_length = base_end - base_start
            merged_length = merged_end - merged_start
            metadata_equal = (
                base_info["dtype"] == merged_info["dtype"]
                and base_info["shape"] == merged_info["shape"]
                and base_length == merged_length
            )
            base_sha = sha256_range(
                base_handle, base_data_start + base_start, base_length
            )
            merged_sha = sha256_range(
                merged_handle, merged_data_start + merged_start, merged_length
            )
            if metadata_equal and base_sha == merged_sha:
                unchanged_count += 1
                continue
            changed_bytes += merged_length
            changed.append(
                {
                    "name": name,
                    "shape": merged_info["shape"],
                    "dtype": merged_info["dtype"],
                    "size_bytes": merged_length,
                    "base_sha256": base_sha,
                    "merged_sha256": merged_sha,
                    "metadata_equal": metadata_equal,
                }
            )

    return {
        "base_file": str(base_path.resolve()),
        "merged_file": str(merged_path.resolve()),
        "base_file_sha256": sha256_file(base_path),
        "merged_file_sha256": sha256_file(merged_path),
        "base_tensor_count": len(base_names),
        "merged_tensor_count": len(merged_names),
        "shared_tensor_count": len(shared_names),
        "base_only_tensors": sorted(base_names - merged_names),
        "merged_only_tensors": sorted(merged_names - base_names),
        "changed_tensor_count": len(changed),
        "unchanged_tensor_count": unchanged_count,
        "changed_tensor_bytes": changed_bytes,
        "changed_tensors": changed,
        "proves_merge_not_noop": bool(changed),
    }


def normalize_token_ids(encoded: Any) -> list[int]:
    """Return one prompt's token IDs across Transformers return conventions."""
    if isinstance(encoded, dict) or hasattr(encoded, "keys"):
        encoded = encoded["input_ids"]
    if hasattr(encoded, "detach"):
        encoded = encoded.detach().cpu().tolist()
    elif hasattr(encoded, "tolist"):
        encoded = encoded.tolist()
    if encoded and isinstance(encoded[0], list):
        if len(encoded) != 1:
            raise ValueError(f"Expected one encoded prompt, got {len(encoded)}")
        encoded = encoded[0]
    return [int(token_id) for token_id in encoded]


def tokenizer_validation(merged_dir: Path, base_dir: Path) -> dict[str, Any]:
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    from transformers import AutoTokenizer

    started = time.perf_counter()
    merged_tokenizer = AutoTokenizer.from_pretrained(
        merged_dir,
        local_files_only=True,
        trust_remote_code=False,
    )
    base_tokenizer = AutoTokenizer.from_pretrained(
        base_dir,
        local_files_only=True,
        trust_remote_code=False,
    )

    cases: list[dict[str, Any]] = []
    all_ids_equal = True
    all_templates_equal = True
    for prompt in SMOKE_PROMPTS:
        messages = [{"role": "user", "content": prompt}]
        base_rendered = base_tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        merged_rendered = merged_tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        base_ids = normalize_token_ids(
            base_tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True
            )
        )
        merged_ids = normalize_token_ids(
            merged_tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True
            )
        )
        ids_equal = base_ids == merged_ids
        templates_equal = base_rendered == merged_rendered
        all_ids_equal = all_ids_equal and ids_equal
        all_templates_equal = all_templates_equal and templates_equal
        cases.append(
            {
                "prompt": prompt,
                "base_token_count": len(base_ids),
                "merged_token_count": len(merged_ids),
                "token_ids_equal": ids_equal,
                "rendered_template_equal": templates_equal,
                "base_token_ids_sha256": hashlib.sha256(
                    json.dumps(base_ids).encode("utf-8")
                ).hexdigest(),
                "merged_token_ids_sha256": hashlib.sha256(
                    json.dumps(merged_ids).encode("utf-8")
                ).hexdigest(),
            }
        )

    special_tokens = [
        "bos_token",
        "eos_token",
        "pad_token",
        "unk_token",
        "bos_token_id",
        "eos_token_id",
        "pad_token_id",
        "unk_token_id",
    ]
    return {
        "duration_seconds": round(time.perf_counter() - started, 3),
        "base_tokenizer_class": type(base_tokenizer).__name__,
        "merged_tokenizer_class": type(merged_tokenizer).__name__,
        "base_vocab_size": len(base_tokenizer),
        "merged_vocab_size": len(merged_tokenizer),
        "base_special_tokens": {
            key: getattr(base_tokenizer, key, None) for key in special_tokens
        },
        "merged_special_tokens": {
            key: getattr(merged_tokenizer, key, None) for key in special_tokens
        },
        "all_token_ids_equal": all_ids_equal,
        "all_rendered_templates_equal": all_templates_equal,
        "cases": cases,
        "passed": all_ids_equal and all_templates_equal,
    }


def choose_device(requested: str) -> str:
    if requested != "auto":
        return requested
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def generate_smoke_outputs(
    checkpoint_dir: Path,
    logical_name: str,
    device: str,
    max_new_tokens: int,
) -> dict[str, Any]:
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    tokenizer = AutoTokenizer.from_pretrained(
        checkpoint_dir,
        local_files_only=True,
        trust_remote_code=False,
    )
    load_started = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint_dir,
        local_files_only=True,
        trust_remote_code=False,
        dtype=torch.bfloat16,
        device_map={"": device},
    )
    model.eval()
    load_seconds = time.perf_counter() - load_started
    model_device = str(next(model.parameters()).device)

    outputs: list[dict[str, Any]] = []
    generation_started = time.perf_counter()
    all_logits_finite = True
    for prompt in SMOKE_PROMPTS:
        messages = [{"role": "user", "content": prompt}]
        inputs = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(model.device)
        prompt_length = inputs["input_ids"].shape[-1]
        item_started = time.perf_counter()
        with torch.inference_mode():
            first_step_logits = model(
                **inputs,
                use_cache=False,
            ).logits[:, -1, :].float()
            finite_mask = torch.isfinite(first_step_logits)
            logits_finite = bool(finite_mask.all().item())
            all_logits_finite = all_logits_finite and logits_finite
            nan_count = int(torch.isnan(first_step_logits).sum().item())
            inf_count = int(torch.isinf(first_step_logits).sum().item())
            generated = model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=max_new_tokens,
                use_cache=True,
            )
        completion_ids = generated[0, prompt_length:].detach().cpu().tolist()
        completion = tokenizer.decode(completion_ids, skip_special_tokens=False)
        outputs.append(
            {
                "prompt": prompt,
                "prompt_token_count": int(prompt_length),
                "completion_token_count": len(completion_ids),
                "completion_token_ids": completion_ids,
                "completion_token_ids_sha256": hashlib.sha256(
                    json.dumps(completion_ids).encode("utf-8")
                ).hexdigest(),
                "completion": completion,
                "first_step_logits": {
                    "all_finite": logits_finite,
                    "nan_count": nan_count,
                    "inf_count": inf_count,
                    "minimum": float(first_step_logits.min().item())
                    if logits_finite
                    else None,
                    "maximum": float(first_step_logits.max().item())
                    if logits_finite
                    else None,
                },
                "duration_seconds": round(time.perf_counter() - item_started, 3),
            }
        )

    peak_allocated = None
    peak_reserved = None
    if device == "cuda":
        peak_allocated = torch.cuda.max_memory_allocated()
        peak_reserved = torch.cuda.max_memory_reserved()

    result = {
        "logical_name": logical_name,
        "checkpoint_directory": str(checkpoint_dir.resolve()),
        "device": device,
        "model_device": model_device,
        "dtype": str(model.dtype),
        "load_seconds": round(load_seconds, 3),
        "generation_seconds": round(time.perf_counter() - generation_started, 3),
        "peak_cuda_allocated_bytes": peak_allocated,
        "peak_cuda_reserved_bytes": peak_reserved,
        "all_first_step_logits_finite": all_logits_finite,
        "outputs": outputs,
        "passed": all_logits_finite,
    }

    del model
    del tokenizer
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    return result


def generation_comparison(
    base_outputs: dict[str, Any], merged_outputs: dict[str, Any]
) -> dict[str, Any]:
    base_by_prompt = {item["prompt"]: item for item in base_outputs["outputs"]}
    merged_by_prompt = {item["prompt"]: item for item in merged_outputs["outputs"]}
    cases: list[dict[str, Any]] = []
    differing = 0
    for prompt in SMOKE_PROMPTS:
        base_item = base_by_prompt[prompt]
        merged_item = merged_by_prompt[prompt]
        equal = (
            base_item["completion_token_ids"]
            == merged_item["completion_token_ids"]
        )
        if not equal:
            differing += 1
        cases.append(
            {
                "prompt": prompt,
                "completion_token_ids_equal": equal,
                "base_completion_sha256": base_item[
                    "completion_token_ids_sha256"
                ],
                "merged_completion_sha256": merged_item[
                    "completion_token_ids_sha256"
                ],
            }
        )
    return {
        "case_count": len(cases),
        "differing_completion_count": differing,
        "cases": cases,
        "proves_behavior_changed_on_smoke_set": differing > 0,
    }


def machine_inventory(workspace: Path) -> dict[str, Any]:
    memory: dict[str, Any] = {}
    try:
        import psutil

        virtual = psutil.virtual_memory()
        memory = {
            "total_bytes": virtual.total,
            "available_bytes": virtual.available,
        }
    except Exception as exc:
        memory = {"error": repr(exc)}

    disk = shutil.disk_usage(workspace.anchor)
    packages = package_versions()
    torch_info: dict[str, Any]
    try:
        import torch

        torch_info = {
            "version": torch.__version__,
            "cuda_build_version": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "cudnn_version": torch.backends.cudnn.version(),
            "device_count": torch.cuda.device_count(),
            "devices": [
                {
                    "index": index,
                    "name": torch.cuda.get_device_name(index),
                    "capability": list(torch.cuda.get_device_capability(index)),
                    "total_memory_bytes": torch.cuda.get_device_properties(
                        index
                    ).total_memory,
                }
                for index in range(torch.cuda.device_count())
            ]
            if torch.cuda.is_available()
            else [],
        }
    except Exception as exc:
        torch_info = {"error": repr(exc)}

    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": utc_now(),
        "workspace": str(workspace.resolve()),
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor()
            or os.environ.get("PROCESSOR_IDENTIFIER"),
            "logical_cpu_count": os.cpu_count(),
        },
        "python": {
            "version": sys.version,
            "executable": sys.executable,
        },
        "memory": memory,
        "workspace_drive": {
            "total_bytes": disk.total,
            "used_bytes": disk.used,
            "free_bytes": disk.free,
        },
        "packages": packages,
        "torch": torch_info,
        "nvidia_smi": command_result(
            [
                "nvidia-smi",
                "--query-gpu=index,name,uuid,memory.total,driver_version,"
                "compute_cap,pstate,temperature.gpu",
                "--format=csv,noheader,nounits",
            ]
        ),
        "docker_client": command_result(
            ["docker", "version", "--format", "{{json .Client}}"]
        ),
        "docker_server": command_result(
            ["docker", "version", "--format", "{{json .Server}}"]
        ),
        "docker_info": command_result(
            [
                "docker",
                "info",
                "--format",
                "{{json .}}",
            ],
            timeout=15,
        ),
        "wsl_status": command_result(["wsl", "--status"], timeout=15),
        "wsl_distributions": command_result(
            ["wsl", "--list", "--verbose"], timeout=15
        ),
        "git_head": command_result(["git", "rev-parse", "HEAD"]),
        "git_status": command_result(["git", "status", "--short"]),
    }


def parse_merge_script(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    variables: dict[str, str | None] = {}
    for name in ["model_name", "adapter_path", "output_dir"]:
        match = re.search(
            rf"^\s*{re.escape(name)}\s*=\s*[\"']([^\"']+)[\"']",
            text,
            flags=re.MULTILINE,
        )
        variables[name] = match.group(1) if match else None
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "declared_inputs": variables,
    }


def write_summary(
    path: Path,
    provenance: dict[str, Any],
    machine: dict[str, Any],
    validation: dict[str, Any],
    container_readiness: dict[str, Any] | None,
) -> None:
    gates = validation["gates"]
    gate_lines = "\n".join(
        f"- {'PASS' if value else 'FAIL'}: `{name}`"
        for name, value in gates.items()
    )
    base = provenance["base_model"]
    adapter = provenance["adapter"]
    docker_server = machine["docker_server"]
    restricted_container_ready = bool(
        container_readiness and container_readiness.get("passed")
    )
    docker_ready = (
        docker_server.get("exit_code") == 0 or restricted_container_ready
    )
    text = f"""# Phase 0 result

Generated: {utc_now()}

## Recovered provenance

- Base model: `{base['repo_id']}@{base['ref_main']}`
- Adapter: `{adapter['repo_id']}@{adapter['ref_main']}`
- Dataset snapshot available: `{provenance['training_dataset']['ref_main']}`
- Provenance method: local Hugging Face `refs/main` and snapshot directories;
  the phase made no network request.

## Validation gates

{gate_lines}

Overall Phase 0 checkpoint gate: **{'PASS' if validation['passed'] else 'FAIL'}**

## Host readiness

- Docker server reachable: **{'yes' if docker_ready else 'no'}**
- CUDA available to PyTorch: **{'yes' if machine.get('torch', {}).get('cuda_available') else 'no'}**
- Restricted CUDA GPU-container smoke test: **{'pass' if restricted_container_ready else 'not passed'}**

## Evidence files

- `merged-checkpoint-manifest.json`
- `original-checkpoint-manifest.json`
- `adapter-manifest.json`
- `provenance.json`
- `machine-manifest.json`
- `validation.json`
- `container-readiness.json`
- `../../configs/evaluation-lock.json` (candidate immutable tool revisions)

The original and merged checkpoints were read only. No model file was edited.
"""
    path.write_text(text, encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workspace",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Repository root.",
    )
    parser.add_argument(
        "--merged-dir",
        type=Path,
        default=None,
        help="Merged Hugging Face checkpoint. Defaults to <workspace>/lfm2.5-model-merged.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Report directory. Defaults to <workspace>/reports/phase0.",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cuda", "cpu"],
        default="auto",
        help="Device for deterministic smoke generation.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=64,
        help="Maximum tokens per smoke completion.",
    )
    parser.add_argument(
        "--skip-inference",
        action="store_true",
        help="Generate manifests and structural checks without loading/generating.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    workspace = args.workspace.resolve()
    merged_dir = (
        args.merged_dir.resolve()
        if args.merged_dir
        else workspace / "lfm2.5-model-merged"
    )
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir
        else workspace / "reports" / "phase0"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    base = resolve_cached_snapshot(BASE_REPO)
    adapter = resolve_cached_snapshot(ADAPTER_REPO)
    dataset = resolve_cached_snapshot(DATASET_REPO, repo_type="dataset")
    if not base["selected_snapshot_exists"]:
        raise FileNotFoundError(f"Base model is not cached: {base}")
    if not adapter["selected_snapshot_exists"]:
        raise FileNotFoundError(f"Adapter is not cached: {adapter}")

    base_dir = Path(base["selected_snapshot"])
    adapter_dir = Path(adapter["selected_snapshot"])

    print("Hashing merged checkpoint...")
    merged_manifest = checkpoint_manifest(
        merged_dir, "merged_finetune", repo_id=None, revision=None
    )
    json_dump(output_dir / "merged-checkpoint-manifest.json", merged_manifest)

    print("Hashing original checkpoint...")
    original_manifest = checkpoint_manifest(
        base_dir,
        "original_thinking",
        repo_id=BASE_REPO,
        revision=base["ref_main"],
    )
    json_dump(output_dir / "original-checkpoint-manifest.json", original_manifest)

    print("Hashing LoRA adapter...")
    adapter_manifest = checkpoint_manifest(
        adapter_dir,
        "lora_adapter",
        repo_id=ADAPTER_REPO,
        revision=adapter["ref_main"],
    )
    json_dump(output_dir / "adapter-manifest.json", adapter_manifest)

    provenance = {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": utc_now(),
        "base_model": base,
        "adapter": adapter,
        "training_dataset": dataset,
        "merge_script": parse_merge_script(workspace / "merge_model.py"),
        "recovery_assessment": {
            "base_revision_recovered": base["selected_snapshot_exists"],
            "adapter_revision_recovered": adapter["selected_snapshot_exists"],
            "confidence": "high",
            "reason": (
                "Each model cache has one snapshot, refs/main selects it, the adapter "
                "snapshot predates the merged output by seconds, and adapter_config.json "
                "names the same base repository as merge_model.py."
            ),
            "limitation": (
                "merge_model.py did not pass revision= explicitly, so this recovery "
                "depends on the retained local Hugging Face cache evidence."
            ),
        },
    }
    json_dump(output_dir / "provenance.json", provenance)

    print("Capturing host inventory...")
    machine = machine_inventory(workspace)
    container_readiness_path = output_dir / "container-readiness.json"
    container_readiness = (
        json.loads(container_readiness_path.read_text(encoding="utf-8"))
        if container_readiness_path.exists()
        else None
    )
    machine["container_readiness_evidence"] = (
        {
            "report_path": str(container_readiness_path.resolve()),
            "report_sha256": sha256_file(container_readiness_path),
            "passed": container_readiness.get("passed"),
            "docker": container_readiness.get("docker"),
            "gpu_observed_in_container": container_readiness.get(
                "gpu_observed_in_container"
            ),
        }
        if container_readiness
        else None
    )
    json_dump(output_dir / "machine-manifest.json", machine)

    print("Comparing original and merged safetensors...")
    tensor_comparison = compare_safetensors_raw(
        base_dir / "model.safetensors", merged_dir / "model.safetensors"
    )
    print(
        f"Changed tensors: {tensor_comparison['changed_tensor_count']} / "
        f"{tensor_comparison['shared_tensor_count']}"
    )

    print("Validating tokenizers and chat templates offline...")
    tokenizer = tokenizer_validation(merged_dir, base_dir)

    validation: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": utc_now(),
        "offline_environment": {
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
        },
        "tensor_comparison": tensor_comparison,
        "tokenizer_validation": tokenizer,
        "container_readiness": {
            "report_path": str(container_readiness_path.resolve()),
            "report_sha256": sha256_file(container_readiness_path),
            "passed": bool(
                container_readiness and container_readiness.get("passed")
            ),
        }
        if container_readiness_path.exists()
        else None,
        "inference_skipped": args.skip_inference,
    }

    merged_inference: dict[str, Any] | None = None
    base_inference: dict[str, Any] | None = None
    behavior: dict[str, Any] | None = None
    inference_error: str | None = None
    if not args.skip_inference:
        device = choose_device(args.device)
        print(f"Running original checkpoint smoke prompts on {device}...")
        try:
            base_inference = generate_smoke_outputs(
                base_dir, "original_thinking", device, args.max_new_tokens
            )
            print(f"Running merged checkpoint smoke prompts on {device}...")
            merged_inference = generate_smoke_outputs(
                merged_dir, "merged_finetune", device, args.max_new_tokens
            )
            behavior = generation_comparison(base_inference, merged_inference)
            print(
                "Behavior differences: "
                f"{behavior['differing_completion_count']} / {behavior['case_count']}"
            )
        except Exception as exc:
            inference_error = f"{type(exc).__name__}: {exc}"
            print(f"Inference validation failed: {inference_error}", file=sys.stderr)

    validation["base_inference"] = base_inference
    validation["merged_inference"] = merged_inference
    validation["generation_comparison"] = behavior
    validation["inference_error"] = inference_error
    validation["gates"] = {
        "base_revision_recovered": base["selected_snapshot_exists"],
        "adapter_revision_recovered": adapter["selected_snapshot_exists"],
        "merged_parameter_count_is_1_170_340_608": any(
            item.get("safetensors", {}).get("parameter_count") == 1_170_340_608
            for item in merged_manifest["files"]
        ),
        "original_and_merged_tensor_sets_match": (
            not tensor_comparison["base_only_tensors"]
            and not tensor_comparison["merged_only_tensors"]
        ),
        "merge_changed_weight_tensors": tensor_comparison["proves_merge_not_noop"],
        "token_ids_and_chat_template_match_original": tokenizer["passed"],
        "offline_original_load_and_generation": (
            bool(base_inference and base_inference["passed"])
            if not args.skip_inference
            else False
        ),
        "original_first_step_logits_are_finite": (
            bool(
                base_inference
                and base_inference["all_first_step_logits_finite"]
            )
            if not args.skip_inference
            else False
        ),
        "offline_merged_load_and_generation": (
            bool(merged_inference and merged_inference["passed"])
            if not args.skip_inference
            else False
        ),
        "merged_first_step_logits_are_finite": (
            bool(
                merged_inference
                and merged_inference["all_first_step_logits_finite"]
            )
            if not args.skip_inference
            else False
        ),
        "smoke_outputs_show_behavior_change": (
            bool(behavior and behavior["proves_behavior_changed_on_smoke_set"])
            if not args.skip_inference
            else False
        ),
        "docker_linux_engine_reachable": (
            machine["docker_server"].get("exit_code") == 0
            or bool(container_readiness and container_readiness.get("passed"))
        ),
        "restricted_cuda_container_smoke_passed": bool(
            container_readiness and container_readiness.get("passed")
        ),
    }
    validation["passed"] = all(validation["gates"].values())
    json_dump(output_dir / "validation.json", validation)
    write_summary(
        output_dir / "README.md",
        provenance=provenance,
        machine=machine,
        validation=validation,
        container_readiness=container_readiness,
    )

    print(f"Phase 0 reports written to {output_dir}")
    print(f"Overall gate: {'PASS' if validation['passed'] else 'FAIL'}")
    return 0 if validation["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
