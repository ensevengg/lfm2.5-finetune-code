#!/usr/bin/env python3
"""Validate Phase 2 GGUF structure without running quality benchmarks."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from gguf import GGUFReader


EXPECTED = {
    "Q8_0": ("lfm2.5-1.2b-thinking-kodcode-q8_0.gguf", 7),
    "Q6_K": ("lfm2.5-1.2b-thinking-kodcode-q6_k.gguf", 18),
    "Q5_K_M": ("lfm2.5-1.2b-thinking-kodcode-q5_k_m.gguf", 17),
    "Q4_K_M": ("lfm2.5-1.2b-thinking-kodcode-q4_k_m.gguf", 15),
    "Q2_K": ("lfm2.5-1.2b-thinking-kodcode-q2_k.gguf", 10),
}

PRESERVED_KEYS = [
    "general.architecture",
    "lfm2.context_length",
    "lfm2.embedding_length",
    "lfm2.block_count",
    "lfm2.attention.head_count",
    "lfm2.attention.head_count_kv",
    "lfm2.rope.freq_base",
    "tokenizer.ggml.model",
    "tokenizer.ggml.bos_token_id",
    "tokenizer.ggml.eos_token_id",
    "tokenizer.ggml.padding_token_id",
    "tokenizer.chat_template",
]


def json_value(value: Any) -> Any:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, list):
        return [json_value(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def field_value(reader: GGUFReader, key: str) -> Any:
    field = reader.get_field(key)
    return None if field is None else json_value(field.contents())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--quant-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-tensors", type=int, default=148)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source_reader = GGUFReader(args.source, "r")
    source_metadata = {key: field_value(source_reader, key) for key in PRESERVED_KEYS}
    source_tensor_count = len(source_reader.tensors)
    results: list[dict[str, Any]] = []

    for quantization, (file_name, expected_file_type) in EXPECTED.items():
        path = args.quant_dir / file_name
        reader = GGUFReader(path, "r")
        metadata = {key: field_value(reader, key) for key in PRESERVED_KEYS}
        file_type = field_value(reader, "general.file_type")
        tensor_types = Counter(tensor.tensor_type.name for tensor in reader.tensors)
        tensor_count = len(reader.tensors)
        gates = {
            "file_type_matches_requested_quantization": file_type == expected_file_type,
            "metadata_matches_bf16_source": metadata == source_metadata,
            "tensor_count_matches_bf16_source": tensor_count == source_tensor_count,
            "tensor_count_matches_expected": tensor_count == args.expected_tensors,
            "contains_quantized_tensors": bool(set(tensor_types) - {"BF16", "F32"}),
            "nonempty_file": path.stat().st_size > 0,
        }
        results.append(
            {
                "quantization": quantization,
                "file_name": file_name,
                "size_bytes": path.stat().st_size,
                "general_file_type": file_type,
                "tensor_count": tensor_count,
                "tensor_types": dict(sorted(tensor_types.items())),
                "gates": gates,
                "passed": all(gates.values()),
            }
        )

    report = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "static GGUF integrity only; no model evaluation performed",
        "source": str(args.source),
        "source_tensor_count": source_tensor_count,
        "preserved_metadata": source_metadata,
        "artifacts": results,
        "passed": all(item["passed"] for item in results),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    for item in results:
        print(
            f"{item['quantization']}: {'PASS' if item['passed'] else 'FAIL'} "
            f"({item['size_bytes']} bytes; {item['tensor_types']})"
        )
    print(f"Phase 2 static GGUF inspection: {'PASS' if report['passed'] else 'FAIL'}")
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

