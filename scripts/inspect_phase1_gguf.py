#!/usr/bin/env python3
"""Inspect the Phase 1 BF16 GGUF and enforce structural conversion gates."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from gguf import GGUFReader


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


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
    if field is None:
        return None
    return json_value(field.contents())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gguf", type=Path, required=True)
    parser.add_argument("--source-config", type=Path, required=True)
    parser.add_argument("--source-chat-template", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-tensors", type=int, default=148)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source_config = json.loads(args.source_config.read_text(encoding="utf-8"))
    source_template = args.source_chat_template.read_text(encoding="utf-8")
    reader = GGUFReader(args.gguf, "r")

    selected_keys = [
        "general.architecture",
        "general.name",
        "general.file_type",
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
    metadata = {key: field_value(reader, key) for key in selected_keys}
    tensor_types = Counter(tensor.tensor_type.name for tensor in reader.tensors)
    tensor_count = len(reader.tensors)
    gguf_template = metadata["tokenizer.chat_template"]

    gates = {
        "architecture_is_lfm2": metadata["general.architecture"] == "lfm2",
        "file_type_is_mostly_bf16": metadata["general.file_type"] == 32,
        "context_length_matches_source": (
            metadata["lfm2.context_length"]
            == source_config["max_position_embeddings"]
        ),
        "block_count_matches_source": (
            metadata["lfm2.block_count"] == source_config["num_hidden_layers"]
        ),
        "embedding_length_matches_source": (
            metadata["lfm2.embedding_length"] == source_config["hidden_size"]
        ),
        "attention_head_count_matches_source": (
            metadata["lfm2.attention.head_count"]
            == source_config["num_attention_heads"]
        ),
        "special_token_ids_match_source": (
            metadata["tokenizer.ggml.bos_token_id"]
            == source_config["bos_token_id"]
            and metadata["tokenizer.ggml.eos_token_id"]
            == source_config["eos_token_id"]
            and metadata["tokenizer.ggml.padding_token_id"]
            == source_config["pad_token_id"]
        ),
        "chat_template_matches_source": gguf_template == source_template,
        "tensor_count_matches_source": tensor_count == args.expected_tensors,
        "all_weight_tensors_are_bf16_or_f32": set(tensor_types) <= {"BF16", "F32"},
    }
    report = {
        "schema_version": 1,
        "generated_at_utc": utc_now(),
        "passed": all(gates.values()),
        "gguf": str(args.gguf),
        "source_config": str(args.source_config),
        "gates": gates,
        "metadata": metadata,
        "tensor_count": tensor_count,
        "tensor_types": dict(sorted(tensor_types.items())),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"gates": gates, "tensor_types": tensor_types}, default=dict))
    print(f"Phase 1 GGUF metadata: {'PASS' if report['passed'] else 'FAIL'}")
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
