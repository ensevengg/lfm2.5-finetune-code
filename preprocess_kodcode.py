"""
Phase 1: Preprocess KodCode-V1-SFT-R1 for LFM2.5-1.2B-Thinking fine-tuning.

Strategy: store formatted text strings (not pre-tokenized IDs).
The training script (SFTTrainer with dataset_text_field="text") handles tokenization
on-the-fly, which is both faster and more memory-efficient.

1. Load 11 train shards
2. Build formatted chat strings for code-only and CoT variants
3. Use character-length pre-filter to determine CoT fit (avoids slow tokenization)
4. Assign 30% CoT / 70% code-only at random
5. Save as parquet
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ── Config ──────────────────────────────────────────────────────────────
MODEL_NAME = "LiquidAI/LFM2.5-1.2B-Thinking"
DATA_DIR = Path(
    r"D:\hf_models\datasets--KodCode--KodCode-V1-SFT-R1"
    r"\snapshots\26c8a11c800d71b6c4bafe12a92c9090ef0b6214\data"
)
OUTPUT_REPO = "enseven/kodcode-lfm2.5"
OUTPUT_DIR = Path("./kodcode_processed")
COT_RATIO = 0.30
SEED = 42
HF_TOKEN = os.environ.get("HF_TOKEN", None)

# Char threshold: texts <= this length are guaranteed to fit in 4096 tok
# At min chars/tok ratio 2.70, 15000 chars = 5555 tok — still > 4096.
# But at P5 ratio 2.99, 15000/2.99 = 5016 tok — also > 4096.
# At mean ratio 3.53, 15000/3.53 = 4249 tok — some > 4096.
# At P75 ratio ~3.7, 15000/3.7 = 4054 tok — fits.
# So about 50-55% of CoT texts at 15000 char limit will fit in 4096 tok.
# We accept false positives (will be truncated during training).
COT_CHAR_LIMIT = 15000

# Chat template strings (no tokenization needed)
BOS = "<|startoftext|>"
IM_START = "<|im_start|>"
IM_END = "<|im_end|>"


def format_chat(question: str, answer: str) -> str:
    return f"{BOS}{IM_START}user\n{question}{IM_END}\n{IM_START}assistant\n{answer}{IM_END}\n"


# ── Processing ──────────────────────────────────────────────────────────
def process_shard(filepath: Path) -> pd.DataFrame:
    print(f"  Loading {filepath.name}", flush=True)
    df = pd.read_parquet(filepath)
    n = len(df)
    print(f"    {n} rows", flush=True)

    questions = df["question"].astype(str).tolist()
    r1_solutions = df["r1_solution"].astype(str).tolist()
    conversations = df["conversations"].tolist()

    # Build formatted text strings for both variants
    code_texts = [format_chat(q, s) for q, s in zip(questions, r1_solutions)]

    cot_texts = [None] * n
    cot_fits = np.zeros(n, dtype=bool)
    for i in range(n):
        conv = conversations[i] if i < len(conversations) else []
        if isinstance(conv, np.ndarray):
            conv = conv.tolist()
        cot_answer = str(conv[-1]["value"]) if len(conv) >= 2 else ""
        cot_text = format_chat(questions[i], cot_answer)
        cot_texts[i] = cot_text
        cot_fits[i] = len(cot_text) <= COT_CHAR_LIMIT

    print(
        f"    CoT fits (char estimate): {cot_fits.sum()}/{n} "
        f"({100 * cot_fits.sum() / n:.1f}%)",
        flush=True,
    )

    return pd.DataFrame(
        {
            "code_text": code_texts,
            "cot_text": cot_texts,
            "cot_fits": cot_fits,
            "subset": df["subset"].values,
            "question_id": df["question_id"].values,
        }
    )


def assign_formats(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    n = len(df)
    cot_candidates = df.index[df["cot_fits"]].tolist()
    n_cot = int(round(COT_RATIO * n))
    n_cot = min(n_cot, len(cot_candidates))
    selected_cot = set(rng.choice(cot_candidates, size=n_cot, replace=False))

    texts = []
    formats = []
    subsets = []
    question_ids = []

    for i in range(n):
        if i in selected_cot:
            texts.append(df.loc[i, "cot_text"])
            formats.append("cot")
        else:
            texts.append(df.loc[i, "code_text"])
            formats.append("code-only")
        subsets.append(df.loc[i, "subset"])
        question_ids.append(df.loc[i, "question_id"])

    return pd.DataFrame(
        {
            "text": texts,
            "format": formats,
            "subset": subsets,
            "question_id": question_ids,
        }
    )


# ── Main ────────────────────────────────────────────────────────────────
def main():
    rng = np.random.default_rng(SEED)

    shard_paths = sorted(DATA_DIR.glob("train-*-of-00011.parquet"))
    print(f"Found {len(shard_paths)} train shards in {DATA_DIR}", flush=True)

    all_shards = []
    for sp in shard_paths:
        processed = process_shard(sp)
        assigned = assign_formats(processed, rng)
        all_shards.append(assigned)

        fmt_counts = assigned["format"].value_counts()
        print(
            f"    {len(assigned)} rows: "
            f"{fmt_counts.get('code-only', 0)} code-only, "
            f"{fmt_counts.get('cot', 0)} CoT",
            flush=True,
        )

    full_df = pd.concat(all_shards, ignore_index=True)
    total = len(full_df)
    n_cot = (full_df["format"] == "cot").sum()
    n_code = total - n_cot
    print(
        f"\nTotal: {total} rows — "
        f"{n_code} code-only ({100 * n_code / total:.1f}%), "
        f"{n_cot} CoT ({100 * n_cot / total:.1f}%)",
        flush=True,
    )

    text_lens = full_df["text"].apply(len)
    total_chars = text_lens.sum()
    print(f"Total chars: {total_chars} (~{total_chars // 3} estimated tokens)", flush=True)
    print(
        f"  mean: {text_lens.mean():.0f}, "
        f"median: {text_lens.median():.0f}, "
        f"max: {text_lens.max()}",
        flush=True,
    )

    # Save
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    local_path = OUTPUT_DIR / "kodcode-lfm2.5.parquet"
    print(f"\nSaving to {local_path}", flush=True)
    full_df.to_parquet(local_path, index=False)
    size_mb = os.path.getsize(local_path) / 1024**2
    print(f"  {size_mb:.0f} MB", flush=True)

    print(f"\nTo push to Hub, run:", flush=True)
    print(f"  from datasets import Dataset, load_dataset", flush=True)
    print(f"  ds = Dataset.from_parquet('{local_path}')", flush=True)
    print(f"  ds.push_to_hub('{OUTPUT_REPO}', private=True)", flush=True)

    print("\nDone!", flush=True)


if __name__ == "__main__":
    main()
