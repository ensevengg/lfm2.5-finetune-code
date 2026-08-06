# LFM2.5 Fine‑Tuning & Evaluation

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue)
[![HF Dataset](https://img.shields.io/badge/%F0%9F%A4%97%20Dataset-enseven%2Fkodcode--lfm2.5-blue)](https://huggingface.co/datasets/enseven/kodcode-lfm2.5)

## Table of Contents
- [Overview](#overview)
- [Datasets](#datasets)
- [Preprocessing](#preprocessing)
- [Training](#training)
- [Evaluation](#evaluation)
- [Requirements](#requirements)
- [Quick Start](#quick-start)
- [License](#license)

## Overview

LoRA fine‑tune [LFM2.5-1.2B-Thinking](https://huggingface.co/LiquidAI/LFM2.5-1.2B-Thinking) — a 1.17B hybrid ShortConv + GQA language model — on [KodCode-V1-SFT-R1](https://huggingface.co/datasets/KodCode/KodCode-V1-SFT-R1) for general coding instruction‑following. The training mix is **70% code-only** (question → `r1_solution`) and **30% chain-of-thought** (question → answer with `<think>` traces).

**Pipeline**: Raw KodCode → `preprocess_kodcode.py` → formatted parquet → `train_lfm25_lora.py` → LoRA adapter on HuggingFace Hub.

## Datasets

- **Source**: [KodCode-V1-SFT-R1](https://huggingface.co/datasets/KodCode/KodCode-V1-SFT-R1) — synthetic coding Q&A from Leetcode, Codeforces, Apps, Taco, Code_Contests, Evol, Package, Algorithm, Data_Structure, Docs.
- **Derived**: [`enseven/kodcode-lfm2.5`](https://huggingface.co/datasets/enseven/kodcode-lfm2.5) — formatted for LFM2.5's chat template.
- **Size**: 268,211 rows, 442 MB parquet (~1.09B chars).
- **Mix**: 187,824 code-only (70%) + 80,387 CoT (30%).
- **License**: CC BY-NC 4.0 (inherited from KodCode-V1-SFT-R1).

## Preprocessing

| Script | Purpose |
|---|---|
| `preprocess_kodcode.py` | Load 11 train shards, build chat strings, assign 70/30 format, save as parquet. |
| `prepare_hf_dataset.py` | Convert parquet → HF Dataset (arrow) format; optionally push to Hub. |

**Chat template**: `<|startoftext|><|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n{answer}<|im_end|>\n`

**Key details**:
- Char-based CoT pre‑filter at ≤15,000 chars (avoids slow per‑row tokenization — 50–55% of CoT texts fit in 4096 tokens).
- Per‑shard controlled random assignment with `np.random.default_rng(seed=42)`.
- Output: `kodcode_processed/kodcode-lfm2.5.parquet` and `kodcode_processed/hf_dataset/`.

## Training

Two scripts targeting different GPU environments:

### `train_lfm25_lora.py` — Lightning.ai L40S Studio

- 48 GB VRAM, **bf16** precision, full LoRA.
- `max_seq_length=4096`, effective batch 16 (4 per device × 4 grad accum).
- Pushes adapter to Hub every `save_steps`.

```bash
# Fresh run with Hub push
python train_lfm25_lora.py

# Resume from latest checkpoint
python train_lfm25_lora.py --resume

# Load dataset from Hub (skip local preprocessing)
python train_lfm25_lora.py --dataset enseven/kodcode-lfm2.5

# Skip Hub push
python train_lfm25_lora.py --no-hub
```

### `train_lfm25_kaggle.py` — Kaggle T4

- 15 GB VRAM, **QLoRA (4-bit NF4 + fp16)** for T4 compatibility.
- `max_seq_length=2048`, effective batch 8.
- Supports cross‑platform resume via `--resume-from-hub`.

```bash
# Fresh run
python train_lfm25_kaggle.py

# Resume adapter from Hub (e.g., after Lightning run)
python train_lfm25_kaggle.py --resume-from-hub enseven/lfm-2.5-think-code
```

### Shared configuration

| Parameter | Value |
|---|---|
| LoRA rank / alpha / dropout | `r=32`, `alpha=64`, `dropout=0.1` |
| Target modules | `in_proj`, `out_proj`, `q/k/v/o_proj`, `w1/w2/w3` |
| Learning rate | 1e‑4 (cosine, warmup 0.03) |
| Epochs | 1 |
| Gradient clipping | 1.0 |
| Packing | Enabled (SFTTrainer packs short examples) |
| Gradient checkpointing | Enabled (`use_reentrant=False`) |
| Tracking | W&B (`lfm25-kodcode` project, optional) |
| Hub output | `enseven/lfm-2.5-think-code` |

## Evaluation

The training scripts hold out 1% of data (`--eval-split 0.01`) for periodic eval during training (`eval_strategy="steps"`). No standalone evaluation script exists yet.

### Phase 0 reproducibility gate

The pre-quantization gate hashes the merged, original, and adapter artifacts;
recovers their local Hugging Face revisions; inventories the machine; compares
original and merged tensors; and runs offline deterministic smoke prompts:

```bash
python scripts/phase0_freeze_and_validate.py --device cuda
```

With Docker Desktop's Linux engine running, validate GPU passthrough and the
restricted container boundary:

```powershell
.\scripts\run_phase0_container_smoke.ps1
```

Evidence is written to [`reports/phase0/`](reports/phase0/). The script never
edits either checkpoint and forces Hugging Face/Transformers offline mode for
tokenizer and model loading.

### Phase 1 pinned GGUF/CUDA toolchain

Phase 1 builds `llama.cpp` commit
`3018a11e79e489b657dbb77c95694889ccff92df` for CUDA compute capability 12.0,
then creates and validates a BF16 GGUF reference:

```powershell
.\scripts\build_phase1_toolchain.ps1
.\scripts\convert_phase1_bf16.ps1
.\scripts\inspect_phase1_gguf.ps1
.\scripts\run_phase1_gguf_validation.ps1
```

The build and run containers use digest-pinned CUDA 12.8.1 images. Conversion
and validation run with no network, a read-only root filesystem, a non-root
user, dropped capabilities, explicit CPU/RAM/process limits, read-only model
mounts, and narrowly scoped writable artifact or report mounts.

The BF16 file is the conversion-parity reference for the later quantization
sweep. Generated model files under `artifacts/` and the
vendored source checkout under `vendor/` are intentionally excluded from Git;
their immutable revisions and SHA-256 hashes are recorded in
[`reports/phase1/`](reports/phase1/).

### Phase 2 quantized GGUF artifacts

Phase 2 independently quantizes the verified BF16 GGUF to Q8_0, Q6_K,
Q5_K_M, Q4_K_M, and Q2_K, then performs static GGUF integrity checks:

```powershell
.\scripts\quantize_phase2.ps1
.\scripts\inspect_phase2_quantizations.ps1
```

The quantizer runs offline in the pinned, restricted container. The BF16 file
is mounted read-only and only `artifacts/gguf/phase2/` is writable. Evidence,
hashes, exact sizes, and tensor inventories are in
[`reports/phase2/`](reports/phase2/). These artifacts have not yet been
benchmarked, long-context-qualified, or ranked; Q2_K is included as an
ultra-low-memory option, not a quality-equivalent recommendation.

## Requirements

- **Python** 3.12+
- **Dependencies** (from `requirements.txt`):

```
torch>=2.4.0
transformers>=4.44.0
peft>=0.12.0
trl>=0.10.0
datasets>=2.20.0
accelerate>=0.33.0
bitsandbytes>=0.43.0
wandb>=0.17.0
sentencepiece>=0.2.0
```

## Quick Start

1. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Preprocess data** (or skip and load from Hub directly)
   ```bash
   python preprocess_kodcode.py
   python prepare_hf_dataset.py --push
   ```

3. **Train on Lightning L40S**
   ```bash
   python train_lfm25_lora.py --wandb-project lfm25-kodcode
   ```

4. **Kaggle alternative** — copy `train_lfm25_kaggle.py` into a Kaggle notebook (T4 GPU) and run:
   ```bash
   pip install --upgrade transformers bitsandbytes trl peft datasets accelerate sentencepiece wandb
   python train_lfm25_kaggle.py --resume-from-hub enseven/lfm-2.5-think-code
   ```

## License

MIT License. See [LICENSE](LICENSE).
