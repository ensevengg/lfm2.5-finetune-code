---
language:
- en
license: cc-by-nc-4.0
size_categories:
- 100K<n<1M
dataset_info:
  features:
  - name: text
    dtype: string
  - name: format
    dtype: string
  - name: subset
    dtype: string
  - name: question_id
    dtype: string
  splits:
  - name: train
    num_examples: 268211
  download_size: 463470592
  dataset_size: 1085161053
task_categories:
- question-answering
tags:
- code
- synthetic
- sft
pretty_name: KodCode LFM2.5
---

# KodCode LFM2.5

A preprocessed version of [KodCode-V1-SFT-R1](https://huggingface.co/datasets/KodCode/KodCode-V1-SFT-R1)
formatted for fine-tuning [LFM2.5-1.2B-Thinking](https://huggingface.co/LiquidAI/LFM2.5-1.2B-Thinking)
with a 70% code-only / 30% CoT (chain-of-thought) mix.

## Dataset Description

- **Homepage:** [KodCode Project](https://kodcode-ai.github.io/)
- **Original paper:** [KodCode: A Diverse, Challenging, and Verifiable Synthetic Dataset for Coding](https://arxiv.org/abs/2503.02951)
- **Original dataset:** [KodCode/KodCode-V1-SFT-R1](https://huggingface.co/datasets/KodCode/KodCode-V1-SFT-R1)
- **Points of contact:** [Zhangchen Xu](mailto:zxu9@uw.edu) (original authors)

### Dataset Summary

This dataset is derived from KodCode-V1-SFT-R1 (CC BY-NC 4.0). For each of the 268,211 training
rows, the question-answer pair is formatted into a flat chat template string using
`<|im_start|>` / `<|im_end|>` markers and a `<|startoftext|>` prefix — the native format for
LFM2.5-1.2B-Thinking.

Each row is assigned to one of two formats:

- **code-only (70%):** `question` → `r1_solution` (code response, no thinking trace)
- **CoT (30%):** `question` → `conversations[-1]["value"]` (full response with `<think>` reasoning)

The CoT selection is limited to texts ≤ 15,000 characters (guaranteed to fit within 4096 tokens
at the observed minimum char-to-token ratio of 2.70).

### Changes from the Original

1. **Format conversion:** Original `conversations` field (list of `from`/`value` dicts) is flattened
   into a single `text` string with `<|im_start|>` / `<|im_end|>` chat template.
2. **Format selection:** Each row is assigned either `code-only` or `cot` format at a 70/30 ratio
   (controlled random per shard).
3. **Char pre-filtering:** CoT texts exceeding 15,000 characters fall back to code-only format
   (instead of being dropped), since only ~50% of CoT texts fit within 4096 tokens.
4. **Reduced columns:** Only 4 columns are kept: `text`, `format`, `subset`, `question_id`.

## Data Fields

| Field | Type | Description |
|---|---|---|
| `text` | `string` | Flattened chat text: `<\|startoftext\|><\|im_start\|>user\n{question}<\|im_end\|>\n<\|im_start\|>assistant\n{answer}<\|im_end\|>\n` |
| `format` | `string` | `"code-only"` or `"cot"` |
| `subset` | `string` | Original KodCode subset (e.g. `"Leetcode"`, `"Codeforces"`, `"Taco"`, etc.) |
| `question_id` | `string` | Original question identifier from KodCode |

### Column mapping to original dataset

| This dataset | KodCode-V1-SFT-R1 |
|---|---|
| `text` (code-only) | `question` + `r1_solution` formatted with chat template |
| `text` (CoT) | `question` + `conversations[-1]["value"]` formatted with chat template |
| `subset` | `subset` |
| `question_id` | `question_id` |
| *(omitted)* | `solution`, `test`, `test_info`, `version`, `style`, `metadata`, `r1_pass_sequence`, `r1_correctness`, `gpt_pass_sequence`, `gpt_difficulty`, `gpt_pass_percentage`, `conversations` |

## Data Splits

| Split | Size |
|---|---|
| `train` | 268,211 rows |

## Usage

### Load with HuggingFace Datasets

```python
from datasets import load_dataset

ds = load_dataset("kodcode_dataset", split="train")
# or from parquet directly:
ds = Dataset.from_parquet("kodcode_dataset/data/kodcode-lfm2.5.parquet")
```

### Load for SFT training (TRL)

```python
from trl import SFTTrainer

trainer = SFTTrainer(
    ...,
    train_dataset=ds,
    dataset_text_field="text",
    max_seq_length=4096,
    packing=True,
)
```

### Subset distribution

```python
print(ds.to_pandas()["subset"].value_counts())
```

### Format distribution

```python
print(ds.to_pandas()["format"].value_counts())
# code-only    187824
# cot           80387
```

## Statistics

| Metric | Value |
|---|---|
| Total rows | 268,211 |
| Code-only rows | 187,824 (70.0%) |
| CoT rows | 80,387 (30.0%) |
| Total characters | ~1.09B |
| Text length (mean) | 4,046 chars |
| Text length (median) | 2,130 chars |
| Text length (max) | 15,000 chars |
| Output size (parquet) | 442 MB |

## Citation

If you use this dataset, please cite the original KodCode work:

```bibtex
@article{xu2025kodcode,
      title={KodCode: A Diverse, Challenging, and Verifiable Synthetic Dataset for Coding},
      author={Zhangchen Xu and Yang Liu and Yueqin Yin and Mingyuan Zhou and Radha Poovendran},
      year={2025},
      eprint={2503.02951},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2503.02951},
}
```

## License

This dataset is derived from [KodCode-V1-SFT-R1](https://huggingface.co/datasets/KodCode/KodCode-V1-SFT-R1)
and is distributed under the same **CC BY-NC 4.0** license.

- **Attribution:** You must give appropriate credit to the original KodCode authors.
- **NonCommercial:** You may not use the material for commercial purposes.
- **No additional restrictions:** You may not apply legal terms that restrict others from doing
  anything the license permits.

See the [full license text](https://creativecommons.org/licenses/by-nc/4.0/) for details.
