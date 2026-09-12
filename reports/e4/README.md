# Phase E4 — Sealed benchmark report (Step 5 complete)

Status: **complete; Q6_K passes the frozen deployment gate**  
Suite: `e4-humanevalplus-sealed-v1` (128 sealed tasks; dev IDs 0/17/119 and all 33 pilot IDs burned)  
Machine-readable report: [`e4-report.json`](e4-report.json)  
Date: 2026-09-12

## What ran

Four model rows × 128 sealed tasks, one greedy sample per task — the same
frozen prompt (`lfm-code-v1`), runtime, retry policy, per-task extraction
policy, and restricted scoring container as the pilot. Zero infrastructure
failures, zero retries, zero duplicate keys, zero host-side program
executions. No sealed task was ever touched before this run.

| Row | pass@1 plus (scored) | plus rate (absence = 0) | extraction yield | decode tok/s | out tokens |
|---|---:|---:|---:|---:|---:|
| original-bf16 | 0.5545 (56/101) | **0.4375** | 0.789 | 101 | 211,732 |
| merged-bf16 | 0.4048 (51/126) | **0.3984** | 0.984 | 114 | 18,992 |
| q8_0 | 0.4173 (53/127) | 0.4141 | 0.992 | 182 | 16,425 |
| q6_k | 0.3889 (49/126) | 0.3828 | 0.984 | 189 | 19,346 |

`plus rate (absence = 0)` counts every unextractable output as a deployed
failure; the scorer's pass@1 silently excludes them from the denominator.
Both are reported; the paired analysis below is the primary comparison.

## Findings

1. **Fine-tune effect: negative, but smaller than the pilot suggested.** On
   the 99 tasks scored in both rows, the fine-tune passes 49 versus the
   original's 56 — 16 gained, 23 lost, net −7 (−7.1 pp paired). Under
   absence-as-zero the gap is 0.4375 vs 0.3984 (−3.9 pp). The pilot's much
   larger gap (−44 pp scorer) was partly slice noise; the direction is
   confirmed at sealed scale.
2. **The mechanism is the one suspected after the pilot.** The original model
   emitted 211,732 completion tokens (≈1,654/task) and its reasoning traces
   were doing real work; the fine-tune emits ≈148 tokens/task — clean,
   fast, near-perfectly extractable, and less correct. Extraction yield and
   decode throughput improved precisely because the fine-tune discards the
   reasoning tokens.
3. **Quantization is within noise of merged BF16.** Q8_0: net +1 paired task.
   Q6_K: net −2 paired tasks (−1.59 pp), all scattered single-task flips
   near token ties (HumanEval/105 is lost by both quants — shared
   quantization sensitivity, not a row-specific failure class).

## Deployment gate verdict (frozen before the run)

**Q6_K: PASS.** −1.59 pp versus merged BF16 (gate: within 2 pp) with no new
paired failure class (gate: losses are scattered single tasks with one gain).
Selected at 963 MB versus Q8_0's 1,246 MB; Q8_0 is quality-equivalent-or-
better and remains the runner-up when file size does not matter.

## Caveats

- One greedy sample per task; no temperature-0.6 seed variance anywhere in
  Step 5 (deterministic regression track only).
- KodCode contamination caveat applies to absolute scores; paired deltas are
  the scientific comparison.
- 32K remains the maximum context-support claim; nothing in Step 5 measured
  beyond a 4,096-token context.

## Step 5 outcome

- Deployment model: **Q6_K** (`artifacts/gguf/phase2/lfm2.5-1.2b-thinking-kodcode-q6_k.gguf`).
- The KodCode LoRA fine-tune is **not recommended** for deployment on
  HumanEval-style code generation: it trades the base model's functional
  reasoning for clean formatting, costing 4–7 pp paired pass@1 while
  improving extraction yield and speed.
- All evidence: per-row `generations.jsonl`, `samples.jsonl`,
  `extraction-report.json`, and `scoring/e3-score-*/` directories under the
  session directories linked from [`e4-report.json`](e4-report.json).
