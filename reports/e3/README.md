# Phase E3 — Pilot report (Step 5)

Status: **complete; E4 sealed gates frozen**  
Suite: `e3-humanevalplus-pilot-v1` (33 tasks, permanently burned after this pilot)  
Machine-readable report: [`pilot-report.json`](pilot-report.json)  
Date: 2026-09-12

## What ran

Four model rows × 33 pilot tasks, one greedy sample per task, all generation
inside the pinned llama.cpp container and all scoring inside the pinned
EvalPlus container. Zero infrastructure failures, zero retries, zero duplicate
keys, zero host-side program executions.

| Row | pass@1 base | pass@1 plus | extraction yield | decode tok/s | out tokens |
|---|---:|---:|---:|---:|---:|
| original-bf16 | 0.750 | 0.714 | 0.848 | 112.3 | 52,186 |
| merged-bf16 | 0.364 | 0.273 | 1.000 | 122.1 | 3,578 |
| q8_0 | 0.364 | 0.273 | 1.000 | 197.2 | 3,582 |
| q6_k | 0.364 | 0.303 | 1.000 | 201.8 | 3,541 |

## Findings

1. **Fine-tune effect is negative on this slice.** The merged model loses
   twelve paired plus-status tasks and gains none versus the original
   (`paired_deltas` in the JSON report). Even crediting the original model
   with zero on its five unextractable outputs, its floor is 0.606 versus the
   merged 0.273. The fine-tune also collapsed output length (52,186 → 3,578
   total completion tokens): prompt v1 largely suppresses the base model's
   reasoning traces, and correctness fell with them. Absolute scores carry
   the KodCode-contamination caveat, but the paired delta is the scientific
   comparison and it is clearly negative.
2. **Quantization is essentially lossless at pilot scale.** Q8_0 is identical
   to merged BF16 on both metrics; Q6_K loses no paired task and passes one
   extra plus task, at 963 MB versus 1,246 MB. Both roughly double decode
   throughput (122 → ~200 tok/s).
3. **Harness health gates are met.** Infrastructure failure rate 0% (gate
   ≤ 2%). Extraction yield is reported per row as model evidence, not a
   harness gate: the original row's 0.848 is a property of the base thinking
   model's reasoning leakage under prompt v1, with five tasks absent from
   scoring by the frozen per-task failure policy.

## Frozen E4 decisions (before any sealed request)

- Sealed rows: original-bf16, merged-bf16, q8_0, q6_k over the 128-task
  sealed slice (`e4-humanevalplus-sealed-v1`).
- Deployment quant: **Q6_K**, subject to the predeclared sealed gate — within
  2 percentage points of merged-bf16 pass@1(plus) and no paired loss pattern
  absent from merged.
- Harness gates: infra failure rate ≤ 2% of requests; extraction yield
  reported, not gated.

## Caveats

- n = 33, one greedy sample per task; the sealed run firms up the deltas.
- One greedy sample per task means no seed variance is measured anywhere in
  Step 5; generalization to temperature-0.6 sampling is not claimed.
- All execution of generated code happened inside the restricted E2-style
  container (no network, non-root, read-only root, CPU-only).

## Evidence layout

Per row: `generations.jsonl`, `samples.jsonl`, `extraction-report.json`,
`scoring/e3-score-*/` (preparation report, docker command, container log,
eval results, run manifest). Session summaries:
`e3-pilot-20260912T13*/pilot-generation-summary.json`.
