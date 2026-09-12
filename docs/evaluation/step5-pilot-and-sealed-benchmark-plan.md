# Step 5 Plan: Baseline, Pilot, and Sealed Benchmark (E3/E4)

Status: **complete 2026-09-12 — pilot and sealed benchmark executed**
([pilot](../../reports/e3/README.md) ·
[sealed](../../reports/e4/README.md) ·
[e4-report.json](../../reports/e4/e4-report.json)).
Outcome: fine-tune regression confirmed at sealed scale (−7.1 pp paired);
deployment quant is Q6_K (gate passed at −1.59 pp, no new failure class).
Created: 2026-09-12
Extends: [E0 contract](e0-contract.md) (E0–E2 complete)
Inputs: E2 development scoring (`reports/e2/e2-score-20260912T121947967Z/`), host
decision ([CachyOS vs WSL2](../research/cachyos-vs-wsl2-evaluation-host.md)),
quantization scope ([research note](../research/lfm25-quantization-offline-evaluation.md))

## Purpose

Step 5 answers three questions with one controlled comparison, and separates
them so no result is attributed to the wrong cause:

1. **Fine-tune effect** — did the KodCode LoRA change HumanEval+ pass@1 versus
   the original `LiquidAI/LFM2.5-1.2B-Thinking`, served by the identical
   llama.cpp runtime?
2. **Quantization effect** — how much pass@1 does each weight quant give up
   against the merged BF16 GGUF, and does any smaller quant meet the
   predeclared deployment gate?
3. **Deployment evidence** — per-row artifact size, load time, prefill/decode
   throughput, and peak VRAM, measured under one frozen protocol.

## Scope and explicit deferrals

This is the **coding track only**. Deferred to later phases, not silently
dropped: the published-comparability general suite (GPQA Diamond, MMLU-Pro,
IFEval, GSM8K, MATH-500, AIME25 at temperature 0.6 — requires validating the
pinned lm-evaluation-harness revision first), 64K/128K long-context
qualification, EvalPerf (Unix-only), and any LM Studio deployment check.

## Decisions already frozen by prior phases

| Decision | Value | Source |
|---|---|---|
| Host | Windows 11 + Docker Desktop WSL2; one host for all quality comparisons | host research doc |
| Runtime | `lfm25/llama-cpp:3018a11e-cuda128-sm120`, one GPU, all layers offload | phase 1 lock |
| Scoring | EvalPlus `26d6d00b`, restricted CPU-only container, `run_e2.ps1` boundary | E2 suite |
| Prompt | `lfm-code-v1` (`f3d4b172…98cf6`), single user message, embedded chat template | E1 suite v2 |
| Extractor | `python-solution-v1`, `ast.parse` only | E2 chain |
| Decoding | Greedy (`temperature=0`), seed 42 recorded, 1 sample per task | E0 contract |
| Context | 4,096 total; 3,072 max output; overflow recorded, never truncated | E0 contract |
| KV cache | F16 K / F16 V, `--fit off`, parallel 1 | E0 contract |

## Permanently burned task IDs

`HumanEval/0`, `HumanEval/17`, `HumanEval/119` are development-only forever
(E0–E2). They are excluded from the pilot and the sealed benchmark and can
never enter a headline score.

## Task slices

Deterministic selection, auditable by hand:

- Eligible pool: task IDs 0–163 minus burned IDs = **161 tasks**.
- **Pilot slice (E3):** every 5th eligible ID, 0-based, ascending → **33 tasks**.
  Canonical list (one `HumanEval/<n>` per line, ascending) SHA-256:
  `d22bef9489c7643b6f61a4b8e0404cdf3499e9be57f0ac7d0dd4c39d7a4119c2`.
  IDs: 1, 6, 11, 16, 22, 27, 32, 37, 42, 47, 52, 57, 62, 67, 72, 77, 82, 87,
  92, 97, 102, 107, 112, 117, 123, 128, 133, 138, 143, 148, 153, 158, 163.
- **Pilot IDs become burned after E3** — they are used to size thresholds and
  validate the harness at scale, so the sealed run stays uncontaminated by
  its own tuning.
- **Sealed slice (E4):** remaining **128 tasks**.

The E3 and E4 suite JSONs (with dataset hash, slice lists, and prompt/extractor
identity) are frozen at implementation time, before any sealed request.

## Model rows and staged screening

Avoid the full Cartesian product (host doc §7). Rows enter later stages only
if they earn it:

| Stage | Rows | Tasks | Purpose |
|---|---|---|---|
| A — pilot (E3) | original BF16 GGUF, merged BF16 GGUF, Q8_0, Q6_K | 33 | harness at scale, retry policy, extraction yield, threshold sizing |
| B — sealed quality (E4) | original BF16, merged BF16, the two strongest pilot quants (expected Q8_0 + Q6_K) | 128 | headline paired correctness |
| C — deployment regression | selected deployment quant vs original + merged | 128 | KV-cache screen (F16 → Q8_0 → Q4_0), 4K/16K/32K context qualification, final speed/memory table |

Q5_K_M and Q4_K_M join Stage B only if Q6_K fails a pilot gate. Q2_K remains a
clearly labelled ultra-low-memory diagnostic and is never a recommendation.

## Prerequisite P5.0 — original-model baseline conversion

The original checkpoint is locally cached and hash-frozen
(`LiquidAI/LFM2.5-1.2B-Thinking` @ `95053d21d8e0b7ca99421a2127ae39c64f685ff3`,
`D:\hf_models\models--LiquidAI--LFM2.5-1.2B-Thinking\snapshots\…`, 1,170,340,608
BF16 params, per `reports/phase0/original-checkpoint-manifest.json`).

Convert it to BF16 GGUF offline with the pinned phase-1 toolchain container
under the same restrictions as `convert_phase1_bf16.ps1`, rerun the existing
conversion-parity validation, and record the artifact hash in a phase-5
report. This closes the host doc's "same-runtime upstream baseline" gap: every
Step 5 row is then one GGUF served by one llama.cpp image.

## Harness extensions frozen for E3

1. **Retry policy** (promised by the E0 contract for E3):

   | Event | E3+ behavior |
   |---|---|
   | Server death, CUDA error, HTTP 5xx, health-check failure | One automatic retry on a fresh server, recorded as `attempt: 2` with the original failure preserved |
   | Second consecutive infrastructure failure | Abort the row; do not skip silently |
   | Empty completion, token-limit finish, malformed answer | Valid model outcome; never retried |
   | Extraction failure (unparseable / missing function / ambiguous) | Record per task as `extraction_failure`, score 0, raw output preserved unchanged; **the batch continues** |
   | Duplicate generation key | Reject unless explicitly resuming |
   | Wrong model hash / alias / context | Abort before the first request |

   Note the deliberate contract evolution: E2's extractor was all-or-nothing
   per batch because three clean samples were expected. At 33–128 tasks a
   single unextractable output must not void a row; the no-repair rule is
   unchanged, only the blast radius moves from batch to task. This is frozen
   in the E3 suite before any run.

2. **Server reuse:** one server load per model row serves the whole slice
   sequentially (parallel 1), instead of one process per task. Row identity is
   re-verified from the lock before the first request of each row.

3. **Slice builder:** generates `e3`/`e4` suite JSONs from the frozen selection
   rule above and fails if the burned-ID set, dataset hash, or prompt hash
   drifts.

## Predeclared gates (final numbers frozen after the pilot)

The pilot's job is to make these feasible; they freeze before any sealed
request, and the sealed report is judged against them mechanically:

- **Deployment quant gate (proposed):** Q6_K is deployable iff sealed
  pass@1(plus) is within **2 percentage points** of the merged BF16 GGUF row
  AND no failure class appears that the merged row does not exhibit (judged by
  paired lost/gained task lists, not rounded percentages). Otherwise Q8_0.
- **Fine-tune claim:** reported as paired delta (gained/lost task counts plus
  pass@1 delta), original vs merged, same runtime.
- **Harness health:** infra failure rate ≤ 2% of requests and extraction yield
  ≥ 95% during the pilot, or the pilot report must explain the gap before E4.
- **Context overflow:** expected 0 (HumanEval prompts tokenize well under 1K;
  budget 3,072 output). Any overflow is recorded, never truncated.

## Measurement protocol

- Quality rows are greedy and deterministic; deltas are exact for this
  configuration and carry no seed variance. Generalization to temperature-0.6
  sampling is **not** claimed by this track.
- Performance numbers come from llama.cpp timing fields; one cold-load record
  per row, then steady-state. AC power, no concurrent GPU load, thermals noted;
  Windows build / WSL kernel / Docker Desktop / driver versions recorded per
  row (WSL NVML is feature-incomplete — sample host-side too).
- Decode speed differences between quants are deployment evidence, never
  quality evidence.

## Feasibility

From the E1 v1 run on this host: ~130 tok/s decode, ~6–7K tok/s prefill,
dev-task completions 73–117 tokens. Budgeting ~200 output tokens average:
a 33-task pilot row is minutes; a 128-task sealed row is well under an hour
including model load. Evidence discipline, not compute, is the schedule.

## Evidence layout and final report

Each row run writes `reports/e3/` (pilot) or `reports/e4/` (sealed) with the
E1/E2 artifact set per row: run manifest, server command, environment,
`generations.jsonl`, extraction report, `samples.jsonl`, scorer evidence, logs.
Existing evidence is never overwritten.

The Step 5 report is one table with one row per model:
artifact + SHA-256, size, pass@1 base / plus, paired delta vs original, paired
delta vs merged BF16, extraction failures, timeouts, mean output tokens,
prefill / decode tok/s, peak VRAM, load time — plus all reproducibility hashes.

## Stated limitations (frozen wording for the report)

- Absolute sealed scores use a custom prompt and greedy decoding: valid for
  **paired internal comparison**, not for public-leaderboard comparability.
- The fine-tune trained on KodCode (synthetic Leetcode-style content); treat
  absolute HumanEval+ scores as contamination-exposed and rely on the
  original-vs-merged paired delta for the fine-tune claim.
- 32K is the maximum support claim anywhere in Step 5; Stage C context results
  beyond 4K are qualification, not endorsement of 128K metadata.

## Acceptance gates

**E3 (pilot) passes when:** all four rows × 33 tasks produce one terminal
record each; retry and extraction-failure evidence is recorded per policy; the
pilot report states measured yield, infra failure rate, per-row timing; the
deployment-quant and harness-health numbers are frozen into the E4 suite.

**E4 (sealed) passes when:** sealed rows complete under the frozen suite; the
final table is produced with every hash recorded; a third party can reproduce
any request from the evidence; no claim exceeds the gates above.
