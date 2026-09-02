# E0 Evaluation Contract

Status: **frozen; ready for E1**  
Contract ID: `e0-humanevalplus-smoke-v1`  
Created: 2026-08-08  
Frozen: 2026-08-11  
Next phase: E1 generation-only vertical slice

## Purpose

E0 freezes one deliberately small development run so E1 can answer a single
question:

> Can three pinned HumanEval+ tasks travel through the validated merged BF16
> GGUF and produce complete, auditable raw generation records?

This is a harness-development contract, not a model-quality benchmark. Its
HumanEval+ tasks are permanently development-only and must be excluded from any
later sealed or headline score.

## Non-goals

E0 and E1 do not:

- execute generated code;
- calculate pass@1;
- compare the base model with the fine-tune;
- compare quantizations;
- tune for 128K context;
- make a release-quality claim;
- silently repair, sanitize, or retry a model answer.

Code extraction and isolated execution begin in E2.

## Why prompt-integrity checks were added

Manual LM Studio checks exposed two behaviors that should be classified before
the benchmark tasks run:

1. A one-word greeting caused the model to invent missing conversation state
   about a custom-pajama order.
2. Unfenced Python was rendered as Markdown, visually hiding indentation and
   interpreting `__init__` as emphasis.

E1 therefore begins with four unscored prompt-integrity checks. They are saved
through the same recorder as benchmark tasks but never enter an EvalPlus score.

## Frozen E1 run shape

| Field | E0 decision |
|---|---|
| Suite | `e0-humanevalplus-smoke-v1` |
| Run kind | Development-only, generation-only |
| Prompt-integrity cases | 4 unscored requests |
| Benchmark | HumanEval+ v0.1.10 |
| Benchmark cases | 3 development-only task IDs |
| Model | Merged fine-tune BF16 GGUF |
| Runtime | Pinned llama.cpp commit `3018a11e79e489b657dbb77c95694889ccff92df` |
| Context | 4,096 total tokens |
| Output maximum | 3,072 tokens |
| Parallel slots | 1 |
| GPU offload | All layers |
| Flash Attention | On |
| KV cache | F16 K and F16 V |
| Automatic fit | Off |
| Sampling | Greedy (`temperature=0`) |
| Seed | 42, recorded even though greedy decoding should be deterministic |
| Samples per case | 1 |
| Automatic retries | None |
| Code execution | Forbidden in E1 |

The machine-readable source of these decisions is
`benchmarks/suites/e0-humanevalplus-smoke-v1.json`.

## Model identity

Artifact:

`artifacts/gguf/lfm2.5-1.2b-thinking-kodcode-bf16.gguf`

Expected SHA-256:

`3e39c292be1f2b2740c3e05a2cb9081260439033e1b05f601117c72c8a73b8e6`

Expected size:

`2,343,326,048` bytes

This artifact already passed the Phase 1 Hugging Face-to-GGUF parity gate. E1
must verify its identity from the existing lock/report before starting the
server. Rehashing the 2.34 GB file on every task is unnecessary; hash once at
run start or validate against a trusted, freshly checked run manifest.

## Runtime identity and arguments

The server must use the pinned CUDA llama.cpp toolchain already recorded in
`configs/evaluation-lock.json`. The intended E1 arguments are:

```text
--model <read-only BF16 GGUF path>
--alias merged-bf16
--ctx-size 4096
--n-gpu-layers all
--flash-attn on
--cache-type-k f16
--cache-type-v f16
--fit off
--parallel 1
--host 127.0.0.1
--port 8080
--offline
--no-warmup
```

E1 must obtain the actual option names from the pinned binary's `--help`, store
the exact argument array in `server-command.json`, and abort if the loaded model
identity or context differs from the contract.

## Prompt-integrity cases

These requests are unscored. Each starts in an independent fresh conversation.

| ID | User request | Intended diagnostic |
|---|---|---|
| `integrity-greeting` | `Hi` | Detect invented conversation state or irrelevant role behavior. |
| `integrity-exact` | `Reply with exactly the single word: Hello` | Detect failure to follow a trivial exact-output constraint. |
| `integrity-state` | `What is the only message I have sent you in this conversation?` | Detect imagined prior turns or stale prompt state. |
| `integrity-code-fence` | Request a two-integer sum function in one fenced Python block | Verify code fencing, indentation preservation, and Markdown-safe output. |

These are diagnostics, not acceptance tests of broad chat quality. The raw
response is authoritative; rendered Markdown is not.

## HumanEval+ development cases

Verified selection from HumanEval+ v0.1.10:

| Task ID | Intended shape | Why selected |
|---|---|---|
| `HumanEval/0` | Numeric/list tolerance logic | Small and easy to inspect manually. |
| `HumanEval/17` | Structured string parsing | Exercises formatting and multiple branches. |
| `HumanEval/119` | String/parenthesis reasoning | Adds a less mechanical edge-case shape. |

The extracted dataset is frozen at:

`HumanEvalPlus.jsonl/HumanEvalPlus.jsonl`

Validation result:

- 164 valid JSONL records;
- 164 unique task IDs;
- numeric IDs 0 through 163 with no gaps;
- no JSON parse errors;
- selected entry points confirmed as `has_close_elements`, `parse_music`, and
  `match_parens`.

Extracted-file SHA-256:

`42526ec0e7d5f3ee0b06d6ced98f8c8bae3d76519151bfb3d36f79010645bd7f`

The uncompressed content hash is the E0 dataset identity. A future asset
acquisition manifest may additionally retain the original gzip hash.

The candidate EvalPlus scorer revision already recorded in
`configs/evaluation-lock.json` is:

`26d6d00bb1fd0fa37f39c99d5290da67891d1c5e`

E1 only loads prompts; E2 will validate that scorer revision against the
v0.1.10 dataset before executing tests.

## Prompt contract

Prompt version: `lfm-code-v0`

The exact source is `benchmarks/prompts/lfm-code-v0.txt`. E1 replaces the
single `{{BENCHMARK_PROMPT}}` placeholder with the unmodified public HumanEval+
prompt and sends the result as one user message in a fresh conversation.

No hidden tests, canonical solution, or generated EvalPlus inputs may be added
to the model request.

The runtime applies the GGUF-embedded LFM chat template. E1 must save:

- the prompt-template file hash;
- the unmodified benchmark prompt;
- the final user-message content;
- the final token count reported by the runtime.

The pinned llama.cpp `/apply-template` endpoint returns a string without the
leading BOS because its ordinary string-completion path adds BOS later. E1
sends token IDs to `/completion`, so its `/tokenize` call must use
`add_special=true`. Every recorded prompt must begin with exactly one frozen
BOS token ID (`1`); zero or duplicate BOS tokens fail the run.

The raw completion, including any reasoning tags, must remain unchanged.
Extraction and sanitization are explicitly deferred to E2.

## Context budget

E1 uses a 4,096-token total context with a maximum of 3,072 generated tokens.

Before inference:

```text
templated_prompt_tokens + max_output_tokens <= 4096
```

If the inequality fails, record `context_overflow` and do not truncate the
prompt or reduce the output allowance silently.

## Required run artifacts

```text
results/<run-id>/
|-- run-manifest.json
|-- server-command.json
|-- environment.json
|-- generations.jsonl
|-- behavior-review.json
`-- logs/
    |-- server.stdout.log
    `-- server.stderr.log
```

Each JSONL generation record must contain at least:

- schema version;
- suite ID and run ID;
- case kind (`prompt_integrity` or `humanevalplus_dev`);
- benchmark release and task ID when applicable;
- sample ID and attempt number;
- model alias, path-relative artifact ID, and SHA-256;
- runtime revision;
- prompt version and prompt SHA-256;
- exact messages sent to the API;
- templated prompt-token count;
- complete generation parameters;
- untouched raw completion;
- finish reason;
- prompt, completion, and total token counts;
- llama.cpp timing fields;
- UTC start/end timestamps;
- status or infrastructure-error classification.

## Failure and retry policy

| Event | E1 behavior |
|---|---|
| Wrong model hash, alias, or context | Abort before the first request. |
| Server fails health check | Record logs and abort. |
| Prompt exceeds context budget | Record `context_overflow`; never truncate. |
| HTTP or CUDA error | Record the full infrastructure failure and stop. |
| Empty completion | Preserve as a valid model outcome; do not retry. |
| Token-limit completion | Preserve the finish reason; do not retry. |
| Irrelevant or malformed answer | Preserve unchanged; it is not a harness error. |
| Duplicate generation key | Reject unless the runner is explicitly resuming. |

There are no automatic retries in E1. Retry policy is implemented and tested in
E3 after the evidence model is understood.

## E1 acceptance gate

E1 passes only when:

1. The server loads the expected BF16 GGUF and runtime configuration.
2. All four prompt-integrity cases and exactly three HumanEval+ development
   cases produce one terminal record each.
3. Every prompt starts with exactly one BOS token and all frozen
   prompt-integrity behavior expectations pass.
4. Nothing is silently truncated, repaired, or retried.
5. Every request can be reconstructed from saved evidence.
6. All result files remain valid after clean shutdown.
7. Repeating one selected case after a server restart produces the same raw
   greedy completion and token counts.
8. No generated code is executed anywhere in E1.
9. The user can open one record and explain the identity, prompt, response,
   timing, and status fields before E2 begins.

## Frozen review decisions

The E0 contract freezes these choices:

- Use merged BF16 GGUF—not Q6_K—for the first reference run.
- Include four unscored prompt-integrity cases.
- Use HumanEval+ task IDs 0, 17, and 119 as permanent development cases.
- Use one user-message wrapper and the embedded LFM chat template.
- Use greedy generation with a 3,072-token maximum.
- Allow no automatic retries and no code execution in E1.

The next work is E1: implement the generation-only vertical slice. Docker is
required for the pinned llama.cpp container, but generated code remains
unexecuted until E2.
