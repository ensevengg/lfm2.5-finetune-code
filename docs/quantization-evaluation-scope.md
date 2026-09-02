# LFM2.5 Quantization and Offline Evaluation Scope

Status: proposed implementation scope  
Date: 2026-07-29  
Source checkpoint: `lfm2.5-model-merged/`

## Outcome

Produce a reproducible GGUF release of the merged coding fine-tune, select a
quality-preserving quantization that runs with comfortable headroom on the
RTX 5060 Laptop 8 GB, and evaluate it against the original
`LiquidAI/LFM2.5-1.2B-Thinking` model in a network-disabled, least-privilege
Docker container.

The deployment target is:

- 32K context as the supported and scored baseline.
- 128K as an experimental qualification target.
- 256K only as an explicitly labelled stretch experiment, not a supported
  release target.
- Q6_K weights with a Q8_0 KV cache as the initial quality-first candidate.
  Q8_0, Q5_K_M, and Q4_K_M will be measured before the final choice.

## Facts established from the repository and host

| Item | Observed value |
|---|---|
| Merged model directory | `lfm2.5-model-merged/` |
| Architecture | `Lfm2ForCausalLM`, 10 convolution + 6 GQA layers |
| Parameter count | 1,170,340,608 |
| Source tensor type | BF16 |
| Safetensors size | 2.18 GiB |
| Safetensors SHA-256 | `C6C7A52C48E14A4B3185F8DF6A17AFB51908A31E41D07568B3027504039B3616` |
| Configured position limit | `max_position_embeddings: 128000` |
| GPU | NVIDIA GeForce RTX 5060 Laptop GPU, 8,151 MiB, compute capability 12.0 |
| NVIDIA driver | 610.74 |
| Docker | 29.6.2 client installed; Linux engine was not running during inventory |
| Current evaluation support | Only the 1% training holdout; no standalone benchmark harness |
| Local llama.cpp tools | Not installed |

There is a provenance gap to close before publication: `merge_model.py` names
the base model and adapter but does not pin either Hugging Face revision.
The merged artifact is hashable, but the exact base and adapter commits must
also be recovered and recorded if possible.

## Context-length decision

Liquid AI's current model card declares a **32,768-token context length** and
describes 32K as the full context in its long-context performance discussion.
The local model config says `128000`. The config value shows that the runtime
can be asked to allocate that context; it does not establish that the released
Thinking checkpoint was trained or validated for useful 128K behavior.

Accordingly:

1. 32K is the supported release baseline.
2. 64K and 128K are experimental and must pass both memory and long-context
   quality gates.
3. 256K is outside both the published 32K context and the local 128K config.
   It would require RoPE extrapolation and a separate quality study. It is not
   part of the v1 acceptance criteria.

Primary sources:

- [Liquid AI model card](https://huggingface.co/LiquidAI/LFM2.5-1.2B-Thinking)
- [Liquid AI release article](https://www.liquid.ai/blog/lfm2-5-1-2b-thinking-on-device-reasoning-under-1gb)

## Memory envelope

Only six layers have attention KV state. With 8 KV heads, a head dimension of
64, and both K and V, the unpadded cache has 6,144 elements per token.

| Context | F16 KV | Q8_0 KV | Q4_0 KV | Qualification |
|---:|---:|---:|---:|---|
| 32,768 | 0.375 GiB | 0.199 GiB | 0.105 GiB | Supported baseline |
| 128,000 | 1.465 GiB | 0.778 GiB | 0.412 GiB | Experimental |
| 256,000 | 2.930 GiB | 1.556 GiB | 0.824 GiB | Unsupported stretch |

These are cache payload estimates, not total process VRAM. CUDA kernels,
runtime buffers, batch size, graph capture, and allocator overhead must be
measured.

Liquid AI's official GGUF repository gives a reliable size guide for the same
architecture:

| Weight format | Published size | Expected role |
|---|---:|---|
| BF16/F16 | 2.34 GB | Conversion/parity reference |
| Q8_0 | 1.25 GB | Near-reference quantized baseline |
| Q6_K | 963 MB | Initial deployment candidate |
| Q5_K_M | 843 MB | Smaller candidate |
| Q4_K_M | 731 MB | Maximum-headroom candidate |

At 128K, Q6_K weights plus a Q8_0 cache are approximately 1.68 GiB before
runtime buffers. This makes 128K plausible on 8 GB by memory alone. The
uncertainties are model quality at that length and full-prompt prefill time,
not model-weight capacity.

Source: [Liquid AI official GGUF repository](https://huggingface.co/LiquidAI/LFM2.5-1.2B-Thinking-GGUF)

## Phase 0: freeze inputs and define reproducibility

Status as of 2026-07-30: the checkpoint/provenance and container-readiness
gates pass. The base, adapter, and training-dataset revisions were recovered
from the retained local Hugging Face cache; both checkpoints load offline;
tokenizer/template parity passes; 92 of 148 tensors changed; and all six
deterministic smoke completions differ. CUDA 12.8.1 build/runtime image
manifests are pinned, and a restricted pinned base image passed GPU,
no-network, read-only, non-root, capability, and writable-results checks.
Benchmark-data and compiler locks remain pending until the exact scored task
bundle is vendored. See the [Phase 0 report](../reports/phase0/README.md).

Deliverables:

- A manifest containing hashes for every file in `lfm2.5-model-merged/`.
- Exact base-model and adapter repository revisions, or an explicit
  "unrecoverable" provenance note.
- A second copy or hash manifest of the untouched original Thinking checkpoint.
- Pinned revisions for llama.cpp, benchmark harnesses, datasets, container base
  images, Python wheels, and compiler packages.
- A machine manifest containing OS, driver, GPU, Docker, CUDA runtime, CPU,
  RAM, and free disk.

Gate:

- The merged Hugging Face checkpoint loads with `local_files_only=True`.
- Tokenization and the chat template work without network access.
- Fixed smoke prompts show that the merge is not a no-op.

## Phase 1: build a pinned GGUF/CUDA toolchain

Status as of 2026-07-30: **PASS**. `llama.cpp` commit
`3018a11e79e489b657dbb77c95694889ccff92df` was built with CUDA 12.8.1 for
compute capability 12.0. The merged checkpoint was converted to a 2.34 GB
BF16 GGUF with SHA-256
`3e39c292be1f2b2740c3e05a2cb9081260439033e1b05f601117c72c8a73b8e6`.
All static metadata gates pass, all 17 layers offload to the RTX 5060, and all
six Phase 0 prompts have identical tokenization and identical 64-token greedy
completions between Hugging Face and GGUF. See the
[Phase 1 report](../reports/phase1/README.md).

Use a known llama.cpp commit that explicitly supports Liquid LFM2/LFM2.5 and
build it for CUDA compute capability 12.0. Do not use an unpinned `master`
binary for final results.

Required tools:

- `convert_hf_to_gguf.py`
- `llama-quantize`
- `llama-cli`
- `llama-server`
- `llama-bench`
- `llama-perplexity`
- `llama-imatrix` if the importance-matrix path is retained

Conversion is a two-stage pipeline:

1. Convert the merged Hugging Face checkpoint to a high-quality BF16 GGUF.
2. Quantize that BF16 GGUF with `llama-quantize`.

The current upstream converter accepts BF16/F16/Q8_0 directly but not K-quants.
Liquid's shorter documentation example that passes `q4_k_m` directly to the
converter should therefore not be copied into the implementation.

Sources:

- [llama.cpp quantization documentation](https://github.com/ggml-org/llama.cpp/blob/master/tools/quantize/README.md)
- [current converter options](https://github.com/ggml-org/llama.cpp/blob/master/convert_hf_to_gguf.py)
- [Liquid AI llama.cpp guide](https://docs.liquid.ai/deployment/on-device/llama-cpp)

## Phase 2: produce and structurally validate the quantization sweep

Status as of 2026-08-02: **PASS — artifact production and static validation
only**. Q8_0, Q6_K, Q5_K_M, Q4_K_M, and Q2_K were each produced directly from
the verified BF16 GGUF. The source hash was unchanged before and after the run.
See the [Phase 2 report](../reports/phase2/README.md).

The sweep deliberately used no importance matrix: no representative,
benchmark-disjoint calibration corpus has been selected. It also avoided
requantization, so every format has exactly one quantization step from BF16.

Completed gates:

1. Each file opens with the exact pinned GGUF parser.
2. `general.file_type` matches the requested quantization.
3. All 148 tensors are present and quantized tensor types are nonempty.
4. Architecture, model dimensions, RoPE base, tokenizer, special-token IDs,
   chat template, and 128,000-token context metadata match the BF16 source.
5. SHA-256, byte size, duration, toolchain identity, and container restrictions
   are recorded for every artifact.

Q2_K is included as an ultra-low-memory option. Its smaller size is not a
quality claim. Perplexity, behavior, code quality, speed, VRAM, context-length
qualification, comparisons against the untouched original, and selection of a
deployment winner are deferred until the benchmark methodology is researched
and frozen.

## Phase 3: qualify context and runtime performance

Run the selected weight quant with:

- Full GPU offload.
- Flash Attention enabled.
- Q8_0 K and V caches as the default long-context configuration.
- F16 and Q4_0 caches as quality/memory comparison points.
- Fixed logical and physical batch sizes, recorded in the result manifest.

The current llama.cpp server exposes `--flash-attn`, `--cache-type-k`, and
`--cache-type-v` for these controls:
[server options](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md).

Test ladder:

1. Allocation and one-token decode at 4K, 8K, 16K, 32K, 64K, and 128K.
2. Actual prefill and decode at 4K, 16K, and 32K.
3. Actual 64K and 128K prefill only after the allocation tests pass.
4. RULER-style retrieval/reasoning plus multi-needle tests at each length.
5. Repeat the 32K and 128K tests with F16, Q8_0, and Q4_0 KV caches to measure
   cache-quantization quality loss.

Record:

- Peak and steady-state VRAM.
- Host RAM.
- Model load time.
- Prompt-processing tokens/second.
- Time to first token.
- Decode tokens/second.
- Long-context task accuracy.
- Any CUDA fallback, OOM, cache, or graph warnings.

128K is qualified only if it both runs and retains acceptable task accuracy.
An allocation success alone is not support evidence.

## Phase 4: build the offline evaluation image

Use separate acquisition/build and execution concerns:

1. While online, pin and vendor all source archives, wheels, benchmark data,
   tokenizers, compilers, and model artifacts.
2. Build a digest-pinned CUDA runtime image containing the compiled llama.cpp
   tools and evaluation harness.
3. Run the final image without a network and prove that it has no missing
   dependency.

Runtime restrictions:

- `--network none`
- `--read-only`
- `--cap-drop ALL`
- `--security-opt no-new-privileges=true`
- Non-root numeric user
- Default or tighter seccomp profile
- Explicit memory, CPU, process, file-size, and timeout limits
- Read-only model and benchmark mounts
- Exactly one writable results mount
- Writable temporary files only on size-limited tmpfs mounts
- One explicitly selected GPU with only `compute,utility` driver capabilities
- No `--privileged`, host IPC, Docker socket, SSH agent, credentials, home
  directory, source tree, or broad drive mounts
- `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`,
  `HF_DATASETS_OFFLINE=1`, and llama.cpp `--offline`

Security acceptance checks:

- DNS and outbound HTTP fail.
- Writes to the image filesystem and read-only mounts fail.
- Paths outside the declared mounts are not visible.
- The writable results directory works.
- The GPU is visible and inference succeeds.
- The container stops at its memory, process, and timeout limits.

Docker isolates files, processes, and networking, but it is not a virtual
machine boundary against kernel or GPU-driver vulnerabilities. If the
benchmark inputs or generated programs are considered actively hostile, the
same image should run inside a disposable Hyper-V/VM host rather than relying
on Docker alone.

Sources:

- [Docker `--network none`](https://docs.docker.com/engine/network/drivers/none/)
- [Docker run security controls](https://docs.docker.com/reference/cli/docker/container/run/)
- [NVIDIA GPU and driver capability controls](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/docker-specialized.html)

## Phase 5: evaluation matrix

Run four model/runtime cells:

| Cell | Model | Purpose |
|---|---|---|
| A | Original Thinking, BF16 | Upstream quality reference |
| B | Merged fine-tune, BF16 | Isolate fine-tuning effect |
| C | Original Thinking, selected GGUF quant | Isolate quantization effect on upstream |
| D | Merged fine-tune, selected GGUF quant | Final deployment result |

Where a BF16 Hugging Face and BF16 GGUF runner cannot be made numerically
equivalent, add BF16 GGUF as a fifth parity cell rather than mixing runners
inside a comparison.

### Upstream benchmark family

The model card reports:

- GPQA Diamond
- MMLU-Pro
- IFEval
- IFBench
- Multi-IF
- GSM8K
- MATH-500
- AIME25
- BFCLv3

The plan is to reproduce the same public datasets, not claim exact replication
of Liquid's numbers unless the exact prompt, answer extraction, custom BFCL
handler, and Artificial Analysis methodology can be reconstructed. Liquid
reports five runs at temperature 0.6 for thinking models.

Start with the public/core set: GPQA Diamond, MMLU-Pro, IFEval, GSM8K,
MATH-500, and AIME25. Add IFBench, Multi-IF, and BFCLv3 after their data,
licensing, and model-specific handlers are fully vendored and validated.

### Coding benchmark family

Required:

- HumanEval+ via EvalPlus.
- MBPP+ via EvalPlus.
- A pinned LiveCodeBench code-generation release.

Optional second wave:

- CRUXEval for code execution/reasoning.
- BigCodeBench for broader instruction-to-code coverage.

SWE-bench and repo-level agent benchmarks are out of the first scope: they test
an agent/tool stack as much as this 1.2B model and would dominate engineering
time before the basic coding signal is known.

### Data contamination audit

KodCode contains material derived from LeetCode, Codeforces, APPS, TACO,
CodeContests, and other programming sources. Before interpreting coding gains:

- Compare normalized problem text and known IDs where available.
- Run exact hashes and fuzzy/MinHash similarity against training prompts.
- Publish clean-only, suspected-overlap, and all-item scores separately.
- Treat LiveCodeBench date/version as part of the result.
- Never use scored benchmark items to build an importance matrix.

### Generation and scoring policy

- Preserve Liquid's published five-seed, temperature-0.6 track for comparisons
  intended to resemble the model card.
- Add a deterministic engineering track for quant-to-BF16 comparisons.
- Pin prompt templates, system prompt, stop tokens, max input/output tokens,
  reasoning-tag handling, code extraction, and answer normalization.
- Use identical settings across cells in each comparison.
- Save raw prompts, token counts, completions, extracted answers, scores,
  timings, and errors as immutable JSONL.
- Execute generated code inside the already restricted container with
  per-sample CPU, memory, process, output, and wall-clock limits.
- Report pass@1 for coding, with pass@k only when the sampling count and cost
  are predeclared.

## Phase 6: reporting and release

Deliverables:

- `scripts/` for manifesting, conversion, quantization, smoke tests, and
  benchmark orchestration.
- `docker/` with a digest-pinned Dockerfile, offline asset lock/manifest, and a
  hardened run wrapper.
- `benchmarks/` with task configurations, prompt versions, extraction rules,
  and contamination audit tooling.
- `artifacts/gguf/` for generated models, excluded from Git.
- `results/<run-id>/` with raw outputs, environment and command manifests,
  summary JSON, and tables.
- A final Markdown report comparing quality, VRAM, prefill/decode performance,
  context behavior, and contamination-adjusted coding scores.
- Checksums and licenses for every redistributed artifact.

Release gate:

- Conversion parity passes.
- At least one Q5-or-better GGUF passes the quality threshold.
- The chosen GGUF stays below the 6.5 GiB peak-VRAM target at the qualified
  context.
- 32K long-context quality passes; 128K is labelled supported only if its
  separate quality gate passes.
- The full selected benchmark subset completes twice from the offline image
  with reproducible scoring.
- All container isolation checks pass.

## Estimated effort

| Work | Engineering time |
|---|---:|
| Provenance, manifests, Docker/GPU prerequisite checks | 0.5 day |
| Pinned llama.cpp CUDA build and GGUF conversion | 0.5-1 day |
| Quant sweep, parity, and selection | 1 day |
| Hardened offline image and asset bundle | 1-2 days |
| Core upstream and coding harness integration | 2-4 days |
| Long-context and performance qualification | 1 day |
| Report and reproducibility rerun | 0.5-1 day |

Expected scope: roughly 6-10 engineering days plus benchmark runtime. The main
schedule risks are exact upstream-methodology reproduction, offline packaging
of every dataset dependency, and Docker Desktop GPU bring-up—not model
quantization itself.

## Explicitly out of scope for v1

- Claiming 256K support.
- Training or changing model weights.
- Editing the merged checkpoint in place.
- Publishing models or results to the internet.
- Installing a host-wide inference stack when a pinned container can provide it.
- Exact replication claims for benchmarks whose private/custom evaluation
  methodology is unavailable.
