# LFM2.5 Thinking: quantization and offline-evaluation scope

Research date: 2026-07-29

This note records the primary-source facts that should govern quantization of the
locally merged `LFM2.5-1.2B-Thinking` checkpoint and its evaluation on an RTX
5060 8 GB. It is a planning note, not an implementation.

## Executive conclusions

1. **Plan around 32,768 tokens, not 128K or 256K.** Liquid AI's current model
   card declares a 32,768-token context, and its launch post calls 32K the
   model's "full" context. Both the official checkpoint config and the local
   merged config nevertheless contain `max_position_embeddings: 128000`.
   Therefore:

   - 32K is the published and validated context target.
   - 64K/128K are experimental stress targets permitted by metadata, but their
     quality is not supported by Liquid AI's published evidence.
   - 256K is out of spec and should not be part of the main deliverable.

   Sources: [Liquid AI model card](https://huggingface.co/LiquidAI/LFM2.5-1.2B-Thinking#model-details),
   [Liquid AI launch post](https://www.liquid.ai/blog/lfm2-5-1-2b-thinking-on-device-reasoning-under-1gb),
   [official config](https://huggingface.co/LiquidAI/LFM2.5-1.2B-Thinking/blob/main/config.json),
   and the [local merged config](../../lfm2.5-model-merged/config.json).

2. **Current llama.cpp has native LFM2 support.** The current converter
   explicitly registers `Lfm2ForCausalLM`, handles attention-only KV heads,
   records the ShortConv cache, and reshapes the convolution tensors. Use a
   recent, pinned llama.cpp commit or image digest; do not combine a current
   converter script with an older `gguf-py` or runtime.

   Sources: [llama.cpp LFM2 converter](https://github.com/ggml-org/llama.cpp/blob/master/conversion/lfm2.py#L14-L83),
   [supported-model list](https://github.com/ggml-org/llama.cpp#supported-models).

3. **An 8 GB GPU does not require an aggressive weight quant for this 1.17B
   model.** Liquid AI's own GGUF files are 1.25 GB at Q8_0, 963 MB at Q6_K,
   843 MB at Q5_K_M, and 731 MB at Q4_K_M. Q8_0 is the quality-first deployment
   candidate; Q6_K is the likely "comfortable" default; Q4_K_M is useful as a
   compact comparison, not as the automatic choice.

   Source: [Liquid AI's official GGUF repository](https://huggingface.co/LiquidAI/LFM2.5-1.2B-Thinking-GGUF/tree/main).

4. **At long context, cache and prompt-processing buffers matter more than
   weights.** LFM2.5's hybrid layout has only six full-attention layers, so its
   KV cache is relatively small, but 128K prefill is still an experimental,
   compute-heavy path. Quantized KV cache can save hundreds of MiB; it must be
   evaluated for quality rather than enabled silently.

5. **Offline enforcement must come from Docker, not only from application
   flags.** Pre-stage a pinned image, models, benchmark datasets, and hashes;
   then run with no network, a read-only root filesystem, no capabilities or
   privilege escalation, read-only input mounts, and only one narrow writable
   results mount. Expose only one GPU and the `compute,utility` NVIDIA driver
   capabilities.

## Checkpoint facts and context discrepancy

The merged checkpoint in this repository is actually named
`lfm2.5-model-merged` (with a trailing `d`). Its config says:

- architecture: `Lfm2ForCausalLM`
- 16 layers: 10 convolution and 6 `full_attention`
- hidden size 2,048
- 32 attention heads and 8 KV heads
- `max_position_embeddings: 128000`
- RoPE base: 1,000,000
- BF16 source dtype

Source: [local merged config](../../lfm2.5-model-merged/config.json).

The official Hugging Face config has the same 128,000 metadata value, but the
current Liquid AI model card explicitly states **32,768 tokens**, and the launch
post reports throughput at 16K and "the full 32K context." No Liquid AI primary
source found for this checkpoint validates 128K or 256K. `max_position_embeddings`
is therefore not sufficient evidence of trained or evaluated long-context
quality.

This distinction should become an acceptance gate:

- Main benchmark and deployment claims stop at 32K.
- 64K and 128K results are labelled experimental and include long-context
  retrieval/recall tests against the original BF16 checkpoint.
- 256K is excluded unless the user explicitly expands the scope to unsupported
  RoPE/context-extension research.

The local coding fine-tune used sequences of at most 4,096 tokens, so it also
provides no new evidence for 128K behavior. Source: [project README](../../README.md#training).

## GGUF conversion and quantization path

The supported upstream flow is two-stage:

1. Convert the local Hugging Face checkpoint to a high-fidelity GGUF.
2. Quantize that GGUF with `llama-quantize`.

Source: [llama.cpp quantization guide](https://github.com/ggml-org/llama.cpp/blob/master/tools/quantize/README.md#prepare-the-input-gguf-file).

The current converter's direct output choices include F32, F16, BF16, Q8_0 and
`auto`; K-quants such as Q6_K and Q4_K_M are produced in the second stage.
For this BF16 source, preserve a BF16 GGUF as the conversion reference and
quantize each candidate from that reference. Do not requantize a previously
quantized file: llama.cpp warns that `--allow-requantize` can severely reduce
quality.

Sources: [converter CLI](https://github.com/ggml-org/llama.cpp/blob/master/convert_hf_to_gguf.py),
[quantizer options](https://github.com/ggml-org/llama.cpp/blob/master/tools/quantize/README.md#quantize-the-gguf).

Current LFM2-specific caveats:

- Use the complete, pinned llama.cpp source tree so `convert_hf_to_gguf.py`,
  `conversion/lfm2.py`, `gguf-py`, and the runtime agree.
- The current LFM2 converter accepts either `block_ff_dim` or
  `intermediate_size`, which matches this merged config.
- Importance-matrix generation previously failed for LFM2/recurrent 3-D
  activations. The fix was merged in August 2025 and was tested on LFM2; use a
  revision containing that fix.

Sources: [current LFM2 conversion code](https://github.com/ggml-org/llama.cpp/blob/master/conversion/lfm2.py#L14-L83),
[merged imatrix fix](https://github.com/ggml-org/llama.cpp/pull/14994).

Recommended candidate ladder:

| Candidate | Official base-model size | Purpose |
|---|---:|---|
| BF16 GGUF | 2.34 GB | Conversion/parity reference |
| Q8_0 | 1.25 GB | Quality-first deployment candidate |
| Q6_K | 963 MB | Likely default on an 8 GB GPU |
| Q5_K_M | 843 MB | Middle comparison |
| Q4_K_M | 731 MB | Compact comparison |

The sizes are from Liquid AI's official GGUF repository and will vary slightly
after the coding fine-tune. The quantizer supports an importance matrix; if one
is used, its calibration text should include representative coding data and
general reasoning/instruction data, and the exact calibration manifest must be
retained. Quantization is selected by measured quality delta, not file size
alone.

## KV-cache memory at long context

From the official/local config:

```text
6 attention layers
× 8 KV heads
× (2048 / 32 = 64 values per head)
× 2 (K and V)
= 6,144 cached scalar values per token
```

The convolution layers have small fixed state and do not add a full
context-length KV tensor. The following are derived cache-only estimates; they
exclude convolution state, graph/scratch buffers, CUDA allocator overhead,
batching, logits, and model weights.

| Context | F16/BF16 KV | Q8_0 KV | Q4_0 KV |
|---:|---:|---:|---:|
| 32,768 | 384 MiB | about 204 MiB | about 108 MiB |
| 128,000 | 1,500 MiB | about 797 MiB | about 422 MiB |
| 256,000 | 3,000 MiB | about 1,594 MiB | about 844 MiB |

The quantized estimates include llama.cpp's Q8_0/Q4_0 block scale overhead.
Block layouts are defined in
[ggml-common.h](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-common.h#L200-L247).

llama.cpp currently accepts `f32`, `f16`, `bf16`, `q8_0`, `q4_0`, `q4_1`,
`iq4_nl`, `q5_0`, and `q5_1` independently for K and V; F16 is the default.
It also exposes Flash Attention and GPU KV offload. Source:
[llama-server options](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md?plain=1#L2300-L2337).

Evaluate cache modes in this order:

1. F16/F16 as the cache-quality reference.
2. Q8_0/Q8_0 as the likely long-context choice.
3. Q4_0/Q4_0 only if additional headroom is needed.

Use the same K and V type initially. Verify the runtime log's allocated context
and cache type. Current llama.cpp has automatic device-memory fitting enabled
by default and can adjust unset parameters; for comparable benchmark runs,
freeze the chosen settings, disable auto-fit after discovery, and record any
CPU offload. Source:
[llama-server `--fit` options](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md?plain=1#L2358-L2367).

Even where the arithmetic fits under 8 GB, 128K is not guaranteed to fit or run
well: prompt-processing scratch space and latency must be measured with Flash
Attention, conservative micro-batches, and a single evaluation slot.

## Published evaluation suite

Liquid AI published the following results for the original Thinking model:

| Benchmark | Published score |
|---|---:|
| GPQA Diamond | 37.86 ± 0.83 |
| MMLU-Pro | 49.65 ± 0.18 |
| IFEval | 88.42 ± 0.35 |
| IFBench | 44.85 ± 0.73 |
| Multi-IF | 69.33 ± 0.09 |
| GSM8K | 85.60 ± 0.00 |
| MATH-500 | 87.96 ± 0.72 |
| AIME25 | 31.73 ± 1.81 |
| BFCLv3 | 56.97 ± 0.30 |

Source: [Liquid AI benchmark table](https://huggingface.co/LiquidAI/LFM2.5-1.2B-Thinking#benchmarks).

Methodology details that affect comparability:

- Thinking-model scores are means and standard deviations over five runs at
  temperature 0.6.
- GPQA Diamond, MMLU-Pro, IFBench, and AIME25 follow Artificial Analysis's
  methodology.
- IFEval and Multi-IF average strict/loose prompt and instruction accuracies.
- BFCLv3 uses a custom Liquid handler for the model's tool-call template.
- Normal model-card generation settings are different: temperature 0.05,
  top-k 50, repetition penalty 1.05.

Liquid AI has not published all artifacts needed for exact BFCL reproduction;
an official response says publication of the custom handler/reproduction method
was still being explored. Source:
[Liquid AI model discussion](https://huggingface.co/LiquidAI/LFM2.5-1.2B-Thinking/discussions/2).

Accordingly, evaluation should have two distinct tracks:

- **Published-comparability track:** five seeded runs at temperature 0.6 where
  the published methodology can be implemented; explicitly flag BFCL handler
  differences.
- **Controlled regression track:** deterministic decoding shared by original
  BF16, merged BF16, BF16 GGUF, and every quant/cache candidate. This isolates
  fine-tune, conversion, weight-quantization, and KV-quantization deltas.

The published table contains no code-execution benchmark. Liquid AI's current
model card and launch blog also give conflicting prose about programming
suitability, making dedicated coding evaluation necessary rather than optional.
Sources: [model card](https://huggingface.co/LiquidAI/LFM2.5-1.2B-Thinking#model-details),
[launch post](https://www.liquid.ai/blog/lfm2-5-1-2b-thinking-on-device-reasoning-under-1gb).

## Offline, restricted Docker boundary

On Windows, Docker Desktop GPU support requires the WSL2 backend and NVIDIA GPU
paravirtualization. Keep Windows, WSL, Docker Desktop, and the NVIDIA driver
current, then validate GPU access before creating the offline benchmark image.
Sources: [Docker Desktop GPU prerequisites](https://docs.docker.com/desktop/features/gpu/),
[Docker Desktop WSL security model](https://docs.docker.com/desktop/features/wsl/).

NVIDIA's container runtime can expose one selected GPU and only the
`compute,utility` driver capabilities needed for CUDA/NVML. It injects GPU
devices and driver mounts into the OCI container, so the GPU/host driver remains
a shared attack surface; this is isolation, not a physically separate machine.
Sources: [NVIDIA specialized Docker configuration](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/docker-specialized.html),
[NVIDIA runtime architecture](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/arch-overview.html).

Build/pull and populate the image while online, then pin and record:

- image digest and llama.cpp commit/build version
- model/GGUF SHA-256 hashes
- dataset and prompt-manifest hashes
- benchmark harness commit/version and Python lock file
- CUDA image, host NVIDIA driver, Docker Desktop, WSL, and GPU details

The offline launch policy should include:

- `--network none`; Docker's `none` driver creates only loopback
- `--read-only`
- `--cap-drop ALL`
- `--security-opt no-new-privileges=true`
- a non-root UID if CUDA access and the output ACL pass a smoke test
- `--gpus "device=0"` and
  `NVIDIA_DRIVER_CAPABILITIES=compute,utility`
- model, datasets, prompts, and harness mounted read-only
- one dedicated writable results mount plus size-limited `tmpfs` mounts
- CPU, host-memory, swap, and PID limits
- no `--privileged`, no host IPC/network, no Docker socket, no device mounts
  beyond the NVIDIA runtime, and no broad project/drive mount
- no published port; run the harness and llama.cpp in the same container
- llama.cpp `--offline` and library offline environment variables as
  defense-in-depth, while `--network none` remains the enforcement boundary

Sources: [Docker `none` network](https://docs.docker.com/engine/network/drivers/none/),
[read-only bind mounts](https://docs.docker.com/engine/storage/bind-mounts/),
[container run security/resource flags](https://docs.docker.com/reference/cli/docker/container/run),
[llama.cpp offline option](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md?plain=1#L2394-L2402).

Only the dedicated results directory should be host-writable. Docker warns that
writeable bind mounts can modify or delete host files; making every input mount
read-only directly addresses the requirement that benchmarks not alter the
host's core state.

## Proposed implementation scope and gates

### Phase 0: integrity and baseline

- Hash the merged checkpoint and audit config, tokenizer, special tokens, and
  chat template against the original.
- Run a native Transformers BF16 smoke test and deterministic baseline.
- Record peak VRAM/RAM and confirm the actual model directory spelling.

**Gate:** no NaNs, intact chat template, reproducible deterministic output.

### Phase 1: pinned GGUF conversion

- Pin a recent llama.cpp commit containing current LFM2 conversion and imatrix
  fixes.
- Convert local BF16 weights to BF16 GGUF without remote access.
- Compare tokenizer behavior and deterministic BF16 HF versus BF16 GGUF outputs;
  use logit/KL checks where the harness permits.

**Gate:** conversion parity is acceptable before any quantization.

### Phase 2: weight/cache candidate selection

- Produce Q8_0, Q6_K, Q5_K_M, and Q4_K_M independently from BF16 GGUF.
- Optionally produce imatrix and non-imatrix variants with a recorded calibration
  manifest.
- Test F16, Q8_0, then Q4_0 KV caches.
- Measure quality, peak VRAM/RAM, load time, prompt-processing throughput,
  generation throughput, and artifact size.

**Gate:** predeclare allowed quality loss; choose Q8_0 or Q6_K unless a smaller
quant demonstrably meets the same criterion. Keep a conservative VRAM margin
rather than targeting all 8 GB.

### Phase 3: context validation

- Qualify all candidates at 4K/16K/32K first.
- Run long-context retrieval/recall and realistic code-context tests at 64K and
  128K as explicitly experimental.
- Test 128K against the original BF16 model as well as the fine-tune so base
  limitations are not misattributed to training or quantization.

**Gate:** main support claim remains 32K. Do not silently report successful
allocation as proof of long-context quality.

### Phase 4: immutable offline evaluation image

- Build a CUDA-enabled llama.cpp/evaluation image with all dependencies and
  datasets embedded or read-only mounted.
- Generate an SBOM/checksum manifest if practical.
- Start it under the restricted launch policy and prove that network access and
  writes outside the result directory fail.

**Gate:** no network, no unexpected writable mount, no privileged/container
socket access, one GPU only, and results reproducible from the manifest.

### Phase 5: benchmark matrix and report

Run at least these model rows:

1. original Liquid AI BF16
2. locally merged BF16
3. locally merged BF16 GGUF
4. Q8_0
5. selected Q6/Q5 candidate
6. Q4_K_M

Run the original published suite where reproducible, plus a separately specified
coding suite. Report absolute score, delta from original, delta from merged BF16,
failure rate, output-token count, peak VRAM/RAM, prefill/decode throughput,
context/cache configuration, and all reproducibility hashes.

