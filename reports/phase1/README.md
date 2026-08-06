# Phase 1 result

Generated: 2026-07-29T22:11:27.967554+00:00

## Outcome

The pinned CUDA `llama.cpp` toolchain and merged BF16 GGUF reference both
pass. Phase 2 can use this GGUF as the source for the Q8_0, Q6_K, Q5_K_M, and
Q4_K_M sweep.

## Toolchain

- `llama.cpp` commit:
  `3018a11e79e489b657dbb77c95694889ccff92df`
- Image: `lfm25/llama-cpp:3018a11e-cuda128-sm120`
- Image ID:
  `sha256:550d53ae97c4d83a6c0fba17cd8f532aa6f775e9498d2452fa6f04e33707b4b0`
- CUDA: 12.8.1, compiled for compute capability 12.0
- GPU smoke: NVIDIA GeForce RTX 5060 Laptop GPU detected
- Converter: Transformers 4.57.6, Torch 2.11.0+cpu, GGUF 0.19.0

The build uses the exact source commit and digest-pinned CUDA build/runtime
images. It disables the bundled UI, network client, RPC, tests, and native
host-specific CPU tuning. The runtime payload retains only the eight tools
required by the quantization and evaluation plan.

## BF16 GGUF

- Artifact:
  `artifacts/gguf/lfm2.5-1.2b-thinking-kodcode-bf16.gguf`
- Size: 2,343,326,048 bytes
- SHA-256:
  `3e39c292be1f2b2740c3e05a2cb9081260439033e1b05f601117c72c8a73b8e6`
- Architecture: `lfm2`
- File type: `MOSTLY_BF16`
- Context metadata: 128,000 tokens
- Tensors: 148 total; 93 BF16 and 55 F32

The 55 F32 tensors are the expected one-dimensional norms and short-conv
weights retained at higher precision by the upstream BF16 converter.

The merged checkpoint was saved with Transformers 5.13's
`TokenizersBackend` class name. The pinned converter uses Transformers 4.57.6,
so conversion uses the read-only compatibility file
`configs/phase1-tokenizer-config.json`, which names the equivalent
`PreTrainedTokenizerFast` class. The source checkpoint remains unchanged.
Exact tokenizer parity below is the acceptance proof for this override.

## Validation gates

- PASS: architecture, file type, context, block, embedding, and head metadata
- PASS: BOS, EOS, and PAD token IDs
- PASS: byte-for-byte chat-template parity
- PASS: source and GGUF tensor count
- PASS: all six Phase 0 prompts produce identical token IDs
- PASS: all six first greedy completion tokens match
- PASS: all six full 64-token greedy completions match
- PASS: CUDA0 selected and 17/17 layers offloaded

The 4K validation run used 2,232.50 MiB of CUDA model buffers, 48.00 MiB of
CUDA KV cache, and 60.03 MiB of CUDA compute buffers. These figures establish
the BF16 conversion reference only; they are not the Phase 3 long-context
memory qualification.

Overall Phase 1 gate: **PASS**

## Container boundary

Conversion and validation used:

- no network and no image pulls;
- a read-only root filesystem;
- numeric non-root UID/GID 65534;
- all Linux capabilities dropped and no-new-privileges;
- explicit CPU, memory, process, and tmpfs limits;
- read-only model, GGUF, reference, and script mounts;
- only the artifact or report directory writable;
- only GPU 0 with NVIDIA `compute,utility` capabilities for inference.

## Evidence files

- `toolchain-build.json`
- `bf16-conversion.json`
- `bf16-gguf-metadata.json`
- `bf16-gguf-validation.json`
- `bf16-gguf-server.log`
- `../../configs/evaluation-lock.json`

The BF16 artifact is excluded from Git; its hash and provenance are locked in
the reports above.
