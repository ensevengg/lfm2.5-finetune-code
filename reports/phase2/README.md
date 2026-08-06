# Phase 2: quantized GGUF artifacts

Status: **PASS — generated and structurally validated; unbenchmarked**

All five artifacts were independently quantized from the validated BF16 GGUF
with the pinned `llama.cpp` toolchain. No artifact was requantized from another
quant, and no importance matrix was supplied because no representative,
benchmark-disjoint calibration set has been established.

| Quant | Bytes | MiB | SHA-256 | Tensor types |
|---|---:|---:|---|---|
| Q8_0 | 1,246,253,408 | 1,188.52 | `3e2223fa4baf49896f1a58d0b73e02f5143b745d8deb71a01a6038b7994ce394` | 55 F32, 93 Q8_0 |
| Q6_K | 962,842,976 | 918.24 | `d1a14fa33fba75118c8262827bfcadc89d0c549d628ac243014df81a14d8f331` | 55 F32, 93 Q6_K |
| Q5_K_M | 843,354,464 | 804.29 | `353493ee33b842e6af3138ab250144706f4602b4240ba3eb9fdd9dce72b5b450` | 55 F32, 82 Q5_K, 11 Q6_K |
| Q4_K_M | 730,894,688 | 697.04 | `64310daeb8280eb65fb01fdf9c97e497b436201a6ad226b087378c7db3473de0` | 55 F32, 82 Q4_K, 11 Q6_K |
| Q2_K | 483,397,984 | 461.00 | `8cabe6523ed99809bd293c872672f40e0aeca3f121970ce383480fd04ee63c63` | 55 F32, 64 Q2_K, 28 Q3_K, 1 Q6_K |

The source BF16 SHA-256 remained
`3e39c292be1f2b2740c3e05a2cb9081260439033e1b05f601117c72c8a73b8e6`
before and after quantization.

## Structural gates

Every artifact:

- opens with the exact pinned GGUF parser;
- declares the requested `general.file_type`;
- retains all 148 tensors;
- retains the BF16 source architecture, 128,000 context metadata, model shape,
  RoPE base, tokenizer model, special-token IDs, and chat template;
- contains nonempty quantized tensors.

The run used `--network none`, a read-only root filesystem, a non-root user,
dropped Linux capabilities, `no-new-privileges`, bounded CPU/RAM/PIDs/tmpfs,
a single read-only BF16 file mount, and only `artifacts/gguf/phase2` as writable
model storage. GPU 0 was exposed with only NVIDIA `compute,utility` driver
capabilities because the pinned CUDA-linked executable needs the injected
driver library even though quantization work is CPU-side.

## Evidence

- `quantization.json`: inputs, pinned toolchain, isolation, durations, sizes,
  and hashes.
- `static-gguf-validation.json`: metadata gates and tensor inventories.
- `quantize-*.log`: raw `llama-quantize` output for each format.

No inference, perplexity, benchmark, throughput, VRAM, or long-context test was
run in this phase. In particular, Q2_K is an ultra-low-memory artifact, not a
quality recommendation. No deployment winner has been selected.

