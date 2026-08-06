# Phase 0 result

Generated: 2026-07-29T21:08:12.449249+00:00

## Recovered provenance

- Base model: `LiquidAI/LFM2.5-1.2B-Thinking@95053d21d8e0b7ca99421a2127ae39c64f685ff3`
- Adapter: `enseven/lfm-2.5-think-code@e01354fecc52e62b9ca86399da10e9e40ebf51e9`
- Dataset snapshot available: `e4d2d9230fe875853fa1d28cc4fd9a04d4b30cbe`
- Provenance method: local Hugging Face `refs/main` and snapshot directories;
  the phase made no network request.

## Validation gates

- PASS: `base_revision_recovered`
- PASS: `adapter_revision_recovered`
- PASS: `merged_parameter_count_is_1_170_340_608`
- PASS: `original_and_merged_tensor_sets_match`
- PASS: `merge_changed_weight_tensors`
- PASS: `token_ids_and_chat_template_match_original`
- PASS: `offline_original_load_and_generation`
- PASS: `original_first_step_logits_are_finite`
- PASS: `offline_merged_load_and_generation`
- PASS: `merged_first_step_logits_are_finite`
- PASS: `smoke_outputs_show_behavior_change`
- PASS: `docker_linux_engine_reachable`
- PASS: `restricted_cuda_container_smoke_passed`

Overall Phase 0 checkpoint gate: **PASS**

## Host readiness

- Docker server reachable: **yes**
- CUDA available to PyTorch: **yes**
- Restricted CUDA GPU-container smoke test: **pass**

## Evidence files

- `merged-checkpoint-manifest.json`
- `original-checkpoint-manifest.json`
- `adapter-manifest.json`
- `provenance.json`
- `machine-manifest.json`
- `validation.json`
- `container-readiness.json`
- `../../configs/evaluation-lock.json` (candidate immutable tool revisions)

The original and merged checkpoints were read only. No model file was edited.
