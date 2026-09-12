[CmdletBinding()]
param()

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$workspace = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$sourceManifest = Get-Content -LiteralPath (
    Join-Path $workspace "reports\phase0\original-checkpoint-manifest.json"
) -Raw | ConvertFrom-Json
$modelDir = $sourceManifest.source_directory
if (-not (Test-Path -LiteralPath $modelDir -PathType Container)) {
    throw "Original checkpoint snapshot is missing: $modelDir"
}
$tokenizerOverride = (Resolve-Path (
    Join-Path $workspace "configs\phase1-tokenizer-config.json"
)).Path
$ggufDir = (Resolve-Path (Join-Path $workspace "artifacts\gguf\phase5")).Path
$reportDir = (Resolve-Path (Join-Path $workspace "reports\phase5")).Path
$validator = (Resolve-Path (Join-Path $PSScriptRoot "validate_phase5_baseline_gguf.py")).Path
$references = (Resolve-Path (
    Join-Path $workspace "reports\e1-differential\e1-hf-diff-20260829T220805Z\original-hf\generations.jsonl"
)).Path
$ggufName = "lfm2.5-1.2b-thinking-original-bf16.gguf"
$ggufPath = Join-Path $ggufDir $ggufName
$imageTag = "lfm25/llama-cpp:3018a11e-cuda128-sm120"

if (-not (Test-Path -LiteralPath $ggufPath)) {
    throw "Original baseline GGUF is missing: $ggufPath. Run convert_phase5_baseline.ps1 first."
}

$dockerArgs = @(
    "run"
    "--rm"
    "--pull", "never"
    "--platform", "linux/amd64"
    "--gpus", "device=0"
    "--network", "none"
    "--read-only"
    "--cap-drop", "ALL"
    "--security-opt", "no-new-privileges=true"
    "--pids-limit", "256"
    "--memory", "7g"
    "--memory-swap", "7g"
    "--cpus", "8"
    "--user", "65534:65534"
    "--tmpfs", "/tmp:rw,nosuid,size=1g,mode=1777"
    "--env", "CUDA_CACHE_PATH=/tmp/cuda-cache"
    "--env", "HOME=/tmp/home"
    "--env", "HF_HUB_OFFLINE=1"
    "--env", "TRANSFORMERS_OFFLINE=1"
    "--env", "PYTHONDONTWRITEBYTECODE=1"
    "--env", "NVIDIA_DRIVER_CAPABILITIES=compute,utility"
    "--mount", "type=bind,source=$modelDir,target=/hf-model,readonly"
    "--mount", "type=bind,source=$tokenizerOverride,target=/hf-model/tokenizer_config.json,readonly"
    "--mount", "type=bind,source=$ggufDir,target=/gguf,readonly"
    "--mount", "type=bind,source=$reportDir,target=/results"
    "--mount", "type=bind,source=$validator,target=/validate.py,readonly"
    "--mount", "type=bind,source=$references,target=/references.jsonl,readonly"
    "--entrypoint", "/opt/convert-venv/bin/python"
    $imageTag
    "/validate.py"
    "--references", "/references.jsonl"
    "--expected-model", "original_thinking"
    "--hf-model", "/hf-model"
    "--gguf", "/gguf/$ggufName"
    "--output", "/results/original-baseline-gguf-validation.json"
    "--server-log", "/results/original-baseline-gguf-server.log"
    "--max-new-tokens", "64"
)

$previousPreference = $ErrorActionPreference
$ErrorActionPreference = "Continue"
try {
    & docker @dockerArgs
    $dockerExitCode = $LASTEXITCODE
}
finally {
    $ErrorActionPreference = $previousPreference
}
if ($dockerExitCode -ne 0) {
    throw "Original baseline GGUF validation failed with exit code $dockerExitCode"
}

Write-Output "Original baseline GGUF validation passed."
Write-Output "Report: $(Join-Path $reportDir 'original-baseline-gguf-validation.json')"
