[CmdletBinding()]
param()

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$workspace = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$modelDir = (Resolve-Path (Join-Path $workspace "lfm2.5-model-merged")).Path
$tokenizerOverride = (Resolve-Path (
    Join-Path $workspace "configs\phase1-tokenizer-config.json"
)).Path
$ggufDir = (Resolve-Path (Join-Path $workspace "artifacts\gguf")).Path
$reportDir = (Resolve-Path (Join-Path $workspace "reports\phase1")).Path
$validator = (Resolve-Path (Join-Path $PSScriptRoot "validate_phase1_gguf.py")).Path
$phase0Validation = (
    Resolve-Path (Join-Path $workspace "reports\phase0\validation.json")
).Path
$ggufName = "lfm2.5-1.2b-thinking-kodcode-bf16.gguf"
$ggufPath = Join-Path $ggufDir $ggufName
$imageTag = "lfm25/llama-cpp:3018a11e-cuda128-sm120"

if (-not (Test-Path -LiteralPath $ggufPath)) {
    throw "BF16 GGUF is missing: $ggufPath"
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
    "--mount", "type=bind,source=$phase0Validation,target=/phase0-validation.json,readonly"
    "--entrypoint", "/opt/convert-venv/bin/python"
    $imageTag
    "/validate.py"
    "--hf-model", "/hf-model"
    "--gguf", "/gguf/$ggufName"
    "--phase0-validation", "/phase0-validation.json"
    "--output", "/results/bf16-gguf-validation.json"
    "--server-log", "/results/bf16-gguf-server.log"
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
    throw "BF16 GGUF validation failed with exit code $dockerExitCode"
}

Write-Output "BF16 GGUF validation passed."
Write-Output "Report: $(Join-Path $reportDir 'bf16-gguf-validation.json')"
