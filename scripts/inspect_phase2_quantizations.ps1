[CmdletBinding()]
param()

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$workspace = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$sourcePath = (Resolve-Path (
    Join-Path $workspace "artifacts\gguf\lfm2.5-1.2b-thinking-kodcode-bf16.gguf"
)).Path
$quantDir = (Resolve-Path (Join-Path $workspace "artifacts\gguf\phase2")).Path
$reportDir = Join-Path $workspace "reports\phase2"
$inspector = (Resolve-Path (
    Join-Path $PSScriptRoot "inspect_phase2_quantizations.py"
)).Path
$imageTag = "lfm25/llama-cpp:3018a11e-cuda128-sm120"

New-Item -ItemType Directory -Force -Path $reportDir | Out-Null

$dockerArgs = @(
    "run"
    "--rm"
    "--pull", "never"
    "--platform", "linux/amd64"
    "--network", "none"
    "--read-only"
    "--cap-drop", "ALL"
    "--security-opt", "no-new-privileges=true"
    "--pids-limit", "64"
    "--memory", "2g"
    "--memory-swap", "2g"
    "--cpus", "2"
    "--user", "65534:65534"
    "--tmpfs", "/tmp:rw,noexec,nosuid,size=64m,mode=1777"
    "--env", "PYTHONDONTWRITEBYTECODE=1"
    "--env", "PYTHONPATH=/opt/llama-convert/gguf-py"
    "--mount", "type=bind,source=$sourcePath,target=/source/model-bf16.gguf,readonly"
    "--mount", "type=bind,source=$quantDir,target=/quants,readonly"
    "--mount", "type=bind,source=$reportDir,target=/results"
    "--mount", "type=bind,source=$inspector,target=/phase2_gguf_inspector.py,readonly"
    "--entrypoint", "/opt/convert-venv/bin/python"
    $imageTag
    "/phase2_gguf_inspector.py"
    "--source", "/source/model-bf16.gguf"
    "--quant-dir", "/quants"
    "--output", "/results/static-gguf-validation.json"
    "--expected-tensors", "148"
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
    throw "Phase 2 static GGUF inspection failed with exit code $dockerExitCode"
}

Write-Output "Phase 2 static GGUF inspection passed."
Write-Output "Report: $(Join-Path $reportDir 'static-gguf-validation.json')"

