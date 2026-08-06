[CmdletBinding()]
param()

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$workspace = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$modelDir = (Resolve-Path (Join-Path $workspace "lfm2.5-model-merged")).Path
$ggufDir = (Resolve-Path (Join-Path $workspace "artifacts\gguf")).Path
$reportDir = (Resolve-Path (Join-Path $workspace "reports\phase1")).Path
$inspector = (Resolve-Path (
    Join-Path $PSScriptRoot "inspect_phase1_gguf.py"
)).Path
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
    "--mount", "type=bind,source=$modelDir,target=/hf-model,readonly"
    "--mount", "type=bind,source=$ggufDir,target=/gguf,readonly"
    "--mount", "type=bind,source=$reportDir,target=/results"
    "--mount", "type=bind,source=$inspector,target=/phase1_gguf_inspector.py,readonly"
    "--entrypoint", "/opt/convert-venv/bin/python"
    $imageTag
    "/phase1_gguf_inspector.py"
    "--gguf", "/gguf/$ggufName"
    "--source-config", "/hf-model/config.json"
    "--source-chat-template", "/hf-model/chat_template.jinja"
    "--output", "/results/bf16-gguf-metadata.json"
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
    throw "BF16 GGUF metadata inspection failed with exit code $dockerExitCode"
}

Write-Output "BF16 GGUF metadata inspection passed."
Write-Output "Report: $(Join-Path $reportDir 'bf16-gguf-metadata.json')"
