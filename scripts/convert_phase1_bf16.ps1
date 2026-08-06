[CmdletBinding()]
param(
    [switch]$Force
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$workspace = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$modelDir = (Resolve-Path (Join-Path $workspace "lfm2.5-model-merged")).Path
$tokenizerOverride = (Resolve-Path (
    Join-Path $workspace "configs\phase1-tokenizer-config.json"
)).Path
$artifactDir = Join-Path $workspace "artifacts\gguf"
$reportDir = Join-Path $workspace "reports\phase1"
$outputName = "lfm2.5-1.2b-thinking-kodcode-bf16.gguf"
$outputPath = Join-Path $artifactDir $outputName
$imageTag = "lfm25/llama-cpp:3018a11e-cuda128-sm120"

New-Item -ItemType Directory -Force -Path $artifactDir, $reportDir | Out-Null

if ((Test-Path -LiteralPath $outputPath) -and -not $Force) {
    throw "Output already exists: $outputPath. Pass -Force only to intentionally replace this generated artifact."
}

if (Test-Path -LiteralPath $outputPath) {
    Remove-Item -LiteralPath $outputPath
}

$imageInspect = (& docker image inspect $imageTag --format "{{json .}}") | ConvertFrom-Json
if ($LASTEXITCODE -ne 0) {
    throw "Pinned Phase 1 toolchain image is missing. Run build_phase1_toolchain.ps1 first."
}

$started = [DateTime]::UtcNow
$dockerArgs = @(
    "run"
    "--rm"
    "--pull", "never"
    "--platform", "linux/amd64"
    "--network", "none"
    "--read-only"
    "--cap-drop", "ALL"
    "--security-opt", "no-new-privileges=true"
    "--pids-limit", "256"
    "--memory", "12g"
    "--memory-swap", "12g"
    "--cpus", "8"
    "--user", "65534:65534"
    "--tmpfs", "/tmp:rw,nosuid,size=1g,mode=1777"
    "--env", "HF_HUB_OFFLINE=1"
    "--env", "TRANSFORMERS_OFFLINE=1"
    "--env", "PYTHONDONTWRITEBYTECODE=1"
    "--env", "TOKENIZERS_PARALLELISM=false"
    "--mount", "type=bind,source=$modelDir,target=/model,readonly"
    "--mount", "type=bind,source=$tokenizerOverride,target=/model/tokenizer_config.json,readonly"
    "--mount", "type=bind,source=$artifactDir,target=/output"
    "--entrypoint", "/opt/convert-venv/bin/python"
    $imageTag
    "/opt/llama-convert/convert_hf_to_gguf.py"
    "/model"
    "--outfile", "/output/$outputName"
    "--outtype", "bf16"
    "--model-name", "LFM2.5-1.2B-Thinking-KodCode-Merged"
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
    throw "BF16 GGUF conversion failed with exit code $dockerExitCode"
}
if (-not (Test-Path -LiteralPath $outputPath)) {
    throw "Converter exited successfully but did not create $outputPath"
}

$ended = [DateTime]::UtcNow
$artifact = Get-Item -LiteralPath $outputPath
$hash = (Get-FileHash -LiteralPath $outputPath -Algorithm SHA256).Hash.ToLowerInvariant()
$inputManifest = Join-Path $workspace "reports\phase0\merged-checkpoint-manifest.json"

$report = [ordered]@{
    schema_version = 1
    generated_at_utc = $ended.ToString("o")
    passed = $true
    started_at_utc = $started.ToString("o")
    duration_seconds = [Math]::Round(($ended - $started).TotalSeconds, 3)
    source_model = [ordered]@{
        path = $modelDir
        manifest_path = $inputManifest
        manifest_sha256 = (
            Get-FileHash -LiteralPath $inputManifest -Algorithm SHA256
        ).Hash.ToLowerInvariant()
    }
    toolchain = [ordered]@{
        image_tag = $imageTag
        image_id = $imageInspect.Id
        llama_cpp_commit = "3018a11e79e489b657dbb77c95694889ccff92df"
    }
    conversion = [ordered]@{
        outtype = "bf16"
        model_name = "LFM2.5-1.2B-Thinking-KodCode-Merged"
        tokenizer_compatibility_override = [ordered]@{
            path = $tokenizerOverride
            sha256 = (
                Get-FileHash -LiteralPath $tokenizerOverride -Algorithm SHA256
            ).Hash.ToLowerInvariant()
            source_tokenizer_config_unchanged = $true
            tokenizer_class = "PreTrainedTokenizerFast"
            reason = "Transformers 4.57.6 does not expose the Transformers 5 TokenizersBackend class name."
        }
        network = "none"
        root_filesystem = "read-only"
        input_mount = "read-only"
        output_mount = "artifacts/gguf only"
    }
    output = [ordered]@{
        path = $outputPath
        file_name = $outputName
        size_bytes = $artifact.Length
        sha256 = $hash
    }
}

$reportPath = Join-Path $reportDir "bf16-conversion.json"
$json = $report | ConvertTo-Json -Depth 10
[IO.File]::WriteAllText(
    $reportPath,
    $json + [Environment]::NewLine,
    [Text.UTF8Encoding]::new($false)
)

Write-Output "BF16 GGUF conversion passed."
Write-Output "Artifact: $outputPath"
Write-Output "SHA-256: $hash"
Write-Output "Report: $reportPath"
