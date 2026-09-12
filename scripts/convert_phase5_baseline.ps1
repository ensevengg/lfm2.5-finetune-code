[CmdletBinding()]
param(
    [switch]$Force
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$workspace = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$sourceManifestPath = Join-Path $workspace "reports\phase0\original-checkpoint-manifest.json"
$sourceManifest = Get-Content -LiteralPath $sourceManifestPath -Raw | ConvertFrom-Json
$modelDir = $sourceManifest.source_directory
if (-not (Test-Path -LiteralPath $modelDir -PathType Container)) {
    throw "Original checkpoint snapshot from the Phase 0 manifest is missing: $modelDir"
}
$tokenizerOverride = (Resolve-Path (
    Join-Path $workspace "configs\phase1-tokenizer-config.json"
)).Path
$artifactDir = Join-Path $workspace "artifacts\gguf\phase5"
$reportDir = Join-Path $workspace "reports\phase5"
$outputName = "lfm2.5-1.2b-thinking-original-bf16.gguf"
$outputPath = Join-Path $artifactDir $outputName
$imageTag = "lfm25/llama-cpp:3018a11e-cuda128-sm120"

New-Item -ItemType Directory -Force -Path $artifactDir, $reportDir | Out-Null

if ((Test-Path -LiteralPath $outputPath) -and -not $Force) {
    throw "Output already exists: $outputPath. Pass -Force only to intentionally replace this generated artifact."
}

if (Test-Path -LiteralPath $outputPath) {
    Remove-Item -LiteralPath $outputPath
}

# Fail before a multi-minute conversion if the source weights drifted from the
# hash-frozen Phase 0 manifest.
$weightsItem = Get-Item -LiteralPath (Join-Path $modelDir "model.safetensors")
$weightsHash = (Get-FileHash -LiteralPath $weightsItem.FullName -Algorithm SHA256).Hash.ToLowerInvariant()
$manifestWeights = @($sourceManifest.files | Where-Object { $_.path -eq "model.safetensors" })[0]
if ($weightsItem.Length -ne $manifestWeights.size_bytes -or $weightsHash -ne $manifestWeights.sha256.ToLowerInvariant()) {
    throw "Original checkpoint weights drifted from the Phase 0 manifest. Got $($weightsItem.Length) bytes, SHA-256 $weightsHash"
}
Write-Host "PASS original weights match the Phase 0 manifest"

$imageInspect = (& docker image inspect $imageTag --format "{{json .}}") | ConvertFrom-Json
if ($LASTEXITCODE -ne 0) {
    throw "Pinned toolchain image is missing. Run build_phase1_toolchain.ps1 first."
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
    "--model-name", "LFM2.5-1.2B-Thinking-Original"
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
    throw "Original baseline BF16 GGUF conversion failed with exit code $dockerExitCode"
}
if (-not (Test-Path -LiteralPath $outputPath)) {
    throw "Converter exited successfully but did not create $outputPath"
}

$ended = [DateTime]::UtcNow
$artifact = Get-Item -LiteralPath $outputPath
$hash = (Get-FileHash -LiteralPath $outputPath -Algorithm SHA256).Hash.ToLowerInvariant()

$report = [ordered]@{
    schema_version = 1
    generated_at_utc = $ended.ToString("o")
    passed = $true
    started_at_utc = $started.ToString("o")
    duration_seconds = [Math]::Round(($ended - $started).TotalSeconds, 3)
    source_model = [ordered]@{
        repo_id = $sourceManifest.repo_id
        revision = $sourceManifest.revision
        path = $modelDir
        manifest_path = $sourceManifestPath
        manifest_sha256 = (
            Get-FileHash -LiteralPath $sourceManifestPath -Algorithm SHA256
        ).Hash.ToLowerInvariant()
        weights_size_bytes = $weightsItem.Length
        weights_sha256 = $weightsHash
    }
    toolchain = [ordered]@{
        image_tag = $imageTag
        image_id = $imageInspect.Id
        llama_cpp_commit = "3018a11e79e489b657dbb77c95694889ccff92df"
    }
    conversion = [ordered]@{
        outtype = "bf16"
        model_name = "LFM2.5-1.2B-Thinking-Original"
        tokenizer_compatibility_override = [ordered]@{
            path = $tokenizerOverride
            sha256 = (
                Get-FileHash -LiteralPath $tokenizerOverride -Algorithm SHA256
            ).Hash.ToLowerInvariant()
            tokenizer_class = "PreTrainedTokenizerFast"
            reason = "Transformers 4.57.6 does not expose the Transformers 5 TokenizersBackend class name."
        }
        network = "none"
        root_filesystem = "read-only"
        input_mount = "read-only"
        output_mount = "artifacts/gguf/phase5 only"
    }
    output = [ordered]@{
        path = $outputPath
        file_name = $outputName
        size_bytes = $artifact.Length
        sha256 = $hash
    }
}

$reportPath = Join-Path $reportDir "original-baseline-conversion.json"
$json = $report | ConvertTo-Json -Depth 10
[IO.File]::WriteAllText(
    $reportPath,
    $json + [Environment]::NewLine,
    [Text.UTF8Encoding]::new($false)
)

Write-Output "Original baseline BF16 GGUF conversion passed."
Write-Output "Artifact: $outputPath"
Write-Output "SHA-256: $hash"
Write-Output "Report: $reportPath"
