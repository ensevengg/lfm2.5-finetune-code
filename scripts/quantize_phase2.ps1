[CmdletBinding()]
param(
    [switch]$Force
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$workspace = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$sourcePath = (Resolve-Path (
    Join-Path $workspace "artifacts\gguf\lfm2.5-1.2b-thinking-kodcode-bf16.gguf"
)).Path
$outputDir = Join-Path $workspace "artifacts\gguf\phase2"
$reportDir = Join-Path $workspace "reports\phase2"
$reportPath = Join-Path $reportDir "quantization.json"
$imageTag = "lfm25/llama-cpp:3018a11e-cuda128-sm120"
$expectedImageId = "sha256:550d53ae97c4d83a6c0fba17cd8f532aa6f775e9498d2452fa6f04e33707b4b0"
$expectedSourceHash = "3e39c292be1f2b2740c3e05a2cb9081260439033e1b05f601117c72c8a73b8e6"
$llamaCppCommit = "3018a11e79e489b657dbb77c95694889ccff92df"
$threads = 8

$formats = @(
    [ordered]@{ name = "Q8_0"; file = "lfm2.5-1.2b-thinking-kodcode-q8_0.gguf" },
    [ordered]@{ name = "Q6_K"; file = "lfm2.5-1.2b-thinking-kodcode-q6_k.gguf" },
    [ordered]@{ name = "Q5_K_M"; file = "lfm2.5-1.2b-thinking-kodcode-q5_k_m.gguf" },
    [ordered]@{ name = "Q4_K_M"; file = "lfm2.5-1.2b-thinking-kodcode-q4_k_m.gguf" },
    [ordered]@{ name = "Q2_K"; file = "lfm2.5-1.2b-thinking-kodcode-q2_k.gguf" }
)

New-Item -ItemType Directory -Force -Path $outputDir, $reportDir | Out-Null

$sourceHashBefore = (Get-FileHash -LiteralPath $sourcePath -Algorithm SHA256).Hash.ToLowerInvariant()
if ($sourceHashBefore -ne $expectedSourceHash) {
    throw "BF16 source hash mismatch. Expected $expectedSourceHash, got $sourceHashBefore"
}

$previousPreference = $ErrorActionPreference
$ErrorActionPreference = "Continue"
try {
    $imageId = (& docker image inspect $imageTag --format "{{.Id}}" 2>&1 | Out-String).Trim()
    $inspectExitCode = $LASTEXITCODE
}
finally {
    $ErrorActionPreference = $previousPreference
}
if ($inspectExitCode -ne 0) {
    throw "Pinned Phase 1 toolchain image is missing: $imageTag"
}
if ($imageId -ne $expectedImageId) {
    throw "Pinned image ID mismatch. Expected $expectedImageId, got $imageId"
}

foreach ($format in $formats) {
    $finalPath = Join-Path $outputDir $format.file
    $partialPath = "$finalPath.partial"
    if ((Test-Path -LiteralPath $finalPath) -and -not $Force) {
        throw "Output already exists: $finalPath. Pass -Force only to intentionally replace generated artifacts."
    }
    if (Test-Path -LiteralPath $partialPath) {
        Remove-Item -LiteralPath $partialPath -Force
    }
    if ((Test-Path -LiteralPath $finalPath) -and $Force) {
        Remove-Item -LiteralPath $finalPath -Force
    }
}

$runStarted = [DateTime]::UtcNow
$artifacts = @()
foreach ($format in $formats) {
    $finalPath = Join-Path $outputDir $format.file
    $partialPath = "$finalPath.partial"
    $logPath = Join-Path $reportDir ("quantize-{0}.log" -f $format.name.ToLowerInvariant())
    $started = [DateTime]::UtcNow
    Write-Output "Quantizing $($format.name) from the verified BF16 source..."

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
        "--pids-limit", "128"
        "--memory", "8g"
        "--memory-swap", "8g"
        "--cpus", "8"
        "--user", "65534:65534"
        "--tmpfs", "/tmp:rw,noexec,nosuid,size=256m,mode=1777"
        "--env", "NVIDIA_DRIVER_CAPABILITIES=compute,utility"
        "--mount", "type=bind,source=$sourcePath,target=/input/model-bf16.gguf,readonly"
        "--mount", "type=bind,source=$outputDir,target=/output"
        "--entrypoint", "/opt/llama/bin/llama-quantize"
        $imageTag
        "/input/model-bf16.gguf"
        "/output/$($format.file).partial"
        $format.name
        "$threads"
    )

    $previousPreference = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        $quantizerOutput = (& docker @dockerArgs 2>&1 | Out-String)
        $dockerExitCode = $LASTEXITCODE
    }
    finally {
        $ErrorActionPreference = $previousPreference
    }
    [IO.File]::WriteAllText(
        $logPath,
        $quantizerOutput,
        [Text.UTF8Encoding]::new($false)
    )
    if ($dockerExitCode -ne 0) {
        throw "$($format.name) quantization failed with exit code $dockerExitCode. See $logPath"
    }
    if (-not (Test-Path -LiteralPath $partialPath)) {
        throw "$($format.name) exited successfully but did not create $partialPath"
    }
    Move-Item -LiteralPath $partialPath -Destination $finalPath

    $ended = [DateTime]::UtcNow
    $artifact = Get-Item -LiteralPath $finalPath
    $artifactHash = (Get-FileHash -LiteralPath $finalPath -Algorithm SHA256).Hash.ToLowerInvariant()
    $artifacts += [ordered]@{
        quantization = $format.name
        file_name = $format.file
        path = $finalPath
        size_bytes = $artifact.Length
        sha256 = $artifactHash
        started_at_utc = $started.ToString("o")
        duration_seconds = [Math]::Round(($ended - $started).TotalSeconds, 3)
        log_path = $logPath
    }
    Write-Output ("Completed {0}: {1:N2} MiB, SHA-256 {2}" -f $format.name, ($artifact.Length / 1MB), $artifactHash)
}

$runEnded = [DateTime]::UtcNow
$sourceHashAfter = (Get-FileHash -LiteralPath $sourcePath -Algorithm SHA256).Hash.ToLowerInvariant()
if ($sourceHashAfter -ne $sourceHashBefore) {
    throw "BF16 source changed during quantization. Before: $sourceHashBefore; after: $sourceHashAfter"
}

$report = [ordered]@{
    schema_version = 1
    generated_at_utc = $runEnded.ToString("o")
    passed = $true
    status = "artifacts-generated-unbenchmarked"
    started_at_utc = $runStarted.ToString("o")
    duration_seconds = [Math]::Round(($runEnded - $runStarted).TotalSeconds, 3)
    source = [ordered]@{
        path = $sourcePath
        format = "BF16 GGUF"
        sha256_before = $sourceHashBefore
        sha256_after = $sourceHashAfter
        mounted_read_only = $true
    }
    toolchain = [ordered]@{
        image_tag = $imageTag
        image_id = $imageId
        llama_cpp_commit = $llamaCppCommit
        executable = "/opt/llama/bin/llama-quantize"
        threads = $threads
    }
    method = [ordered]@{
        requantized = $false
        importance_matrix = $null
        rationale = "Every artifact was quantized independently from the verified BF16 GGUF; no validated importance matrix was available."
    }
    isolation = [ordered]@{
        network = "none"
        root_filesystem = "read-only"
        capabilities = "all dropped"
        no_new_privileges = $true
        input_mount = "single BF16 file, read-only"
        output_mount = "artifacts/gguf/phase2 only"
        host_gpu_access = "NVIDIA device 0; compute and utility driver capabilities only"
    }
    artifacts = $artifacts
    evaluation = [ordered]@{
        performed = $false
        winner_selected = $false
        note = "Quality, speed, VRAM, long-context, and benchmark evaluation are intentionally deferred."
    }
}
[IO.File]::WriteAllText(
    $reportPath,
    (($report | ConvertTo-Json -Depth 10) + [Environment]::NewLine),
    [Text.UTF8Encoding]::new($false)
)

Write-Output "Phase 2 quantization artifact generation passed."
Write-Output "Report: $reportPath"

