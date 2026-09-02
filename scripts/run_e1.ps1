[CmdletBinding()]
param(
    [switch]$ValidateOnly,
    [switch]$SkipReplay,
    [string]$ReplayTaskId = "HumanEval/0",
    [string]$SuiteRelativePath = "benchmarks\suites\e1-humanevalplus-smoke-v2.json"
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

function Write-Step([string]$Message) {
    Write-Host "`n[E1] $Message" -ForegroundColor Cyan
}

function Resolve-RequiredFile([string]$Path, [string]$Label) {
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        throw "$Label not found: $Path"
    }
    return (Resolve-Path -LiteralPath $Path).Path
}

function Assert-HashAndSize(
    [string]$Path,
    [string]$ExpectedHash,
    [Int64]$ExpectedSize,
    [string]$Label
) {
    $item = Get-Item -LiteralPath $Path
    if ($item.Length -ne $ExpectedSize) {
        throw "$Label size mismatch. Expected $ExpectedSize bytes; got $($item.Length)."
    }
    $actualHash = (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($actualHash -ne $ExpectedHash.ToLowerInvariant()) {
        throw "$Label SHA-256 mismatch. Expected $ExpectedHash; got $actualHash."
    }
    Write-Host "  PASS $Label ($ExpectedSize bytes, SHA-256 $actualHash)"
}

function New-BindMount([string]$Source, [string]$Target, [switch]$ReadOnly) {
    $mount = "type=bind,source=$Source,target=$Target"
    if ($ReadOnly) {
        $mount += ",readonly"
    }
    return $mount
}

function Invoke-E1Container(
    [string]$Mode,
    [string]$ResultDirectory,
    [string]$RunId,
    [string]$ReferenceGenerations = ""
) {
    New-Item -ItemType Directory -Path $ResultDirectory -Force | Out-Null

    $mountArguments = @(
        "--mount", (New-BindMount $ModelPath "/model/model.gguf" -ReadOnly),
        "--mount", (New-BindMount $PromptPath "/inputs/prompts.jsonl" -ReadOnly),
        "--mount", (New-BindMount $TemplatePath "/inputs/prompt-template.txt" -ReadOnly),
        "--mount", (New-BindMount $SuitePath "/inputs/suite.json" -ReadOnly),
        "--mount", (New-BindMount $RunnerPath "/app/run_e1.py" -ReadOnly),
        "--mount", (New-BindMount $ReviewerPath "/app/review_e1.py" -ReadOnly),
        "--mount", (New-BindMount $HostEnvironmentPath "/inputs/host-environment.json" -ReadOnly),
        "--mount", (New-BindMount $ResultDirectory "/results")
    )
    if ($ReferenceGenerations) {
        $mountArguments += @(
            "--mount", (New-BindMount $ReferenceGenerations "/inputs/reference-generations.jsonl" -ReadOnly)
        )
    }

    $dockerArguments = @(
        "run", "--rm",
        "--pull", "never",
        "--platform", "linux/amd64",
        "--gpus", "device=0",
        "--network", "none",
        "--read-only",
        "--cap-drop", "ALL",
        "--security-opt", "no-new-privileges=true",
        "--pids-limit", "256",
        "--memory", "7g",
        "--memory-swap", "7g",
        "--cpus", "8",
        "--user", "65534:65534",
        "--tmpfs", "/tmp:rw,nosuid,nodev,noexec,size=1g,mode=1777"
    ) + $mountArguments + @(
        $ImageTag,
        "/usr/bin/python3", "/app/run_e1.py",
        "--suite", "/inputs/suite.json",
        "--prompts", "/inputs/prompts.jsonl",
        "--prompt-template", "/inputs/prompt-template.txt",
        "--model", "/model/model.gguf",
        "--results-dir", "/results",
        "--run-id", $RunId,
        "--mode", $Mode,
        "--host-environment", "/inputs/host-environment.json",
        "--image-tag", $ImageTag,
        "--image-id", $ImageId,
        "--replay-task-id", $ReplayTaskId
    )
    if ($ValidateOnly) {
        $dockerArguments += "--validate-only"
    }
    if ($ReferenceGenerations) {
        $dockerArguments += @(
            "--reference-generations", "/inputs/reference-generations.jsonl"
        )
    }

    Write-Host "  Container isolation: no network, read-only root, dropped capabilities, one GPU."
    Write-Host "  Only the model, suite, prompt template, three-prompt JSONL, runner, and result folder are mounted."
    & docker @dockerArguments
    if ($LASTEXITCODE -ne 0 -and -not ($Mode -eq "primary" -and $LASTEXITCODE -eq 4)) {
        throw "E1 $Mode container exited with code $LASTEXITCODE. See $ResultDirectory"
    }
    if ($Mode -eq "primary" -and $LASTEXITCODE -eq 4) {
        Write-Warning "Primary gate is red. The wrapper will inspect its gates and preserve the replay diagnostic if infrastructure is healthy."
    }
}

$Workspace = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$SuitePath = Resolve-RequiredFile (Join-Path $Workspace $SuiteRelativePath) "E1 suite"
$Suite = Get-Content -LiteralPath $SuitePath -Raw | ConvertFrom-Json
$PromptPath = Resolve-RequiredFile (Join-Path $Workspace $Suite.benchmark.generation_asset.path) "E1 prompt asset"
$TemplatePath = Resolve-RequiredFile (Join-Path $Workspace $Suite.prompt.path) "Prompt template"
$PromptManifestPath = Resolve-RequiredFile (Join-Path $Workspace $Suite.prompt.manifest_path) "Prompt manifest"
$ModelPath = Resolve-RequiredFile (Join-Path $Workspace $Suite.model.path) "BF16 model"
$RunnerPath = Resolve-RequiredFile (Join-Path $Workspace "scripts\run_e1.py") "E1 Python runner"
$ReviewerPath = Resolve-RequiredFile (Join-Path $Workspace "scripts\review_e1.py") "E1 behavior reviewer"
$ExtractorManifestPath = Resolve-RequiredFile (Join-Path $Workspace $Suite.extractor.manifest_path) "Extractor manifest"
$ExtractorPath = Resolve-RequiredFile (Join-Path $Workspace $Suite.extractor.implementation_path) "Extractor implementation"
$PythonPath = Resolve-RequiredFile (Join-Path $Workspace ".venv\Scripts\python.exe") "Workspace Python"
$ImageTag = $Suite.runtime.toolchain_image

Write-Step "Preflight: verify Docker is responding and the exact pinned image is local."
$dockerServer = (docker info --format "Docker server {{.ServerVersion}}").Trim()
if ($LASTEXITCODE -ne 0) {
    throw "Docker Engine is not responding."
}
Write-Host $dockerServer
$ImageId = (docker image inspect $ImageTag --format "{{.Id}}").Trim()
if ($LASTEXITCODE -ne 0 -or -not $ImageId) {
    throw "Pinned image is not available locally: $ImageTag"
}
if ($ImageId -ne $Suite.runtime.toolchain_image_id) {
    throw "Pinned image ID mismatch. Expected $($Suite.runtime.toolchain_image_id); got $ImageId."
}
Write-Host "  PASS image tag: $ImageTag"
Write-Host "  PASS image ID:  $ImageId"

Write-Step "Preflight: hash the frozen inputs before any model request."
Assert-HashAndSize $PromptPath $Suite.benchmark.generation_asset.sha256 $Suite.benchmark.generation_asset.size_bytes "three-prompt JSONL"
Assert-HashAndSize $TemplatePath $Suite.prompt.sha256 $Suite.prompt.size_bytes "prompt template"
Assert-HashAndSize $PromptManifestPath $Suite.prompt.manifest_sha256 $Suite.prompt.manifest_size_bytes "prompt manifest"
Assert-HashAndSize $ExtractorManifestPath $Suite.extractor.manifest_sha256 $Suite.extractor.manifest_size_bytes "extractor manifest"
Assert-HashAndSize $ExtractorPath $Suite.extractor.implementation_sha256 $Suite.extractor.implementation_size_bytes "extractor implementation"
Assert-HashAndSize $ModelPath $Suite.model.sha256 $Suite.model.size_bytes "merged BF16 GGUF"

$SessionId = "e1-v1-smoke-" + (Get-Date).ToUniversalTime().ToString("yyyyMMddTHHmmssZ")
$SessionDirectory = Join-Path $Workspace "reports\e1\$SessionId"
$PrimaryDirectory = Join-Path $SessionDirectory "primary"
$ReplayDirectory = Join-Path $SessionDirectory "replay"
New-Item -ItemType Directory -Path $SessionDirectory -Force | Out-Null
$HostEnvironmentPath = Join-Path $SessionDirectory "host-environment.json"
$hostEnvironment = [ordered]@{
    captured_at_utc = (Get-Date).ToUniversalTime().ToString("o")
    operating_system = [System.Environment]::OSVersion.VersionString
    powershell = $PSVersionTable.PSVersion.ToString()
    machine = $env:COMPUTERNAME
    processor_count = [System.Environment]::ProcessorCount
    docker_server_version = (docker version --format "{{.Server.Version}}").Trim()
    docker_image_tag = $ImageTag
    docker_image_id = $ImageId
}
[System.IO.File]::WriteAllText(
    $HostEnvironmentPath,
    (($hostEnvironment | ConvertTo-Json -Depth 5) + "`n"),
    [System.Text.UTF8Encoding]::new($false)
)
$HostEnvironmentPath = (Resolve-Path -LiteralPath $HostEnvironmentPath).Path

Write-Step "Primary smoke: start a fresh restricted container and generate 7 evidence records."
Invoke-E1Container "primary" $PrimaryDirectory "$SessionId-primary"

if ($ValidateOnly) {
    Write-Step "Validation-only run passed. No model requests were sent."
    Write-Host "Evidence: $PrimaryDirectory"
    exit 0
}

$primaryManifestPath = Join-Path $PrimaryDirectory "run-manifest.json"
$primaryManifest = Get-Content -LiteralPath $primaryManifestPath -Raw | ConvertFrom-Json
$failedPrimaryGates = @(
    $primaryManifest.gates.PSObject.Properties |
        Where-Object { -not $_.Value.passed } |
        ForEach-Object { $_.Name }
)
$nonBehaviorFailures = @($failedPrimaryGates | Where-Object { $_ -ne "prompt_integrity_behavior" })
if ($nonBehaviorFailures.Count -gt 0) {
    throw "Primary infrastructure/mechanical gates failed: $($nonBehaviorFailures -join ', '). Review $primaryManifestPath"
}
$PrimaryBehaviorPassed = [bool]$primaryManifest.gates.prompt_integrity_behavior.passed

Write-Step "Extraction: derive three EvalPlus samples without importing or executing generated Python."
$primaryGenerationsPath = Resolve-RequiredFile (Join-Path $PrimaryDirectory "generations.jsonl") "Primary generation evidence"
$samplesPath = Join-Path $PrimaryDirectory "samples.jsonl"
$extractionReportPath = Join-Path $PrimaryDirectory "extraction-report.json"
& $PythonPath $ExtractorPath `
    --generations $primaryGenerationsPath `
    --prompt-manifest $PromptManifestPath `
    --samples-output $samplesPath `
    --report-output $extractionReportPath
$extractionExitCode = $LASTEXITCODE
if ($extractionExitCode -notin @(0, 4)) {
    throw "Extractor harness exited with code $extractionExitCode. Review $extractionReportPath"
}
$extractionReport = Get-Content -LiteralPath $extractionReportPath -Raw | ConvertFrom-Json
if ($extractionReport.selected_record_count -ne $Suite.acceptance_gate.selected_extraction_records) {
    throw "Extractor selected $($extractionReport.selected_record_count) records; expected $($Suite.acceptance_gate.selected_extraction_records)."
}
if ($extractionReport.generated_programs_executed) {
    throw "Extractor safety invariant failed: generated program execution was reported."
}
$ExtractionPassed = [bool]$extractionReport.passed
if ($ExtractionPassed) {
    if ($extractionReport.exported_sample_count -ne $Suite.acceptance_gate.unique_exported_samples) {
        throw "Extractor exported $($extractionReport.exported_sample_count) samples; expected $($Suite.acceptance_gate.unique_exported_samples)."
    }
    Write-Host "  PASS three scorer-ready samples were derived; generated code was not executed."
} else {
    Write-Warning "Extraction is red. No partial samples file was published; review $extractionReportPath"
}

if (-not $SkipReplay) {
    Write-Step "Determinism check: use a second fresh container to replay only $ReplayTaskId."
    $referencePath = Resolve-RequiredFile (Join-Path $PrimaryDirectory "generations.jsonl") "Primary generation evidence"
    Invoke-E1Container "replay" $ReplayDirectory "$SessionId-replay" $referencePath
    $replayManifestPath = Join-Path $ReplayDirectory "run-manifest.json"
    $replayManifest = Get-Content -LiteralPath $replayManifestPath -Raw | ConvertFrom-Json
    if (-not $replayManifest.passed) {
        throw "Replay gate failed. Review $replayManifestPath"
    }
}

Write-Step "Smoke-test execution is complete. The run executed and scored zero generated programs."
Write-Host "Primary evidence: $PrimaryDirectory"
Write-Host "Extraction report: $extractionReportPath"
if ($ExtractionPassed) {
    Write-Host "Derived samples:   $samplesPath"
}
if (-not $SkipReplay) {
    Write-Host "Replay evidence:  $ReplayDirectory"
}
$RedOutcomeGates = @()
if (-not $PrimaryBehaviorPassed) {
    $RedOutcomeGates += "prompt-integrity behavior"
}
if (-not $ExtractionPassed) {
    $RedOutcomeGates += "solution extraction"
}
if ($RedOutcomeGates.Count -gt 0) {
    throw "Infrastructure and replay completed, but these outcome gates are red: $($RedOutcomeGates -join ', '). Evidence: $PrimaryDirectory"
}
Write-Host "Overall E1 gate: PASS" -ForegroundColor Green
