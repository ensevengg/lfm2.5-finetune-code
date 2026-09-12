[CmdletBinding()]
param(
    [string]$RowAlias = "",
    [switch]$SkipExtraction,
    [string]$SuiteRelativePath = "benchmarks\suites\e3-humanevalplus-pilot-v1.json",
    [string]$SessionPrefix = "e3-pilot"
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

function Write-Step([string]$Message) {
    Write-Host "`n[E3] $Message" -ForegroundColor Cyan
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
    Write-Host "  PASS $Label ($ExpectedSize bytes)"
}

function New-BindMount([string]$Source, [string]$Target, [switch]$ReadOnly) {
    $mount = "type=bind,source=$Source,target=$Target"
    if ($ReadOnly) {
        $mount += ",readonly"
    }
    return $mount
}

$Workspace = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$SuitePath = Resolve-RequiredFile (Join-Path $Workspace $SuiteRelativePath) "Generation suite"
$Suite = Get-Content -LiteralPath $SuitePath -Raw | ConvertFrom-Json
$DatasetPath = Resolve-RequiredFile (Join-Path $Workspace $Suite.dataset.path) "HumanEvalPlus dataset"
$TemplatePath = Resolve-RequiredFile (Join-Path $Workspace $Suite.prompt.template_path) "Prompt template"
$PromptManifestPath = Resolve-RequiredFile (Join-Path $Workspace $Suite.prompt.manifest_path) "Prompt manifest"
$RunnerPath = Resolve-RequiredFile (Join-Path $Workspace "scripts\run_e3.py") "E3 runner"
$RunnerLibPath = Resolve-RequiredFile (Join-Path $Workspace "scripts\e3_runner_lib.py") "E3 runner library"
$ExtractorPath = Resolve-RequiredFile (Join-Path $Workspace "scripts\export_evalplus_samples_v2.py") "Extractor v2"
$PythonPath = Resolve-RequiredFile (Join-Path $Workspace ".venv\Scripts\python.exe") "Workspace Python"
$ImageTag = $Suite.runtime.toolchain_image

Write-Step "Preflight: verify Docker and the exact pinned toolchain image."
$dockerServer = (docker info --format "Docker server {{.ServerVersion}}").Trim()
if ($LASTEXITCODE -ne 0) {
    throw "Docker Engine is not responding."
}
Write-Host $dockerServer
$imageInspect = (& docker image inspect $ImageTag) | ConvertFrom-Json
if ($LASTEXITCODE -ne 0 -or -not $imageInspect) {
    throw "Pinned toolchain image is not available locally: $ImageTag"
}
$ImageId = $imageInspect[0].Id
if ($ImageId -ne $Suite.runtime.toolchain_image_id) {
    throw "Pinned image ID mismatch. Expected $($Suite.runtime.toolchain_image_id); got $ImageId."
}
Write-Host "  PASS image $ImageTag"

Write-Step "Preflight: hash the frozen inputs before any model request."
Assert-HashAndSize $DatasetPath $Suite.dataset.sha256 $Suite.dataset.size_bytes "HumanEvalPlus dataset"
Assert-HashAndSize $TemplatePath $Suite.prompt.template_sha256 $Suite.prompt.template_size_bytes "prompt template"
Assert-HashAndSize $PromptManifestPath $Suite.prompt.manifest_sha256 $Suite.prompt.manifest_size_bytes "prompt manifest"

$SuiteTaskCount = 0
foreach ($TaskIdProperty in @("pilot_task_ids", "sealed_task_ids")) {
    if ($Suite.PSObject.Properties.Name -contains $TaskIdProperty) {
        $SuiteTaskCount += @($Suite.$TaskIdProperty).Count
    }
}
$SessionId = "$SessionPrefix-" + (Get-Date).ToUniversalTime().ToString("yyyyMMddTHmmssZ")
$SessionDirectory = Join-Path $Workspace "reports\e4\$SessionId"
New-Item -ItemType Directory -Path $SessionDirectory -Force | Out-Null
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
    (Join-Path $SessionDirectory "host-environment.json"),
    (($hostEnvironment | ConvertTo-Json -Depth 5) + "`n"),
    [System.Text.UTF8Encoding]::new($false)
)

$AllRows = @($Suite.model_rows)
if ($RowAlias) {
    $AllRows = @($AllRows | Where-Object { $_.alias -eq $RowAlias })
    if ($AllRows.Count -eq 0) {
        throw "Row alias is not in the suite: $RowAlias"
    }
}

$RowResults = @()
foreach ($Row in $AllRows) {
    $alias = $Row.alias
    Write-Step "Row $alias — verify artifact identity."
    $ModelPath = Resolve-RequiredFile (Join-Path $Workspace ($Row.path -replace "/", "\")) "Model artifact $alias"
    Assert-HashAndSize $ModelPath $Row.sha256 $Row.size_bytes "GGUF $alias"

    $RowDirectory = Join-Path $SessionDirectory $alias
    $ResultsDirectory = Join-Path $RowDirectory "results"
    New-Item -ItemType Directory -Path $ResultsDirectory -Force | Out-Null
    $RunId = "$SessionId-$alias"

    Write-Step "Row $alias — generate $SuiteTaskCount tasks in a restricted container."
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
        "--tmpfs", "/tmp:rw,nosuid,nodev,noexec,size=1g,mode=1777",
        "--env", "CUDA_CACHE_PATH=/tmp/cuda-cache",
        "--env", "HOME=/tmp/home",
        "--env", "HF_HUB_OFFLINE=1",
        "--env", "TRANSFORMERS_OFFLINE=1",
        "--env", "PYTHONDONTWRITEBYTECODE=1",
        "--env", "NVIDIA_DRIVER_CAPABILITIES=compute,utility",
        "--mount", (New-BindMount $ModelPath "/model/model.gguf" -ReadOnly),
        "--mount", (New-BindMount $DatasetPath "/inputs/dataset.jsonl" -ReadOnly),
        "--mount", (New-BindMount $TemplatePath "/inputs/prompt-template.txt" -ReadOnly),
        "--mount", (New-BindMount $SuitePath "/inputs/suite.json" -ReadOnly),
        "--mount", (New-BindMount $RunnerPath "/app/run_e3.py" -ReadOnly),
        "--mount", (New-BindMount $RunnerLibPath "/app/e3_runner_lib.py" -ReadOnly),
        "--mount", (New-BindMount $ResultsDirectory "/results"),
        $ImageTag,
        "/usr/bin/python3", "/app/run_e3.py",
        "--suite", "/inputs/suite.json",
        "--dataset", "/inputs/dataset.jsonl",
        "--prompt-template", "/inputs/prompt-template.txt",
        "--model", "/model/model.gguf",
        "--model-alias", $alias,
        "--model-sha256", $Row.sha256,
        "--results-dir", "/results",
        "--run-id", $RunId
    )

    & docker @dockerArguments
    $RowExitCode = $LASTEXITCODE
    $RowPassed = $RowExitCode -eq 0
    if (-not $RowPassed) {
        Write-Warning "Row $alias exited with code $RowExitCode (0=pass, 3=aborted, 4=gates red). Continuing with remaining rows."
    }

    $ExtractionYield = $null
    $ExtractionPassed = $null
    if ($RowPassed -and -not $SkipExtraction) {
        Write-Step "Row $alias — extract scorer samples without executing generated Python."
        & $PythonPath $ExtractorPath `
            --generations (Join-Path $ResultsDirectory "generations.jsonl") `
            --prompt-manifest $PromptManifestPath `
            --samples-output (Join-Path $RowDirectory "samples.jsonl") `
            --report-output (Join-Path $RowDirectory "extraction-report.json")
        $ExtractionExitCode = $LASTEXITCODE
        $ExtractionReportPath = Join-Path $RowDirectory "extraction-report.json"
        if (Test-Path -LiteralPath $ExtractionReportPath) {
            $ExtractionReport = Get-Content -LiteralPath $ExtractionReportPath -Raw | ConvertFrom-Json
            $ExtractionYield = $ExtractionReport.extraction_yield
            $ExtractionPassed = [bool]$ExtractionReport.passed
        }
        if ($ExtractionExitCode -notin @(0, 4)) {
            throw "Extractor v2 exited with code $ExtractionExitCode for row $alias."
        }
    }

    $RowManifestPath = Join-Path $ResultsDirectory "run-manifest.json"
    $RowSummaryValue = $null
    if (Test-Path -LiteralPath $RowManifestPath) {
        $RowManifest = Get-Content -LiteralPath $RowManifestPath -Raw | ConvertFrom-Json
        $RowSummaryValue = $RowManifest.summary
        $RowPassed = $RowPassed -and [bool]$RowManifest.passed
    }
    $RowResults += [ordered]@{
        row_alias = $alias
        model_sha256 = $Row.sha256
        exit_code = $RowExitCode
        passed = $RowPassed
        extraction_passed = $ExtractionPassed
        extraction_yield = $ExtractionYield
        summary = $RowSummaryValue
        evidence_directory = $RowDirectory
    }
}

$OverallPassed = @($RowResults | Where-Object { -not $_.passed }).Count -eq 0
$SessionSummary = [ordered]@{
    schema_version = 1
    run_id = $SessionId
    mode = "pilot-generation"
    suite_id = $Suite.suite_id
    generated_at_utc = (Get-Date).ToUniversalTime().ToString("o")
    skipped_extraction = [bool]$SkipExtraction
    overall_passed = $OverallPassed
    rows = $RowResults
    generated_programs_executed = 0
}
[System.IO.File]::WriteAllText(
    (Join-Path $SessionDirectory "pilot-generation-summary.json"),
    (($SessionSummary | ConvertTo-Json -Depth 10) + "`n"),
    [System.Text.UTF8Encoding]::new($false)
)

Write-Step "Pilot generation finished."
Write-Host "Session summary: $(Join-Path $SessionDirectory 'pilot-generation-summary.json')"
if (-not $OverallPassed) {
    throw "One or more pilot rows failed. Review $SessionDirectory"
}
Write-Host "Overall E3 pilot generation gate: PASS" -ForegroundColor Green
