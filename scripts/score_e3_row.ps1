[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$RowDirectory,
    [string]$SuiteRelativePath = "benchmarks\suites\e3-humanevalplus-pilot-v1.json"
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

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
    $Item = Get-Item -LiteralPath $Path
    $ActualHash = (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($Item.Length -ne $ExpectedSize -or $ActualHash -ne $ExpectedHash.ToLowerInvariant()) {
        throw "$Label identity mismatch. Got $($Item.Length) bytes, SHA-256 $ActualHash"
    }
    Write-Host "  PASS $Label"
}

function New-BindMount([string]$Source, [string]$Target, [switch]$ReadOnly) {
    $Mount = "type=bind,source=$Source,target=$Target"
    if ($ReadOnly) {
        $Mount += ",readonly"
    }
    return $Mount
}

function Write-Json([string]$Path, [object]$Value) {
    $Json = ($Value | ConvertTo-Json -Depth 20) + "`n"
    [System.IO.File]::WriteAllText($Path, $Json, [System.Text.UTF8Encoding]::new($false))
}

$Workspace = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$RowDirectory = (Resolve-Path -LiteralPath $RowDirectory).Path
$E3SuitePath = Resolve-RequiredFile (Join-Path $Workspace $SuiteRelativePath) "Generation suite"
$E2SuitePath = Resolve-RequiredFile (Join-Path $Workspace "benchmarks\suites\e2-humanevalplus-dev-v1.json") "E2 suite (frozen scorer identity)"
$E3Suite = Get-Content -LiteralPath $E3SuitePath -Raw | ConvertFrom-Json
$E2Suite = Get-Content -LiteralPath $E2SuitePath -Raw | ConvertFrom-Json
$PythonPath = Resolve-RequiredFile (Join-Path $Workspace ".venv\Scripts\python.exe") "Workspace Python"
$PreparerPath = Resolve-RequiredFile (Join-Path $Workspace "scripts\prepare_e3_inputs.py") "E3 input preparer"

function Write-Step([string]$Message) {
    Write-Host "`n[E3-score] $Message" -ForegroundColor Cyan
}

Write-Step "Preflight: verify Docker and the frozen scorer image identity."
$DockerServer = (docker info --format "Docker server {{.ServerVersion}}").Trim()
if ($LASTEXITCODE -ne 0) {
    throw "Docker Engine is not responding."
}
Write-Host $DockerServer
$ImageId = (docker image inspect $E2Suite.scorer.image_tag --format "{{.Id}}").Trim()
if ($LASTEXITCODE -ne 0 -or $ImageId -ne $E2Suite.scorer.image_id) {
    throw "Pinned scorer image is missing or mismatched. Expected $($E2Suite.scorer.image_id); got $ImageId"
}
Write-Host "  PASS scorer image $ImageId"

$SamplesPath = Resolve-RequiredFile (Join-Path $RowDirectory "samples.jsonl") "Row samples"
$ExtractionReportPath = Resolve-RequiredFile (Join-Path $RowDirectory "extraction-report.json") "Row extraction report"

$Mode = "score"
$SessionId = "e3-score-" + (Get-Date).ToUniversalTime().ToString("yyyyMMddTHHmmssfffZ")
$SessionDirectory = Join-Path $RowDirectory "scoring\$SessionId"
if (Test-Path -LiteralPath $SessionDirectory) {
    throw "Refusing to overwrite E3 scoring evidence: $SessionDirectory"
}
New-Item -ItemType Directory -Path $SessionDirectory | Out-Null
$InputsDirectory = Join-Path $SessionDirectory "inputs"
$ResultsDirectory = Join-Path $SessionDirectory "results"
New-Item -ItemType Directory -Path $ResultsDirectory | Out-Null

Write-Step "Prepare a matched pilot sample/dataset subset; execute nothing."
$PrepareArguments = @(
    $PreparerPath,
    "--suite", $E3SuitePath,
    "--workspace", $Workspace,
    "--samples", $SamplesPath,
    "--extraction-report", $ExtractionReportPath,
    "--output-dir", $InputsDirectory
)
& $PythonPath @PrepareArguments
if ($LASTEXITCODE -ne 0) {
    throw "E3 input preparation failed."
}

$PreparedSamples = Resolve-RequiredFile (Join-Path $InputsDirectory "samples.jsonl") "Prepared samples"
$PreparedDataset = Resolve-RequiredFile (Join-Path $InputsDirectory "HumanEvalPlus-subset.jsonl") "Prepared dataset"
$ContainerArguments = @(
    "run", "--rm",
    "--pull", "never",
    "--platform", "linux/amd64",
    "--network", $E2Suite.container.network,
    "--read-only",
    "--cap-drop", "ALL",
    "--security-opt", "no-new-privileges=true",
    "--pids-limit", [string]$E2Suite.container.pids_limit,
    "--memory", $E2Suite.container.memory,
    "--memory-swap", $E2Suite.container.memory_swap,
    "--cpus", [string]$E2Suite.container.cpus,
    "--shm-size", $E2Suite.container.shm_size,
    "--ulimit", "nofile=256:256",
    "--user", $E2Suite.container.user,
    "--tmpfs", $E2Suite.container.tmpfs,
    "--env", "XDG_CACHE_HOME=/tmp/cache",
    "--env", "HUMANEVAL_OVERRIDE_PATH=/inputs/HumanEvalPlus-subset.jsonl",
    "--env", "EVALPLUS_MAX_MEMORY_BYTES=$($E2Suite.container.evalplus_max_memory_bytes_per_process)",
    "--mount", (New-BindMount $PreparedSamples "/inputs/samples.jsonl" -ReadOnly),
    "--mount", (New-BindMount $PreparedDataset "/inputs/HumanEvalPlus-subset.jsonl" -ReadOnly),
    "--mount", (New-BindMount $ResultsDirectory "/results"),
    $E2Suite.scorer.image_tag,
    "python", "-m", "evalplus.evaluate",
    "--dataset", $E2Suite.evaluation.dataset_argument,
    "--samples", "/inputs/samples.jsonl",
    "--parallel", [string]$E2Suite.evaluation.parallel,
    "--min-time-limit", [string]$E2Suite.evaluation.min_time_limit_seconds,
    "--gt-time-limit-factor", [string]$E2Suite.evaluation.ground_truth_time_limit_factor,
    "--output-file", "/results/eval-results.json"
)

Write-Json (Join-Path $SessionDirectory "docker-command.json") ([ordered]@{
    schema_version = 1
    mode = $Mode
    image_tag = $E2Suite.scorer.image_tag
    image_id = $ImageId
    row_directory = $RowDirectory
    docker_arguments = $ContainerArguments
    generated_program_execution_expected = $true
})

Write-Step "Score the pilot samples inside the restricted container."
$LogPath = Join-Path $SessionDirectory "container.log"
& docker @ContainerArguments 2>&1 | Tee-Object -FilePath $LogPath
$ContainerExitCode = $LASTEXITCODE
if ($ContainerExitCode -ne 0) {
    throw "Restricted E3 scoring container exited with code $ContainerExitCode. Evidence: $SessionDirectory"
}

$EvalResultPath = Resolve-RequiredFile (Join-Path $ResultsDirectory "eval-results.json") "EvalPlus result"
$EvalResult = Get-Content -LiteralPath $EvalResultPath -Raw | ConvertFrom-Json
$ResultTaskIds = @($EvalResult.eval.PSObject.Properties.Name)
$PreparationReport = Get-Content (
    Join-Path $InputsDirectory "preparation-report.json"
) -Raw | ConvertFrom-Json
$ExpectedTaskIds = @($PreparationReport.sampled_task_ids)
$MissingResults = @($ExpectedTaskIds | Where-Object { $_ -notin $ResultTaskIds })
$Passed = $MissingResults.Count -eq 0
Write-Json (Join-Path $SessionDirectory "run-manifest.json") ([ordered]@{
    schema_version = 1
    run_id = $SessionId
    mode = $Mode
    passed = $Passed
    expected_task_ids = $ExpectedTaskIds
    result_task_ids = $ResultTaskIds
    missing_result_task_ids = $MissingResults
    extraction_failures_scored_zero = @($PreparationReport.extraction_failures_scored_zero)
    scorer_image_id = $ImageId
    generated_programs_executed = $ExpectedTaskIds.Count
    scores_computed = $ResultTaskIds.Count
    evalplus_pass_at_k = $EvalResult.pass_at_k
    evidence_directory = $SessionDirectory
})
if (-not $Passed) {
    throw "E3 scoring completeness gate failed. Missing: $($MissingResults -join ', ')"
}

Write-Step "Scoring finished."
Write-Host "Results: $EvalResultPath" -ForegroundColor Green
Write-Host "Evidence: $SessionDirectory"
