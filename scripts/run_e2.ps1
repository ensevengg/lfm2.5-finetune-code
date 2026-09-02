[CmdletBinding()]
param(
    [switch]$ValidateOnly,
    [string]$TaskId = "",
    [switch]$AllDevelopmentTasks,
    [switch]$ConfirmExecution
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

function Write-Step([string]$Message) {
    Write-Host "`n[E2] $Message" -ForegroundColor Cyan
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

if ($AllDevelopmentTasks -and $TaskId) {
    throw "Use either -TaskId or -AllDevelopmentTasks, not both."
}
if (-not $ValidateOnly -and -not $ConfirmExecution) {
    throw "Scoring executes generated Python. Re-run with -ConfirmExecution only after -ValidateOnly passes."
}

$Workspace = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$SuitePath = Resolve-RequiredFile (Join-Path $Workspace "benchmarks\suites\e2-humanevalplus-dev-v1.json") "E2 suite"
$Suite = Get-Content -LiteralPath $SuitePath -Raw | ConvertFrom-Json
$PythonPath = Resolve-RequiredFile (Join-Path $Workspace ".venv\Scripts\python.exe") "Workspace Python"
$PreparerPath = Resolve-RequiredFile (Join-Path $Workspace $Suite.preparation.implementation_path) "E2 input preparer"
$DockerfilePath = Resolve-RequiredFile (Join-Path $Workspace $Suite.scorer.dockerfile.path) "E2 Dockerfile"
$ProbePath = Resolve-RequiredFile (Join-Path $Workspace "scripts\validate_e2_container.py") "E2 container probe"

Assert-HashAndSize $PreparerPath $Suite.preparation.implementation_sha256 $Suite.preparation.implementation_size_bytes "input preparer"
Assert-HashAndSize $DockerfilePath $Suite.scorer.dockerfile.sha256 $Suite.scorer.dockerfile.size_bytes "scorer Dockerfile"

if ($AllDevelopmentTasks) {
    $SelectedTaskIds = @($Suite.development_task_ids)
} elseif ($TaskId) {
    $SelectedTaskIds = @($TaskId)
} else {
    $SelectedTaskIds = @($Suite.default_manual_task_id)
}
$UnknownTaskIds = @($SelectedTaskIds | Where-Object { $_ -notin $Suite.development_task_ids })
if ($UnknownTaskIds.Count -gt 0) {
    throw "Task IDs are outside the frozen development set: $($UnknownTaskIds -join ', ')"
}

Write-Step "Preflight: verify Docker and the exact scorer image."
$DockerServer = (docker info --format "Docker server {{.ServerVersion}}").Trim()
if ($LASTEXITCODE -ne 0) {
    throw "Docker Engine is not responding."
}
Write-Host $DockerServer
$ImageId = (docker image inspect $Suite.scorer.image_tag --format "{{.Id}}").Trim()
if ($LASTEXITCODE -ne 0 -or $ImageId -ne $Suite.scorer.image_id) {
    throw "Pinned scorer image is missing or mismatched. Expected $($Suite.scorer.image_id); got $ImageId"
}
$ImageRevision = (docker image inspect $Suite.scorer.image_tag --format '{{index .Config.Labels "org.opencontainers.image.revision"}}').Trim()
if ($ImageRevision -ne $Suite.scorer.revision) {
    throw "Scorer revision label mismatch."
}
Write-Host "  PASS image ID $ImageId"

$ModeName = if ($ValidateOnly) { "validate" } else { "score" }
$SessionId = "e2-$ModeName-" + (Get-Date).ToUniversalTime().ToString("yyyyMMddTHHmmssfffZ")
$SessionDirectory = Join-Path $Workspace "reports\e2\$SessionId"
if (Test-Path -LiteralPath $SessionDirectory) {
    throw "Refusing to overwrite E2 evidence: $SessionDirectory"
}
New-Item -ItemType Directory -Path $SessionDirectory | Out-Null
$InputsDirectory = Join-Path $SessionDirectory "inputs"
$ResultsDirectory = Join-Path $SessionDirectory "results"
New-Item -ItemType Directory -Path $ResultsDirectory | Out-Null

Write-Step "Prepare a matching sample/dataset subset; execute nothing."
$PrepareArguments = @(
    $PreparerPath,
    "--suite", $SuitePath,
    "--workspace", $Workspace,
    "--output-dir", $InputsDirectory
)
foreach ($SelectedTaskId in $SelectedTaskIds) {
    $PrepareArguments += @("--task-id", $SelectedTaskId)
}
& $PythonPath @PrepareArguments
if ($LASTEXITCODE -ne 0) {
    throw "E2 input preparation failed."
}

$PreparedSamples = Resolve-RequiredFile (Join-Path $InputsDirectory "samples.jsonl") "Prepared samples"
$PreparedDataset = Resolve-RequiredFile (Join-Path $InputsDirectory "HumanEvalPlus-subset.jsonl") "Prepared dataset"
$ContainerArguments = @(
    "run", "--rm",
    "--pull", "never",
    "--platform", "linux/amd64",
    "--network", $Suite.container.network,
    "--read-only",
    "--cap-drop", "ALL",
    "--security-opt", "no-new-privileges=true",
    "--pids-limit", [string]$Suite.container.pids_limit,
    "--memory", $Suite.container.memory,
    "--memory-swap", $Suite.container.memory_swap,
    "--cpus", [string]$Suite.container.cpus,
    "--shm-size", $Suite.container.shm_size,
    "--ulimit", "nofile=256:256",
    "--user", $Suite.container.user,
    "--tmpfs", $Suite.container.tmpfs,
    "--env", "XDG_CACHE_HOME=/tmp/cache",
    "--env", "HUMANEVAL_OVERRIDE_PATH=/inputs/HumanEvalPlus-subset.jsonl",
    "--env", "EVALPLUS_MAX_MEMORY_BYTES=$($Suite.container.evalplus_max_memory_bytes_per_process)",
    "--mount", (New-BindMount $PreparedSamples "/inputs/samples.jsonl" -ReadOnly),
    "--mount", (New-BindMount $PreparedDataset "/inputs/HumanEvalPlus-subset.jsonl" -ReadOnly),
    "--mount", (New-BindMount $ResultsDirectory "/results")
)

if ($ValidateOnly) {
    Write-Step "Validate the restricted container. Generated Python will not run."
    $ContainerArguments += @(
        "--mount", (New-BindMount $ProbePath "/probe/validate_e2_container.py" -ReadOnly),
        $Suite.scorer.image_tag,
        "python", "/probe/validate_e2_container.py"
    )
} else {
    Write-Step "Score $($SelectedTaskIds.Count) development task(s) inside the restricted container."
    Write-Warning "This is the first stage that executes generated Python. It does not execute on Windows."
    $ContainerArguments += @(
        $Suite.scorer.image_tag,
        "python", "-m", "evalplus.evaluate",
        "--dataset", $Suite.evaluation.dataset_argument,
        "--samples", "/inputs/samples.jsonl",
        "--parallel", [string]$Suite.evaluation.parallel,
        "--min-time-limit", [string]$Suite.evaluation.min_time_limit_seconds,
        "--gt-time-limit-factor", [string]$Suite.evaluation.ground_truth_time_limit_factor,
        "--output-file", "/results/eval-results.json"
    )
}

Write-Json (Join-Path $SessionDirectory "docker-command.json") ([ordered]@{
    schema_version = 1
    mode = $ModeName
    image_tag = $Suite.scorer.image_tag
    image_id = $ImageId
    selected_task_ids = $SelectedTaskIds
    docker_arguments = $ContainerArguments
    generated_program_execution_expected = (-not $ValidateOnly)
})

$LogPath = Join-Path $SessionDirectory "container.log"
& docker @ContainerArguments 2>&1 | Tee-Object -FilePath $LogPath
$ContainerExitCode = $LASTEXITCODE
if ($ContainerExitCode -ne 0) {
    throw "Restricted E2 container exited with code $ContainerExitCode. Evidence: $SessionDirectory"
}

if ($ValidateOnly) {
    $ContainerValidationPath = Resolve-RequiredFile (Join-Path $ResultsDirectory "container-validation.json") "Container validation report"
    $ContainerValidation = Get-Content -LiteralPath $ContainerValidationPath -Raw | ConvertFrom-Json
    if (-not $ContainerValidation.passed -or $ContainerValidation.generated_programs_executed) {
        throw "Restricted-container validation failed. Review $ContainerValidationPath"
    }
    Write-Json (Join-Path $SessionDirectory "run-manifest.json") ([ordered]@{
        schema_version = 1
        run_id = $SessionId
        mode = "validate"
        passed = $true
        selected_task_ids = $SelectedTaskIds
        scorer_image_id = $ImageId
        generated_programs_executed = 0
        scores_computed = 0
        evidence_directory = $SessionDirectory
    })
    Write-Step "Validation passed. Stop here before your first manual score."
    Write-Host "Evidence: $SessionDirectory" -ForegroundColor Green
    exit 0
}

$EvalResultPath = Resolve-RequiredFile (Join-Path $ResultsDirectory "eval-results.json") "EvalPlus result"
$EvalResult = Get-Content -LiteralPath $EvalResultPath -Raw | ConvertFrom-Json
$ResultTaskIds = @($EvalResult.eval.PSObject.Properties.Name)
$Passed = $ResultTaskIds.Count -eq $SelectedTaskIds.Count -and @($SelectedTaskIds | Where-Object { $_ -notin $ResultTaskIds }).Count -eq 0
Write-Json (Join-Path $SessionDirectory "run-manifest.json") ([ordered]@{
    schema_version = 1
    run_id = $SessionId
    mode = "score"
    passed = $Passed
    selected_task_ids = $SelectedTaskIds
    result_task_ids = $ResultTaskIds
    scorer_image_id = $ImageId
    generated_programs_executed = $SelectedTaskIds.Count
    scores_computed = $ResultTaskIds.Count
    evalplus_pass_at_k = $EvalResult.pass_at_k
    evidence_directory = $SessionDirectory
})
if (-not $Passed) {
    throw "E2 result completeness gate failed. Review $SessionDirectory"
}

Write-Step "Scoring finished."
Write-Host "Results: $EvalResultPath" -ForegroundColor Green
Write-Host "Evidence: $SessionDirectory"
