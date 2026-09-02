[CmdletBinding()]
param()

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$Workspace = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$SourceDirectory = Join-Path $Workspace "vendor\evalplus"
$Dockerfile = Join-Path $Workspace "docker\e2-evalplus.Dockerfile"
$ExpectedRevision = "26d6d00bb1fd0fa37f39c99d5290da67891d1c5e"
$ExpectedPackageVersion = "0.3.1.post44"
$ExpectedBaseImageId = "sha256:9534e5a8e315485d4061ed659af0fd78a284c015f9b73661b41d6bab25604534"
$BaseImage = "python@sha256:9534e5a8e315485d4061ed659af0fd78a284c015f9b73661b41d6bab25604534"
$ImageTag = "lfm25/evalplus:26d6d00-e2"

if (-not (Test-Path -LiteralPath $Dockerfile -PathType Leaf)) {
    throw "Scorer Dockerfile not found: $Dockerfile"
}
if (-not (Test-Path -LiteralPath $SourceDirectory -PathType Container)) {
    throw "Pinned EvalPlus source not found: $SourceDirectory"
}

$ActualRevision = (git -C $SourceDirectory rev-parse HEAD).Trim()
if ($LASTEXITCODE -ne 0 -or $ActualRevision -ne $ExpectedRevision) {
    throw "EvalPlus revision mismatch. Expected $ExpectedRevision; got $ActualRevision"
}
$DirtySource = @(git -C $SourceDirectory status --porcelain)
if ($LASTEXITCODE -ne 0 -or $DirtySource.Count -ne 0) {
    throw "EvalPlus source has local changes. Refusing to build a mislabeled scorer."
}

$DockerServer = (docker info --format "Docker server {{.ServerVersion}}").Trim()
if ($LASTEXITCODE -ne 0) {
    throw "Docker Engine is not responding."
}
Write-Host $DockerServer

$BaseImageId = (docker image inspect $BaseImage --format "{{.Id}}").Trim()
if ($LASTEXITCODE -ne 0 -or $BaseImageId -ne $ExpectedBaseImageId) {
    throw "Pinned Python base image is missing or mismatched. Run: docker pull python:3.11-slim"
}
Write-Host "PASS pinned Python base: $BaseImageId"
Write-Host "Building pinned EvalPlus scorer. This downloads Python packages but executes no model output."

docker build `
    --pull=false `
    --build-arg "EVALPLUS_REVISION=$ExpectedRevision" `
    --build-arg "EVALPLUS_PACKAGE_VERSION=$ExpectedPackageVersion" `
    --tag $ImageTag `
    --file $Dockerfile `
    $SourceDirectory
if ($LASTEXITCODE -ne 0) {
    throw "E2 scorer image build failed."
}

$ImageId = (docker image inspect $ImageTag --format "{{.Id}}").Trim()
$ImageRevision = (docker image inspect $ImageTag --format '{{index .Config.Labels "org.opencontainers.image.revision"}}').Trim()
if ($ImageRevision -ne $ExpectedRevision) {
    throw "Built image revision label mismatch. Expected $ExpectedRevision; got $ImageRevision"
}

Write-Host "PASS scorer image tag: $ImageTag" -ForegroundColor Green
Write-Host "PASS scorer image ID:  $ImageId" -ForegroundColor Green
Write-Host "No generated program was executed."
