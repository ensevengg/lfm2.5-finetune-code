[CmdletBinding()]
param()

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$workspace = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$llamaSource = Join-Path $workspace "vendor\llama.cpp"
$dockerfile = Join-Path $workspace "docker\phase1-llama-cuda.Dockerfile"
$reportDir = Join-Path $workspace "reports\phase1"

$commit = "3018a11e79e489b657dbb77c95694889ccff92df"
$imageTag = "lfm25/llama-cpp:3018a11e-cuda128-sm120"
$develImage = "nvidia/cuda:12.8.1-devel-ubuntu24.04@sha256:520292dbb4f755fd360766059e62956e9379485d9e073bbd2f6e3c20c270ed66"
$runtimeImage = "nvidia/cuda:12.8.1-runtime-ubuntu24.04@sha256:ebef3c171eeef0298e4eb2e4be843105edf3b8b0ac45e0b43acee358e8046867"

function Invoke-DockerCaptured {
    param([string[]]$Arguments)

    $previousPreference = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        $output = & docker @Arguments 2>&1
        $exitCode = $LASTEXITCODE
    }
    finally {
        $ErrorActionPreference = $previousPreference
    }
    [pscustomobject]@{
        ExitCode = $exitCode
        Output = @($output | ForEach-Object { $_.ToString() })
    }
}

if (-not (Test-Path -LiteralPath $llamaSource)) {
    throw "Pinned llama.cpp checkout is missing: $llamaSource"
}

$actualCommit = (& git -C $llamaSource rev-parse HEAD).Trim()
if ($LASTEXITCODE -ne 0 -or $actualCommit -ne $commit) {
    throw "llama.cpp checkout is $actualCommit; expected $commit"
}

$dirty = & git -C $llamaSource status --short
if ($LASTEXITCODE -ne 0 -or $dirty) {
    throw "Pinned llama.cpp checkout has local changes; refusing an unrepeatable build."
}

New-Item -ItemType Directory -Force -Path $reportDir | Out-Null

$buildArgs = @(
    "build"
    "--progress", "plain"
    "--file", $dockerfile
    "--target", "toolchain"
    "--build-arg", "CUDA_DEVEL_IMAGE=$develImage"
    "--build-arg", "CUDA_RUNTIME_IMAGE=$runtimeImage"
    "--build-arg", "LLAMA_CPP_COMMIT=$commit"
    "--build-arg", "CUDA_ARCHITECTURE=120"
    "--label", "ai.liquid.workspace=lfm2.5-finetune"
    "--tag", $imageTag
    $llamaSource
)

& docker @buildArgs
if ($LASTEXITCODE -ne 0) {
    throw "Phase 1 toolchain image build failed with exit code $LASTEXITCODE"
}

$inspect = (& docker image inspect $imageTag --format "{{json .}}") | ConvertFrom-Json
if ($LASTEXITCODE -ne 0) {
    throw "Unable to inspect Phase 1 toolchain image."
}

$versionResult = Invoke-DockerCaptured @(
    "run", "--rm", "--pull", "never", "--gpus", "device=0"
    "--network", "none", "--read-only"
    "--cap-drop", "ALL", "--security-opt", "no-new-privileges=true"
    "--pids-limit", "64", "--memory", "1g", "--memory-swap", "1g"
    "--cpus", "2", "--user", "65534:65534"
    "--tmpfs", "/tmp:rw,noexec,nosuid,size=64m,mode=1777"
    "--env", "NVIDIA_DRIVER_CAPABILITIES=compute,utility"
    $imageTag, "llama-cli", "--version"
)
if ($versionResult.ExitCode -ne 0) {
    throw "Built llama.cpp CLI failed its restricted version smoke test."
}

$deviceResult = Invoke-DockerCaptured @(
    "run", "--rm", "--pull", "never", "--gpus", "device=0"
    "--network", "none", "--read-only"
    "--cap-drop", "ALL", "--security-opt", "no-new-privileges=true"
    "--pids-limit", "64", "--memory", "1g", "--memory-swap", "1g"
    "--cpus", "2", "--user", "65534:65534"
    "--tmpfs", "/tmp:rw,noexec,nosuid,size=64m,mode=1777"
    "--env", "NVIDIA_DRIVER_CAPABILITIES=compute,utility"
    $imageTag, "llama-cli", "--list-devices"
)
if ($deviceResult.ExitCode -ne 0) {
    throw "Built llama.cpp image did not enumerate the CUDA GPU."
}

$packageResult = Invoke-DockerCaptured @(
    "run", "--rm", "--pull", "never", "--network", "none", "--read-only"
    "--cap-drop", "ALL", "--security-opt", "no-new-privileges=true"
    "--pids-limit", "64", "--memory", "1g", "--memory-swap", "1g"
    "--cpus", "2", "--user", "65534:65534"
    "--tmpfs", "/tmp:rw,noexec,nosuid,size=64m,mode=1777"
    "--entrypoint", "/bin/cat", $imageTag
    "/opt/llama-convert/python-packages.lock.txt"
)
if ($packageResult.ExitCode -ne 0) {
    throw "Unable to read converter dependency lock from the built image."
}

$report = [ordered]@{
    schema_version = 1
    generated_at_utc = [DateTime]::UtcNow.ToString("o")
    passed = $true
    llama_cpp_commit = $commit
    source_checkout = $llamaSource
    cuda_architecture = "120"
    image = [ordered]@{
        tag = $imageTag
        id = $inspect.Id
        repo_digests = @($inspect.RepoDigests)
        created = $inspect.Created
        size_bytes = $inspect.Size
        labels = $inspect.Config.Labels
    }
    base_images = [ordered]@{
        devel = $develImage
        runtime = $runtimeImage
    }
    build_options = @(
        "GGML_CUDA=ON"
        "CMAKE_CUDA_ARCHITECTURES=120"
        "CMAKE_EXE_LINKER_FLAGS=-Wl,--allow-shlib-undefined"
        "GGML_CUDA_FA_ALL_QUANTS=ON"
        "GGML_NATIVE=OFF"
        "GGML_RPC=OFF"
        "LLAMA_CURL=OFF"
        "LLAMA_BUILD_UI=OFF"
        "LLAMA_USE_PREBUILT_UI=OFF"
    )
    version_output = ($versionResult.Output -join [Environment]::NewLine).Trim()
    device_output = ($deviceResult.Output -join [Environment]::NewLine).Trim()
    converter_packages = @($packageResult.Output)
}

$reportPath = Join-Path $reportDir "toolchain-build.json"
$json = $report | ConvertTo-Json -Depth 10
[IO.File]::WriteAllText(
    $reportPath,
    $json + [Environment]::NewLine,
    [Text.UTF8Encoding]::new($false)
)

Write-Output "Phase 1 toolchain build passed."
Write-Output "Image: $imageTag"
Write-Output "Report: $reportPath"
