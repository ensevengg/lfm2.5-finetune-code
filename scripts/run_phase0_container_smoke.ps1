[CmdletBinding()]
param()

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$workspace = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$modelDir = (Resolve-Path (Join-Path $workspace "lfm2.5-model-merged")).Path
$resultDir = (Resolve-Path (Join-Path $workspace "reports\phase0")).Path
$smokeScript = (Resolve-Path (Join-Path $PSScriptRoot "phase0_container_smoke.sh")).Path

$baseImage = "nvidia/cuda:12.8.1-base-ubuntu24.04@sha256:133c78a0575303be34164d0b90137a042172bdf60696af01a3c424ab402d86e2"
$runtimeImage = "nvidia/cuda:12.8.1-runtime-ubuntu24.04@sha256:ebef3c171eeef0298e4eb2e4be843105edf3b8b0ac45e0b43acee358e8046867"
$develImage = "nvidia/cuda:12.8.1-devel-ubuntu24.04@sha256:520292dbb4f755fd360766059e62956e9379485d9e073bbd2f6e3c20c270ed66"

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
    "--pids-limit", "64"
    "--memory", "1g"
    "--memory-swap", "1g"
    "--cpus", "2"
    "--user", "65534:65534"
    "--tmpfs", "/tmp:rw,noexec,nosuid,size=64m,mode=1777"
    "--env", "CUDA_CACHE_PATH=/tmp/cuda-cache"
    "--env", "HOME=/tmp/home"
    "--env", "NVIDIA_DRIVER_CAPABILITIES=compute,utility"
    "--mount", "type=bind,source=$modelDir,target=/models,readonly"
    "--mount", "type=bind,source=$resultDir,target=/results"
    "--mount", "type=bind,source=$smokeScript,target=/phase0-smoke.sh,readonly"
    "--entrypoint", "/bin/bash"
    $baseImage
    "/phase0-smoke.sh"
)

& docker @dockerArgs
if ($LASTEXITCODE -ne 0) {
    throw "Restricted CUDA container smoke test failed with exit code $LASTEXITCODE"
}

$dockerVersion = (& docker version --format "{{json .}}") | ConvertFrom-Json
if ($LASTEXITCODE -ne 0) {
    throw "Unable to read Docker version"
}
$dockerInfo = (& docker info --format "{{json .}}") | ConvertFrom-Json
if ($LASTEXITCODE -ne 0) {
    throw "Unable to read Docker info"
}
$imageInspect = (& docker image inspect $baseImage --format "{{json .}}") | ConvertFrom-Json
if ($LASTEXITCODE -ne 0) {
    throw "Unable to inspect pinned CUDA image"
}

$assertions = [ordered]@{}
foreach ($line in Get-Content -LiteralPath (Join-Path $resultDir "container-isolation-smoke.txt")) {
    $parts = $line -split "=", 2
    if ($parts.Count -eq 2) {
        $assertions[$parts[0]] = $parts[1]
    }
}

$gpuLine = (
    Get-Content -Raw -LiteralPath (Join-Path $resultDir "container-gpu-smoke.txt")
).Trim()
$expectedAssertions = [ordered]@{
    uid = "65534"
    cap_eff = "0"
    no_new_privs = "1"
    network_interfaces = "lo"
    root_read_only = "1"
    model_mount_read_only = "1"
    results_mount_writable = "1"
    tmpfs_writable = "1"
}
$failedAssertions = @(
    $expectedAssertions.GetEnumerator() |
        Where-Object {
            -not $assertions.Contains($_.Key) -or
            $assertions[$_.Key] -ne $_.Value
        }
)
$allAssertionsPassed = $failedAssertions.Count -eq 0

$report = [ordered]@{
    schema_version = 1
    generated_at_utc = [DateTime]::UtcNow.ToString("o")
    passed = (
        $allAssertionsPassed -and
        $dockerInfo.OSType -eq "linux" -and
        $dockerInfo.Runtimes.PSObject.Properties.Name -contains "nvidia" -and
        -not [string]::IsNullOrWhiteSpace($gpuLine)
    )
    docker = [ordered]@{
        desktop_version = $dockerVersion.Server.Platform.Name
        client_version = $dockerVersion.Client.Version
        server_version = $dockerVersion.Server.Version
        server_os = $dockerVersion.Server.Os
        server_architecture = $dockerVersion.Server.Arch
        kernel_version = $dockerInfo.KernelVersion
        cgroup_version = $dockerInfo.CgroupVersion
        default_runtime = $dockerInfo.DefaultRuntime
        runtimes = @($dockerInfo.Runtimes.PSObject.Properties.Name)
        security_options = @($dockerInfo.SecurityOptions)
        memory_limit_supported = $dockerInfo.MemoryLimit
        swap_limit_supported = $dockerInfo.SwapLimit
        pids_limit_supported = $dockerInfo.PidsLimit
        cpu_quota_supported = $dockerInfo.CpuCfsQuota
    }
    pinned_images = [ordered]@{
        smoke = [ordered]@{
            reference = $baseImage
            manifest_list_digest = "sha256:133c78a0575303be34164d0b90137a042172bdf60696af01a3c424ab402d86e2"
            linux_amd64_digest = "sha256:e711c99333fdfe8ae1e677b4972be6c5021f0128a1d31f775c7e58d88921b6a9"
            local_image_id = $imageInspect.Id
            local_repo_digests = @($imageInspect.RepoDigests)
        }
        runtime = [ordered]@{
            reference = $runtimeImage
            manifest_list_digest = "sha256:ebef3c171eeef0298e4eb2e4be843105edf3b8b0ac45e0b43acee358e8046867"
            linux_amd64_digest = "sha256:828c4d878adcaa4265d80c95d8ec877149b49bb2419a4cf3bb6aa889bbb7ca2e"
        }
        build = [ordered]@{
            reference = $develImage
            manifest_list_digest = "sha256:520292dbb4f755fd360766059e62956e9379485d9e073bbd2f6e3c20c270ed66"
            linux_amd64_digest = "sha256:4b9ed5fa8361736996499f64ecebf25d4ec37ff56e4d11323ccde10aa36e0c43"
        }
    }
    gpu_observed_in_container = $gpuLine
    assertions = $assertions
    run_policy = @(
        "--pull never"
        "--gpus device=0"
        "--network none"
        "--read-only"
        "--cap-drop ALL"
        "--security-opt no-new-privileges=true"
        "--pids-limit 64"
        "--memory 1g --memory-swap 1g"
        "--cpus 2"
        "--user 65534:65534"
        "size-limited noexec/nosuid tmpfs"
        "read-only model and smoke-script mounts"
        "one writable results mount"
        "NVIDIA_DRIVER_CAPABILITIES=compute,utility"
    )
}

$reportPath = Join-Path $resultDir "container-readiness.json"
$json = $report | ConvertTo-Json -Depth 10
[IO.File]::WriteAllText(
    $reportPath,
    $json + [Environment]::NewLine,
    [Text.UTF8Encoding]::new($false)
)

if (-not $report.passed) {
    throw "Container readiness assertions did not all pass; inspect $reportPath"
}

Write-Output "Restricted CUDA container smoke test passed."
Write-Output "Report: $reportPath"
