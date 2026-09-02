# Evaluation host: CachyOS or Windows with WSL2?

Status: recommendation for the E1+ evaluation work  
Date: 2026-08-12

## Recommendation

Use **Windows 11 with Docker Desktop's WSL2 backend** for E1, E2, the first
pilot, and the first sealed comparison. It is the lower-headache choice for
this repository.

This recommendation is specific to the current project, not a claim that WSL2
is universally faster than native Linux. Phases 0-2 have already passed on the
Windows/WSL2/Docker Desktop stack, all implemented host wrappers are
PowerShell, the model and Hugging Face caches live on Windows drives, and the
recorded evidence contains Windows paths and a WSL2 kernel. Changing to
CachyOS now would introduce an OS migration, a wrapper portability pass, a new
NVIDIA driver and container-runtime setup, artifact transfer, and a complete
new host qualification before it produced a single new model-quality result.
See the existing [Phase 0 report](../../reports/phase0/README.md),
[Phase 1 report](../../reports/phase1/README.md),
[Phase 2 report](../../reports/phase2/README.md), and
[PowerShell wrappers](../../scripts).

If the project later needs a dedicated native-Linux benchmark host, native
Linux is attractive, but I would choose **Ubuntu 24.04 LTS rather than
CachyOS** for that machine. Ubuntu 24.04 is in NVIDIA Container Toolkit's
tested platform matrix and is an explicitly supported Docker Engine platform;
CachyOS/Arch is absent from both vendor support lists. NVIDIA says other
distributions may work, and Arch supplies Docker and
`nvidia-container-toolkit`, but that is a community-supported path rather than
the least-friction vendor path.
Sources: [NVIDIA Container Toolkit platform support](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/supported-platforms.html),
[Docker Engine supported platforms](https://docs.docker.com/engine/install/),
[Arch Docker documentation](https://wiki.archlinux.org/title/Docker).

## Why Windows/WSL2 wins for this repository

| Consideration | Windows 11 + WSL2 + Docker Desktop | CachyOS native |
|---|---|---|
| Existing evidence | Exact host path already passed restricted CUDA-container smoke, BF16 parity, and quantization | Must be qualified from zero |
| Existing automation | All host orchestration is `.ps1`; reports contain Windows paths; Phase 0 explicitly inventories WSL | PowerShell Core might run much of it, but portability is untested and Phase 0 contains Windows/WSL-specific inventory |
| GPU-container setup | Docker officially supports NVIDIA GPU-PV only through its WSL2 backend; the current machine has already passed it | Direct GPU access is simpler after setup, but requires a Linux NVIDIA driver, Docker, and NVIDIA Container Toolkit |
| Vendor-tested host | Docker Desktop/WSL2 is a documented path | Arch/CachyOS is not in Docker's or NVIDIA Container Toolkit's tested distro list |
| Filesystem and container overhead | Cross-filesystem Windows-to-Linux bind mounts are slower | Native filesystem and cgroups are cleaner |
| Performance telemetry | Good enough for the planned throughput/VRAM measurements, but WSL CUDA/NVML has documented limitations | Better choice for low-level Linux and GPU profiling |
| Maintenance risk | Keep Windows driver, WSL, and Docker Desktop current | CachyOS is rolling; its own docs note NVIDIA's out-of-tree module can be incompatible with the latest kernel and may need downstream patches |

Docker documents that Windows GPU containers require the WSL2 backend and an
up-to-date Windows installation, NVIDIA driver, and WSL kernel. NVIDIA's WSL
guide also makes driver ownership comparatively simple: install the Windows
NVIDIA driver and **do not** install a Linux display driver inside WSL. Existing
Linux CUDA applications can then run through the driver exposed into WSL.
Sources: [Docker Desktop GPU support](https://docs.docker.com/desktop/features/gpu/),
[NVIDIA CUDA on WSL](https://docs.nvidia.com/cuda/wsl-user-guide/).

CachyOS does automate NVIDIA setup through `chwd` and ships precompiled NVIDIA
modules, including open modules appropriate for current GPUs. That is useful,
but its kernel documentation explicitly says the out-of-tree NVIDIA module does
not follow the kernel release cadence and the stock configuration can
occasionally be incompatible with the latest kernel. Its troubleshooting guide
also treats regular command-line package maintenance as part of running the
rolling system. Those are manageable costs for a CachyOS enthusiast, but they
are not evidence for “less headache” on an evaluation appliance.
Sources: [CachyOS hardware detection](https://wiki.cachyos.org/features/chwd/chwd/),
[CachyOS kernel and NVIDIA modules](https://wiki.cachyos.org/features/kernel/),
[CachyOS maintenance guidance](https://wiki.cachyos.org/cachyos_basic/faq/).

Native Linux does have two real advantages. First, it avoids Docker Desktop's
VM and Windows/Linux filesystem boundary. Docker and Microsoft both recommend
keeping bind-mounted Linux workloads on the Linux filesystem because access to
Windows-mounted files is slower. Second, NVIDIA documents incomplete WSL2
support for features such as full Unified Memory, pinned system memory, and
some NVML queries. These are reasons to use native Linux for specialized
profiling or an optional EvalPerf campaign, not reasons to migrate before the
small E1 generation slice. EvalPlus itself labels EvalPerf as Unix-only and
requires performance-monitor access.
Sources: [Docker WSL filesystem best practices](https://docs.docker.com/desktop/features/wsl/best-practices/),
[Microsoft WSL filesystem performance](https://learn.microsoft.com/en-us/windows/dev-environment/wsl-interop#file-system-interop),
[NVIDIA WSL limitations](https://docs.nvidia.com/cuda/wsl-user-guide/#known-limitations-for-linux-cuda-applications),
[EvalPlus official README](https://github.com/evalplus/evalplus#code-efficiency-evaluation-evalperf-nix-only).

## Recommended Windows operating model

1. Keep **Windows PowerShell as the trusted control plane** and Docker Desktop's
   Linux engine as the execution plane. Do not install and run a second Docker
   Engine inside the Ubuntu WSL distribution; Docker explicitly warns that the
   duplicate installations can conflict. Docker commands are already available
   from a Windows terminal with the WSL2 engine.
   Source: [Docker Desktop WSL2 backend](https://docs.docker.com/desktop/features/wsl/).
2. Before an evaluation session, run `wsl --update`, start Docker Desktop, and
   repeat the restricted GPU smoke. At the time of this review, WSL2 and the
   host GPU were available, but the Docker daemon was stopped; that is a simple
   startup prerequisite, not a reason to change operating systems.
3. Do not move the repository or the already-hashed model artifacts merely to
   optimize filesystem throughput before E1. Its workload is one model load and
   seven small requests. For later large image builds or file-heavy scoring,
   put dependency trees and temporary work inside the container image,
   Docker-managed volumes, or WSL's ext4 filesystem rather than repeatedly
   traversing `D:` from Linux. Preserve and recheck hashes if any immutable
   artifact is copied.
4. Run quality comparisons on one host and one pinned container stack. Treat OS
   as a controlled variable. Do not merge CachyOS and WSL2 latency samples into
   one table.
5. For timed runs, record the Windows build, WSL version/kernel, Docker Desktop
   version, NVIDIA driver, image ID, power state, and thermal state. Sample host
   GPU memory/power from the trusted host as well as retaining llama.cpp's
   buffer logs, because WSL's NVML view is not feature-complete.
6. If the real deployment experience will be native Windows/LM Studio, add one
   small post-selection deployment check there. Keep the scientific
   BF16/quantization matrix on the pinned llama.cpp container so runtime changes
   do not contaminate the model comparison.

## Assessment of the evaluation plan

The core plan is unusually strong. Its best decisions are:

- separating fine-tune, conversion, weight-quantization, and KV-cache effects;
- saving raw evidence before scoring and keeping raw completions separate from
  extracted code;
- using the same llama.cpp client/runtime across GGUF cells;
- separating GPU generation from no-GPU code execution;
- pinning artifacts, prompts, datasets, images, and upstream scorer revisions;
- treating benchmark contamination as a first-class result dimension;
- refusing to call successful 128K allocation proof of useful 128K context;
- starting with a three-task, generation-only vertical slice before executing
  model-written code.

Those choices are reflected consistently in the
[benchmark methodology](../../benchmark_guide.html),
[harness guide](../../evaluation_harness_guide.html), and frozen
[E0 contract](../evaluation/e0-contract.md). The checked prompt file's SHA-256
matches the suite (`bd97a6ce...a8a8a788`), and the extracted HumanEval+ file's
hash and 164-record count match the lock and suite. The suite is therefore at a
sensible boundary: E0 is frozen, but the harness is not yet implemented.

### Changes to make before scaling up

1. **Resolve the E1 replay-record ambiguity.** The acceptance gate requires one
   case to be repeated after a server restart, while also requiring exactly four
   prompt-integrity and three HumanEval+ terminal records and rejecting duplicate
   generation keys. Define the replay as a separate run ID or a distinct
   `determinism_replay` record with an explicit reference to the original. Do not
   append an ambiguous eighth duplicate to the primary run.
2. **Create a same-runtime upstream baseline.** The final fine-tune claim is
   cleanest if the original Liquid AI model is also converted to a validated
   BF16 GGUF and served by the exact llama.cpp image. Otherwise the A/B
   Transformers comparison and the C/D llama.cpp comparison are individually
   useful, but there is no single-runtime base-to-deployment comparison. Keep
   the existing token-parity proof as a conversion gate, not as a substitute
   for the full task-level baseline.
3. **Name the sandbox mechanism, not only its policy.** An outer scoring
   container gives batch-level CPU/RAM/PID limits, while the plan promises
   per-sample limits. E2 should specify how each candidate gets a wall timeout,
   CPU/address-space/file/output/process limits, a fresh working directory, and
   process-group cleanup. Validate the path with known pass, wrong-answer,
   syntax-error, timeout, memory, output, and fork/process fixtures before real
   completions. EvalPlus documents its own time, memory, and parallelism
   controls, but the project still owns the outer no-network/read-only/no-GPU
   boundary. Source: [EvalPlus program execution](https://github.com/evalplus/evalplus/blob/master/docs/execution.md).
4. **Keep E1's network model internally consistent.** `--network none` leaves
   only loopback, so the E1 runner and `llama-server` need to share the same
   generation container. Publishing port 8080 to a Windows-host runner would
   weaken or contradict the offline contract.
5. **Predeclare numeric release gates before sealed data.** The documents say
   to predeclare acceptable loss but do not yet contain the actual pass@1,
   paired-retention, failure-rate, context-accuracy, latency, or memory
   thresholds. Use an unscored pilot to choose feasible thresholds, then freeze
   them before the sealed run. Report task counts and paired gained/lost cases,
   not only rounded percentages.
6. **Separate internal comparisons from leaderboard comparability.** The custom
   fenced-code prompt, decoding parameters, output budget, sanitizer, and task
   exclusions are legitimate for paired project evaluation, but absolute scores
   are comparable to public leaderboards only when the complete official
   protocol matches. LiveCodeBench, for example, versions its releases, has
   explicit prompt/model styles, and notes that evaluation timeouts can move
   scores. Sources: [LiveCodeBench official runner and release semantics](https://github.com/LiveCodeBench/LiveCodeBench#inference-and-evaluation),
   [EvalPlus prompt/backend behavior](https://github.com/evalplus/evalplus#llm-backends).
7. **Avoid the full Cartesian product.** The proposed suite multiplied by six
   model formats, three KV-cache types, several context lengths, two decoding
   tracks, and five seeds will become expensive and hard to interpret. Screen in
   stages:
   - E1: merged BF16 GGUF, seven generation-only cases;
   - E2: the same cases plus isolated extraction/scoring fixtures;
   - pilot: original BF16 GGUF, merged BF16 GGUF, Q8_0, and Q6_K on 30-50
     development tasks;
   - full coding suite: base, merged BF16, and only the two strongest quants;
   - regression suite: base, merged BF16, and the selected deployment quant;
   - long context: base, merged BF16, and the selected quant, with a small KV
     cache screen before the full 32K/64K/128K run.
   Q2_K should remain a clearly labelled extreme-memory diagnostic unless it
   unexpectedly survives the pilot.
8. **Freeze a performance protocol separately from the quality protocol.** E1's
   `--no-warmup` is fine for a plumbing test, but final performance reporting
   needs distinct cold-load and warmed steady-state measurements, a fixed run
   order or counterbalanced order, AC power, controlled background GPU use, and
   repeated samples. Do not interpret a WSL2-vs-CachyOS difference as a model
   difference.

## Immediate next move

Implement and run E1 exactly as frozen on the current Windows/WSL2/Docker
Desktop stack. The only contract clarification needed first is where the
post-restart replay record lives. After E1 passes and one record can be audited
by hand, build E2's extractor and scorer boundary. Operating-system migration,
128K work, and the broad benchmark suite should wait until that seven-request
vertical slice is boringly reliable.
