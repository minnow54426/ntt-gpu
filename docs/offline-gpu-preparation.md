# Before renting a GPU

## Prepare profiles without renting

The resident NTT and ModUp/ModDown experiments have separate profiling
executables. NVTX is NVIDIA's timeline-labeling interface. These executables
mark CPU setup, session creation, input uploads, warmups, measured samples,
verification and cleanup in the `ntt-gpu` domain. Each sample also names the
operation it runs. The ordinary benchmark executables use an empty labeling
backend and retain their existing timing boundary.

| Purpose | NTT executable | Conversion executable |
|---|---|---|
| Ordinary GPU measurements | `ntt_resident_bench` | `ntt_resident_conversion_bench` |
| GPU profiles only | `ntt_resident_profile` | `ntt_resident_conversion_profile` |
| CPU checks of profiling code | `ntt_resident_profile_host` | `ntt_resident_conversion_profile_host` |

The profiling variants emit `profiled: true`. The raw-measurement validator
rejects that output, so instrumented times cannot enter the ordinary benchmark
analysis. CPU variants also retain `simulation: true`. The NVTX headers are
vendored at a pinned NVIDIA revision with their license; these builds require
no GPU driver, profiler installation or network download.

In a CUDA-disabled build with pinned OpenFHE enabled, run:

```sh
cmake --build build/cpu --target ntt_resident_profile_host \
  ntt_resident_conversion_profile_host ntt_benchmark_profile_tests
ctest --test-dir build/cpu \
  -R 'benchmark_profile_ranges|resident_profile_host|resident_conversion_profile_host' \
  --output-on-failure
```

Use the build directory from your local setup in place of `build/cpu`.
These checks exercise the exact host-driver calculations, profiling metadata,
range nesting and cleanup on exceptions. They do not establish that a GPU
profiler displays the ranges correctly; that check stays in the later batch.
The batch selects the profiling binaries for Nsight, fingerprints them
separately, and filters counters to the `ntt.sample` range. A whole-process
timeline still includes setup and verification: inspect the labeled sample
and operation before attributing costs.

Sources: [range helper](../include/ntt/benchmark_profile.hpp),
[measurement boundary](../src/resident_benchmark_common.hpp),
[range test](../tests/benchmark_profile_tests.cpp),
[NVTX provenance and license](../third_party/nvtx/README.md), and
[Nsight Compute NVTX filtering](https://docs.nvidia.com/nsight-compute/NsightComputeCli/index.html#nvtx-filtering).

Do the builds, host tests and documentation on a CPU machine. Rent a GPU when
ready to run native CUDA checks and collect measurements. The main target is
RTX 5090 at N=65536; RTX 4090 is the planned control.

A [November 14 Ada run](batch40-native-analysis.md) passed its selected native
tests. The current source includes later changes, especially the resident
runtime, that still need native verification. D11/P2 remains open on the
[roadmap](openfhe-integration-plan.md); working days are milestones, not rental days.

## 1. What is ready

The following results were observed using the November 16 source version
on Linux, without a GPU driver. They describe that version; they do not
claim that later changes have run on a GPU:

| Check | Result |
|---|---|
| CPU/OpenFHE suite | 24 CTests pass with CUDA disabled |
| CUDA-linked host suite | 26 CTests pass; GPU-labelled tests excluded |
| CUDA compilation | Stock nvcc 13.0.88 builds sm_89 and sm_120 |
| Resident prototype | Host ownership/failure tests and exact OpenFHE comparisons pass |
| Native execution of this increment | Pending; compilation is not GPU execution |

The original offline collector record, `bench-results-offline-final/report.json`,
reported `OFFLINE_READY` and `gpu_execution: NOT_RUN`. Its older test counts are
preserved in the [November 12 journal](journal/2026-11-12.md).
Toolchains, binaries and generated reports remain in ignored directories.
Use the [rental survey](gpu-rental-market-survey.md) for dated provider evidence,
then recheck the actual offer and profiling permissions.

## 2. Reproduce the offline preparation

Use Linux x86_64, Python 3.12+, CMake 3.24+ and a compatible C++20 host compiler.
The installer downloads specific versions of NVIDIA’s compiler, runtime
libraries and headers into `build/`. It checks each download’s SHA-256 hash
(a file fingerprint) and keeps the license notices. It does not install a
GPU driver or change system packages or shell settings.

From the repository root, with `cmake`/`ctest` on PATH:

```sh
bash scripts/openfhe_setup.sh
python3 scripts/install_cuda_toolkit.py
python3 scripts/gpu_session.py --mode offline \
  --cuda-root build/toolchains/cuda-13.0.2 \
  --openfhe-prefix build/openfhe-compat/install
```

The OpenFHE helper pins commit
`b59e1f291965f14fbee5172cd83a20f5b856204f`, NativeInteger 64 and reduced-noise off,
and deliberately builds CPU reference tests with CUDA disabled. Its parallelism
defaults to two jobs; set `NTT_BUILD_JOBS` if the machine supports more.
The session runner selects project-local CMake/CTest from `.venv/bin` when present.

The [toolkit lock](../scripts/cuda-toolkit-lock.json) pins NVIDIA components
and SHA-256 values. CUDA 13.0.2 contains compiler version 13.0.88. Matching
installs are reused; incomplete installs or checksum mismatches fail.

<details markdown="1">
<summary>Compiler and dependency portability notes</summary>

On this host, CUDA 12.8.93 conflicts with glibc 2.41's `sinpi`/`cospi`
declarations. Stock CUDA 13 compiles without vendor-header patches. A rental
may use another supported OS/toolkit combination; match its driver, toolkit
and host compiler. Offline compilation does not establish driver compatibility.

Reuse OpenFHE binaries only across compatible OS, glibc, architecture and
compiler settings. Otherwise build the pinned dependency on a matching image
before the session, or rebuild it on the GPU machine.

</details>

## 3. Fixes made before rental

The [synchronous wrapper](cuda-buffer-dataflow.md) now gives every allocated
buffer an owner that releases it automatically when destroyed. This C++
technique is called RAII, and it also handles failures partway through setup.
The wrapper selects the device that owns the buffers, checks N, primes and
roots, and rejects arrays with the wrong length or values outside 0…q−1
before copying them to the GPU. The first nvcc
build also repaired a namespace closure and private-type launcher issue;
transform arithmetic and stage order were retained.

Host tests and CUDA-linked invalid-constructor tests pass without allocating
device memory. Native pre-call setup-failure checks passed in the older Ada
batch. Real GPU tests must still cover out-of-memory errors, interrupted copies,
restoring the caller’s selected device, and cleanup after the GPU context
fails. A nonthrowing
destructor cannot report cleanup errors to its caller.

These fixes do not implement the full [backend proposal](device-backend-contract.md),
cache or transactional OpenFHE adapter. The separate
[resident prototype](resident-operation-chain.md) has its own scope and limits.

## 4. Prepared native exactness matrix

[The CUDA/OpenFHE test source](../tests/cuda_openfhe_tests.cpp) is registered as
`cuda_openfhe_exact`. It compares the CUDA wrapper with the independent CPU reference
and actual OpenFHE [`NativePoly transforms`](https://github.com/openfheorg/openfhe-development/blob/b59e1f291965f14fbee5172cd83a20f5b856204f/src/core/include/lattice/hal/default/poly-impl.h#L454-L474)/products at N=65536. Its prepared cases
include:

- deliberately unsorted 55/50/60-bit primes selected by [`LastPrime`](https://github.com/openfheorg/openfhe-development/blob/b59e1f291965f14fbee5172cd83a20f5b856204f/src/core/include/math/nbtheory-impl.h#L357-L380) and [`RootOfUnity`](https://github.com/openfheorg/openfhe-development/blob/b59e1f291965f14fbee5172cd83a20f5b856204f/src/core/include/math/nbtheory-impl.h#L190-L239);
- the exact D8 Q/P moduli and roots, with Q and QlP bases at L=5,4,3,1;
- zero, one, q−1, alternating bounds, x¹⁷, xᴺ⁻¹ and deterministic dense residues;
- forward, inverse from independent evaluation arrays, round trip, pointwise
  multiplication followed by inverse, and the complete negacyclic product;
- exact complete-array comparisons and unchanged host inputs.

The `--smoke` option runs a small subset: the three prime widths with zero
and dense inputs, under each of the four Compute Sanitizer tools. A smoke
test is a quick check for obvious failures. It does not replace checks for
every supported parameter and error case. All GPU cases are
compiled but **unrun** in this workspace.

The current collector requires seven GPU-labelled gates: `cuda_verify`,
`cuda_matches_ref`, `cuda_oracle_crosscheck`, `cuda_openfhe_exact`,
`cuda_setup_fault_injection`, `cuda_benchmark_matrix_exact` and
`resident_cuda_exact`. The last two cover the later benchmark matrix and
[resident operation-chain prototype](resident-operation-chain.md). The wrapper
constructor and resident factory preflights are separate host tests safe without
a driver. The older Ada batch exercised the benchmark matrix and setup probe;
it predates the new resident path and cannot validate it.

## 5. First session on real hardware

Choose an RTX 5090 image for the primary target. An RTX 4090 can establish the
control run; label it accordingly and leave primary-device evidence open.
Use an image with a matching NVIDIA driver, CUDA compiler, Compute Sanitizer,
and Nsight tools for later profiling. Native architecture mode needs CMake 3.24+.

Use the intended clean source checkout and the pinned OpenFHE install. First
check the image, then run the collector:

```sh
nvidia-smi
nvcc --version
compute-sanitizer --version
NTT_CUDA_ROOT=/usr/local/cuda \
NTT_OPENFHE_PREFIX="$PWD/build/openfhe-compat/install" \
  bash scripts/autodl_setup.sh
```

The helper uses the source and packages already present; it does not pull
new commits or install dependencies. First it records the source revision,
GPU, driver, clock speeds, power settings and build tools. Then it builds
CUDA, verifies that the expected OpenFHE/GPU tests exist, checks the device,
and runs the exact-answer tests. It runs four sanitizer tools on the legacy,
benchmark-matrix and resident-chain smoke cases. Finally it measures
30/50/55/60-bit primes, N=4096–65536 and K=1/3/8/16/24/32/40 towers. Each of
the 140 workloads has two GPU timing scopes, giving 280 jobs. The helper now defaults to
`--benchmark-scope gpu`; request `--benchmark-scope all` for the previous
three-scope, 420-job combined collection.
GPU mode rejects dirty source
unless explicitly marked exploratory with `--allow-dirty`.

For full control, call the Python collector directly:

```sh
python3 scripts/gpu_session.py --mode gpu \
  --cuda-root /usr/local/cuda \
  --openfhe-prefix "$PWD/build/openfhe-compat/install" \
  --architectures native --benchmark --benchmark-scope gpu
```

Collect the matching OpenFHE baseline on a local CPU machine, without CUDA:

```sh
python3 scripts/gpu_session.py --mode cpu \
  --openfhe-prefix "$PWD/build/openfhe-compat/install" \
  --benchmark --output-dir bench-results-local-cpu
```

CPU mode records CPU identity, configures CUDA off, runs host/OpenFHE gates and
collects 140 CPU jobs. It reports GPU execution as `NOT_RUN`. Keep CPU and GPU
records separate, with matching workload parameters and repetition counts.
Both modes support resume within the same clean source/machine/scope context;
changing the selection requires a new output directory.

`--skip-sanitizers` is available for an explicitly limited diagnostic run;
its report says `EXPLICITLY_SKIPPED`. Such a run cannot satisfy sanitizer gates.
Missing tools, missing GPU/OpenFHE tests, build/test/sanitizer/benchmark failures
all produce a nonzero exit and a failed report. Existing directories are rejected
unless explicitly using `--resume --output-dir <previous-record>`; resume archives
the prior report, reruns gates and preserves logs. Open `report.json` and the
named stage logs after every run.

The legacy `bench` command still uses 46-bit primes and public-call wall time.
The collector now separates `bench-jsonl` public calls, `events-jsonl` resident
work/pinned copies and `ntt_openfhe_bench` CPU arithmetic. Recorded Ada runs
checked resident NTT and conversion. Newer runtime and Hybrid paths compile
offline and still need native validation. The CPU comparator is locally verified.
Optional `--profile` needs working `nsys` and `ncu` plus counter permissions;
it prepares a bounded capture, not a completed D14 bottleneck report.
See the [protocol and limits](offline-test-tooling.md).

## 6. When to stop the rental

Use the first GPU session to establish D11 hardware facts, execute D12 smoke and
B02 failure-path work, run D13 exactness, and collect the D14 measurements that
are ready. Export all raw records before stopping the machine. Analyze and
publish results offline for D15.

A report of `GPU_CHECKS_PASSED` means the selected checks passed. Work still
remains: deliberately trigger real GPU failures, cover production parameters,
review the individual GPU-event timing samples, and analyze the Nsight profiles. Completing this offline checklist does not automatically close
D11, B02, B03, D14 or P2. The [backlog](implementation-backlog.md) controls those
acceptance criteria.
