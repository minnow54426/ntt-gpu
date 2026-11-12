# Before renting a GPU

**D11 preparation complete · hardware verification still pending · N=65536**

The compiler, host checks and first-session collection path can be prepared
without paying for a GPU. That preparation is now complete on this Linux
workspace. D11 itself remains open because its final deliverable includes an
actual GPU/driver environment record.

Rent or borrow a GPU when ready to verify that environment and run the prepared
native checks. The [roadmap](openfhe-integration-plan.md) working-day numbers
are engineering milestones, not continuous rental days.

## 1. What is ready

| Work | Observed result |
|---|---|
| CUDA build | Stock CUDA 13.0.88 compiler builds and links sm_89 and sm_120 targets without a driver |
| Host tests | 19 host CTests pass; four registered GPU tests are explicitly excluded |
| CPU build | All 15 CTests pass with CUDA disabled |
| Reports | Eleven control-flow tests cover false success, missing coverage, stage failures and record preservation |
| GPU tests | Compiled against real OpenFHE, with full-size 50/55/60-bit and D8 level/tower fixtures; not executed on a GPU |
| GPU probe | Correctly refuses this driverless environment; no GPU arithmetic result is claimed |

The local preparation report is `bench-results-offline-final/report.json`,
with status `OFFLINE_READY` and GPU execution `NOT_RUN`. Generated toolchains,
binaries and reports stay in ignored directories; source and instructions are
published. CI also compiles CUDA without a GPU and runs only the host gates.

## 2. Reproduce the offline preparation

Use Linux x86_64, Python 3.12+, CMake 3.24+ and a compatible C++20 host compiler.
The pinned installer downloads NVIDIA's compiler/runtime/header components,
verifies each SHA-256, retains component notices and installs under `build/`.
It installs no driver and changes no system packages or shell configuration.

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

The [toolkit lock](../scripts/cuda-toolkit-lock.json) records exact upstream
components from NVIDIA's CUDA 13.0.2 redistributable manifest. The compiler's
component version is 13.0.88. Existing matching installs are reused; conflicting
or incomplete destinations and checksum mismatches fail visibly.

On this host, CUDA 12.8.93 failed its compiler-identification build because its
math headers conflict with glibc 2.41's `sinpi`/`cospi` declarations. Stock CUDA
13.0 compiles successfully here; no vendor-header patch was applied. A rental
image may use a different supported OS/toolkit combination. Match its driver,
CUDA toolkit and host compiler rather than assuming that this compiler-only
installation establishes driver compatibility.

OpenFHE libraries and build directories are not universally portable between
machines. Reuse binaries only with compatible OS, glibc, architecture and
compiler settings; otherwise build the pinned dependency on a matching CPU
machine/image before the paid session, or rebuild it on the GPU machine.

## 3. Fixes made before rental

The first real nvcc compilation exposed a missing closing namespace in the CUDA
source. The launch helpers also named a private nested implementation type from
free functions; they now deduce that internal type without exposing it publicly.
The shared schedule and modular arithmetic algorithms are unchanged.

The synchronous wrapper now:

- owns `Impl` through `unique_ptr`, so constructor exceptions destroy partial state;
- captures the owning device and selects/restores it around operations and cleanup;
- checks prime moduli below 2⁶³, canonical primitive roots, checked buffer sizes,
  nonempty matching tower lists and power-of-two N in [2,65536] before CUDA calls;
- rejects wrong-length or noncanonical operands before upload;
- rejects unsafe benchmark exponents/counts before shifting or allocating;
- reports device information through `ntt_cuda_cli info` when a usable GPU exists.

The [D5 map](cuda-buffer-dataflow.md) reflects the updated ownership. Host tests
exercise validation, including three invalid constructors linked against the
real CUDA-compiled wrapper. They do not allocate device memory. Device selection,
partial allocation/upload failures, launch errors and cleanup under a failed
context still need native fault-injection tests. Cleanup selection/free errors
cannot be reported by the nonthrowing destructor; this remains a production
lifecycle limitation, not an accepted asynchronous backend implementation.

The compiler-only checks do not implement the [D9 backend](device-backend-contract.md),
its caches, or the transactional OpenFHE host adapter.

## 4. Prepared native exactness matrix

`cuda_openfhe_exact` compares the CUDA wrapper with the independent CPU reference
and actual OpenFHE NativePoly transforms/products at N=65536. Its prepared cases
include:

- deliberately unsorted 55/50/60-bit primes with actual OpenFHE minimum roots;
- the exact D8 Q/P moduli and roots, with Q and QlP bases at L=5,4,3,1;
- zero, one, q−1, alternating bounds, x¹⁷, xᴺ⁻¹ and deterministic dense residues;
- forward, inverse from independent evaluation arrays, round trip, pointwise
  multiplication followed by inverse, and the complete negacyclic product;
- exact complete-array comparisons and unchanged host inputs.

The `--smoke` option limits that executable to the three-width context with zero
and dense patterns for the four Compute Sanitizer tools. This is a bounded smoke
scope, not the complete production sanitizer/parameter matrix. All GPU cases are
compiled but **unrun** in this workspace.

The existing `cuda_oracle_crosscheck`, `cuda_matches_ref` and `cuda_verify` gates
remain. All four native tests carry the `gpu` CTest label. The wrapper-constructor
preflight is a separate host test and is safe without a driver.

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

The helper no longer performs an implicit git pull or package installation.
It records the selected commit, hardware/driver/clocks/power data and toolchain,
requires a native CUDA build plus all OpenFHE/GPU tests, probes the actual device,
runs the exact suite and four sanitizer smoke tools, then collects the existing
46-bit transfer-inclusive baseline benchmark. GPU mode rejects dirty source
unless explicitly marked exploratory with `--allow-dirty`.

For full control, call the Python collector directly:

```sh
python3 scripts/gpu_session.py --mode gpu \
  --cuda-root /usr/local/cuda \
  --openfhe-prefix "$PWD/build/openfhe-compat/install" \
  --architectures native --benchmark
```

`--skip-sanitizers` is available for an explicitly limited diagnostic run;
its report says `EXPLICITLY_SKIPPED`. Such a run cannot satisfy sanitizer gates.
Missing tools, missing GPU/OpenFHE tests, build/test/sanitizer/benchmark failures
all produce a nonzero exit and a failed report. An existing report directory is
never overwritten. Open `report.json` and the named stage logs after every run.

The basic benchmark still uses 46-bit primes and wall time around public calls,
including transfers, host validation and device guards. It is not a 50–60-bit
production performance report, CUDA-event timing, or a completed D14 profile.

## 6. When to stop the rental

Use the first GPU session to establish D11 hardware facts, execute D12 smoke and
B02 failure-path work, run D13 exactness, and collect the D14 measurements that
are ready. Export all raw records before stopping the machine. Analyze and
publish results offline for D15.

The collector returning `GPU_CHECKS_PASSED` still leaves native failure injection,
full production coverage, kernel-only raw timing and Nsight profiling as explicit
remaining gates. Completing this offline checklist does not automatically close
D11, B02, B03, D14 or P2. The [backlog](implementation-backlog.md) controls those
acceptance criteria.
