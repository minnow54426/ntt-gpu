# Implementation backlog after D10

**Accepted by [ADR 0001](architecture-decision.md) · 2026-11-12 · N=65536**

P1's contracts are complete. The next working day is **D11**: provision and
record the native GPU environment. B01–B04 now have
[completed offline preparation](offline-gpu-preparation.md); their native
acceptance gates remain open. B05–B12 are **NOT STARTED**. Design acceptance,
compiler-only builds and portable CPU checks do not satisfy CUDA/runtime gates.

The [95-working-day roadmap](openfhe-integration-plan.md) retains its phase
boundaries. Each item lists a deliverable, dependencies and acceptance criteria.
Update its state with reproducible evidence.

## 1. Deliverables and dependencies

Select a task ID for its acceptance criteria. The comparison scrolls horizontally.

<div class="site-table-scroll" role="region" aria-label="Implementation tasks and dependencies" tabindex="0">
<table style="min-width:42rem">
<colgroup><col style="width:5rem"><col style="width:10rem"><col><col style="width:9rem"></colgroup>
<thead><tr><th>ID</th><th>Planned phase/days</th><th>Deliverable</th><th>Depends on</th></tr></thead>
<tbody>
<tr><td><a href="#b01-environment-record">B01</a></td><td>P2 / D11</td><td>Named GPU environment and reproducibility record</td><td>Accepted ADR</td></tr>
<tr><td><a href="#b02-native-build-and-baseline-safety">B02</a></td><td>P2 / D12</td><td>Native build, constructor safety and validated baseline</td><td>B01</td></tr>
<tr><td><a href="#b03-exact-cuda-matrix">B03</a></td><td>P2 / D13</td><td>Exact native CUDA/OpenFHE baseline matrix</td><td>B02</td></tr>
<tr><td><a href="#b04-timing-and-bottleneck-report">B04</a></td><td>P2 / D14–D15</td><td>Reproducible timing, profiler evidence and first bottleneck</td><td>B03</td></tr>
<tr><td><a href="#b05-arithmetic-policies">B05</a></td><td>P3 / D16–D23</td><td>Checked arithmetic policies and optimized reductions</td><td>B03; B04 informs selection</td></tr>
<tr><td><a href="#b06-resident-runtime-and-registry">B06</a></td><td>P3 / D24</td><td>Resident buffers, explicit stream and owned completions</td><td>B02, B03</td></tr>
<tr><td><a href="#b07-cache-and-memory-admission">B07</a></td><td>P3 / D24–D25</td><td>Plan cache, budget admission and lifecycle regression</td><td>B06</td></tr>
<tr><td><a href="#b08-production-ntt-selection">B08</a></td><td>P3 / D26–D40</td><td>Measured NTT layout/radix selection and production NTT gate</td><td>B04–B07</td></tr>
<tr><td><a href="#b09-base-conversion">B09</a></td><td>P4–P5 / D41–D61</td><td>Broader exact basis-conversion model and GPU implementation</td><td>B08; D8 graph</td></tr>
<tr><td><a href="#b10-hybrid-runtime-keys-and-hoists">B10</a></td><td>P6–P7 / D62–D85</td><td>Production Hybrid fixtures, keys, hoisting and GPU graph</td><td>B09; D8/D9 contracts</td></tr>
<tr><td><a href="#b11-transactional-adapter">B11</a></td><td>P8 / D86–D93</td><td>Transactional OpenFHE adapter and integrated dispatch</td><td>B08–B10</td></tr>
<tr><td><a href="#b12-integration-release">B12</a></td><td>P8 / D94–D95</td><td>Application, fallback and integration release evidence</td><td>B11</td></tr>
</tbody>
</table>
</div>

B02's safety fixes are prerequisites for a trustworthy native baseline. They
are scheduled explicitly inside P2 rather than silently assuming the current
wrapper is failure-safe. B06/B07 implement lifecycle rules before production
resident execution. B11 is the first OpenFHE host adapter; library tests against
OpenFHE in earlier phases do not themselves integrate that adapter.

## 2. P2: establish the native baseline

### B01 — environment record

Record the actual GPU model, memory and compute capability; driver; CUDA
compiler/runtime; host compiler; OS; CMake flags; source commit; OpenFHE commit,
NativeInteger size and build options. Record clocks/power mode and profiler
versions before performance measurements.

Start with RTX 5090; use RTX 4090 as the control. If only a control GPU is
available, label that run and keep the primary-device gate open. Use native
architecture compilation on the selected device; `sm_120` requires an appropriate
toolkit (the current configured target starts at CUDA 12.8).

**Acceptance:** checked-in or attached environment record plus successful device,
compiler and driver queries. A CPU machine, toolkit version string alone, or PTX
fallback without a native build does not complete B01/B02. This task does not
provision or rent hardware as part of D10.

Offline status: the compiler/toolchain record and collection scripts are ready.
Actual GPU/driver discovery and native device execution are still pending.

### B02 — native build and baseline safety

Work in [cuda_ntt.hpp](../include/ntt/cuda_ntt.hpp) and
[ntt_kernels.cu](../src/cuda/ntt_kernels.cu). Replace raw constructor ownership
with exception-safe RAII, retain the owning device, use a device guard, and
validate checked extents and supported canonical inputs before enqueue.
Resolve compiler/architecture blockers exposed by the first native build.
Keep the baseline arithmetic and transform convention intact.

Offline status: unique ownership, device guards and checked host preflight are
implemented; sm_89/sm_120 compilation and driverless invalid-constructor checks
pass. Native allocation/upload fault injection and context-error cleanup remain
required before B02 is complete.

**Acceptance:** native build and smoke run; injected failures at each allocation
and constant upload unwind all prior allocations; invalid inputs submit no work;
normal/destructor cleanup uses the owning device and does not throw. Update the
[D5 source-contract tests](../tests/docs/test_cuda_dataflow.py) and buffer map to
match the actual fix. Static source checks alone cannot complete this item.

### B03 — exact CUDA matrix

Offline status: `cuda_openfhe_exact` is compiled and registered with real
50/55/60-bit OpenFHE and D8 level/basis fixtures. All native cases remain unrun.

Extend the native harness to import the exact OpenFHE fixture moduli, roots,
input arrays and expected arrays. At N=65536 cover 50/55/60-bit primes, active
Q levels and representative tower counts; include D7's deliberately unsorted
50/55/60-bit packing case and D8's actual context parameters where applicable.

Exercise zero, one, q−1, alternating bounds, monomials, deterministic dense
residues, independent evaluation input, forward, inverse, round-trip, pointwise
and negacyclic product. Compare complete arrays against the CPU oracle and
pinned OpenFHE using the same fixtures. Run CUDA Compute Sanitizer memory,
race, initialization and synchronization checks on the affected kernels.

**Acceptance:** every selected case passes exactly with a recorded native build
and no relevant sanitizer errors. The current CLI's generated baseline uses
46-bit parameters; its success alone does not cover this matrix. Keep the
existing CPU/OpenFHE and small schedule-boundary gates green.

### B04 — timing and bottleneck report

Separate construction/table upload, device-resident CUDA-event timings,
transfer-inclusive wall time, and explicit synchronization. Add raw samples,
warm-up, median/p5/p95, workload metadata and `ptxas` resource reports. Profile
the strided column pass's sectors/requested byte, L2/DRAM traffic, bank conflicts,
occupancy, registers/spills and launch gaps.

The [session collector](../scripts/gpu_session.py), called by the
[AutoDL helper](../scripts/autodl_setup.sh), now requires CUDA/OpenFHE coverage,
retains logs and propagates failed stages, including benchmarks. Eleven host
control-flow tests exercise those failure classifications. Actual timings,
CUDA events, raw samples and profiler collection are still pending; its basic
46-bit wall-clock benchmark does not complete B04.

**Acceptance:** reproducible raw results, exact-gate status and one evidence-backed
first bottleneck decision on named hardware. Follow the
[benchmark plan](benchmark-tuning-plan.md). Do not label the current wrapper's
transfer-inclusive wall time as kernel-only time.

## 3. P3: implement resident NTT safely

### B05 — arithmetic policies

Retain the exact oracle path while adding Shoup twiddle multiplication and
Barrett variable multiplication, with their exact range proofs and tables.
Run independent bigint, maximum-width adversarial, complete-transform and
native OpenFHE comparisons before selecting optimized dispatch.

**Acceptance:** no hidden integer-remainder path in selected optimized kernels;
range/overflow tests and hardware-specific exactness evidence accompany each
policy. Alternative encodings need a superseding contract, not a changed label.

### B06 — resident runtime and registry

Implement the accepted metadata through a private registry, move-only allocation
owners, one explicit initial stream, immutable views/results, owned upload and
download staging, and completions. Pin all resources before attempting submission.
Retain failed operation records even when event recording or submission throws
before a completion is returned. Poison/drain or quarantine a failed session.

**Acceptance:** test wrong device/session, stale generations, alias rejection,
input immutability, chained resident forward/pointwise/inverse, completion
abandonment, close-with-live-handles, constructor unwinding, submission failure,
event-recording failure, and failure before quiescence. Confirm a host download
cannot expose data before success. Multi-stream capability stays disabled until
exact cross-stream waits, event-generation handling and race tests pass.

### B07 — cache and memory admission

Implement complete semantic plan keys, a single pending upload per key,
readiness waits, external leases, in-flight pins and eligible-entry LRU eviction.
Apply ADR 0001's separate 256 MiB cache, 256 MiB device workspace and 128 MiB
host staging defaults, with checked allocation-size charging and configurable
limits. The initial allocator owns individual allocations.

**Acceptance:** tests cover concurrent same-key lookup, changed roots/basis/device,
live-reader eviction attempts, failed uploads, pending entries, budget exhaustion,
actual released bytes, all-or-nothing budget reduction and retained output/host
result credits. Busy/budget failures are explicit, bounded and cannot deadlock
waiting for the caller's own lease. Do not enable multi-stream dispatch merely
because the cache lookup mutex is tested.

### B08 — production NTT selection

Evaluate coalesced transpose/Four-Step, radix-4/8, shared-memory padding and
warp exchange only after the baseline is measured. Preserve exact ordering,
roots, normalization and the required out-of-place copy cost.

**Acceptance:** device-resident composition without intermediate host waits;
exact gates for every supported workload; recorded launch/resource parameters;
latency and throughput selections for N=65536 on the primary GPU, with labeled
control-device evidence. P3 closes only with the complete regression and dispatch
record, not one faster microbenchmark.

## 4. P4–P7: preserve exact conversion and Hybrid behavior

### B09 — base conversion

Upstream entry points: [`ApproxSwitchCRTBasis`](https://github.com/openfheorg/openfhe-development/blob/b59e1f291965f14fbee5172cd83a20f5b856204f/src/core/include/lattice/hal/default/dcrtpoly-impl.h#L938-L982) and [`ApproxModDown`](https://github.com/openfheorg/openfhe-development/blob/b59e1f291965f14fbee5172cd83a20f5b856204f/src/core/include/lattice/hal/default/dcrtpoly-impl.h#L1020-L1055).

Broaden D8's model into independent Q→QP and QP→Q CPU fixtures across the intended
levels, tower counts and boundary coefficients. Freeze exact constants, integer
representatives, reduction order, and separate digit-extension/ModDown plan kinds.
Then implement batched GPU conversion over the resident views.

**Acceptance:** every intermediate array matches pinned OpenFHE; shortened digits
and dynamic active prefixes work; NTT and conversion compose without host copies.
Canonical CRT substitution must remain a detectable negative case. Record native
sanitizer and performance evidence before the P5 production gate.

### B10 — Hybrid runtime, keys and hoists

Trace [`EvalKeySwitchPrecomputeCore`](https://github.com/openfheorg/openfhe-development/blob/b59e1f291965f14fbee5172cd83a20f5b856204f/src/pke/lib/keyswitch/keyswitch-hybrid.cpp#L300-L366) → [`EvalFastKeySwitchCoreExt`](https://github.com/openfheorg/openfhe-development/blob/b59e1f291965f14fbee5172cd83a20f5b856204f/src/pke/lib/keyswitch/keyswitch-hybrid.cpp#L389-L421) → [`EvalFastKeySwitchCore`](https://github.com/openfheorg/openfhe-development/blob/b59e1f291965f14fbee5172cd83a20f5b856204f/src/pke/lib/keyswitch/keyswitch-hybrid.cpp#L368-L387). The full wrappers are [`RelinearizeInPlace`](https://github.com/openfheorg/openfhe-development/blob/b59e1f291965f14fbee5172cd83a20f5b856204f/src/pke/lib/schemebase/base-leveledshe.cpp#L315-L329) and [`EvalFastRotationExt`](https://github.com/openfheorg/openfhe-development/blob/b59e1f291965f14fbee5172cd83a20f5b856204f/src/pke/lib/scheme/ckksrns/ckksrns-leveledshe.cpp#L595-L649).

Broaden the production parameter, digit, level and key-family fixtures before
GPU composition. Snapshot immutable A/B evaluation-key material with versioned
identity; cache full-context metadata; preserve reduced-level P offsets, B/A
accumulation, delayed QP scaling and one final ModDown per accumulated result.
Add an explicit immutable hoist result tied to its source allocation/version.

**Acceptance:** complete intermediate/final arrays match for fixed input/key
material; altered key snapshots cannot reuse stale cache entries; stale hoists
are rejected. Actual rotation automorphisms and higher-degree relinearization
need full exact and CKKS decryption tests; D8's source trace alone is insufficient.
Record key upload reuse, peak live memory and full-operation timing.

## 5. P8: integrate and release

### B11 — transactional adapter

Upstream boundaries: [`SetFormat`](https://github.com/openfheorg/openfhe-development/blob/b59e1f291965f14fbee5172cd83a20f5b856204f/src/core/include/lattice/ilelement.h#L474-L478), [`DCRTPoly::SwitchFormat`](https://github.com/openfheorg/openfhe-development/blob/b59e1f291965f14fbee5172cd83a20f5b856204f/src/core/include/lattice/hal/default/dcrtpoly-impl.h#L1956-L1969), [`NativePoly::SetValues`](https://github.com/openfheorg/openfhe-development/blob/b59e1f291965f14fbee5172cd83a20f5b856204f/src/core/include/lattice/hal/default/poly-impl.h#L226-L243) and [`DCRT move assignment`](https://github.com/openfheorg/openfhe-development/blob/b59e1f291965f14fbee5172cd83a20f5b856204f/src/core/include/lattice/hal/default/dcrtpoly.h#L131-L138). These CPU APIs motivate the proposed adapter; they are not GPU adapter implementations.

Implement the explicit opt-in SetFormat-equivalent DCRT seam before integrating
broader operations. Validate every tower and outer format, use per-element
packing/import, reserve the complete transaction's resource credits, and commit
through a verified nonthrowing move only after successful download/import.
Use `RequireDevice` by default; permit CPU only before submission under the
explicit fallback policy. Add later conversion/Hybrid dispatch behind their gates.

**Acceptance:** same-format requests are validated no-ops; CPU-only builds keep
working; unsupported/unavailable/unverified cases obey policy. Recoverable
packing, allocation, upload, launch, event, download and import failures leave
the original host object unchanged. No post-submission automatic fallback occurs.
Test partial submission without a returned completion, concurrent host-object
access rules and upstream `noexcept` limitations explicitly. Application input,
keys and CKKS metadata must survive both device and permitted CPU paths.

### B12 — integration release

Run relevant upstream CKKS examples/tests and the complete exact pipeline,
including all target levels, relinearization, rotations and explicit fallback.
Keep constants/keys resident across repeated operations and publish cold/warm,
end-to-end and kernel-only measurements with raw evidence and source revisions.

**Acceptance:** all P8 exit gates are green; runtime failure and teardown tests
pass; exact arrays and secondary decryption checks agree; per-device dispatch,
limits, configuration and reproducibility instructions are published. Production
promotion removes the proposal label only after these relevant implementation
gates, with a reviewed API migration.

## 6. D11 handoff and evidence record

D11 begins on the provided GPU machine. Record its environment before installing
or changing the selected toolkit. Configure a separate native build directory;
reuse the pinned OpenFHE installation produced by `scripts/openfhe_setup.sh`.
For example, after setting the install prefix for that machine:

```sh
cmake -S . -B build/native-gpu -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=native \
  -DNTT_ENABLE_OPENFHE_COMPAT=ON \
  -DCMAKE_PREFIX_PATH="$NTT_OPENFHE_PREFIX"
cmake --build build/native-gpu -j2
ctest --test-dir build/native-gpu --output-on-failure
```

CMake auto-detection can produce a CPU-only build. B02/B03 require an actual
CUDA compiler, the CUDA targets and their registered CTests, a real GPU, and
successful native execution. The commands above are a handoff, not an observed
D10 GPU run or proof that the current source already compiles with nvcc.

For each completed item, retain the source commit, parameter/fixture identifiers,
hardware/toolchain record, exact commands, raw logs, failing-case diagnosis where
relevant, and the final gate result. Update this backlog, the roadmap and that
calendar day's journal together. Leave unexecuted items visibly open.
