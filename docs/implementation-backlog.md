<div id="implementation-backlog-after-d10" aria-hidden="true"></div>

# Implementation tasks and acceptance checks

The design has been accepted, but the production GPU implementation is
still being built. Each task below lists work to do and the checks required
to call it complete. Those checks are its acceptance criteria. The
[roadmap](openfhe-integration-plan.md) gives phase order; [ADR 0001](architecture-decision.md)
defines the behavior each task must preserve.

**Current state:** D11/P2 remains open. The temporary Ada baseline is measured;
RTX 5090 acceptance and remaining native failure/profiling checks are pending.
Tasks B01–B04 have local preparation and some results from real GPU runs.
B06 has a limited prototype verified on reported compute 8.9. B05 and B09
also have native arithmetic/conversion checks from the November 19 batch. Production
work on B07–B12 remains open. Preparing part of a later task does
not complete its phase. B labels a backlog task, P a phase, and D a
working-day milestone; these are not calendar dates.

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

**Evidence so far:** the November 14 [Ada diagnostic](journal/2026-11-14.md)
recorded native execution on a machine reporting RTX 4080 SUPER. It is neither
the primary RTX 5090 nor the planned RTX 4090 control.

### B02 — native build and baseline safety

Work in [cuda_ntt.hpp](../include/ntt/cuda_ntt.hpp) and
[ntt_kernels.cu](../src/cuda/ntt_kernels.cu). Give allocated resources owners that release them automatically, including
when construction fails (RAII). Remember which GPU owns them and select it
for each runtime call with a device guard. Before queuing work, check array
sizes for overflow and require every input residue to be in 0…q−1.
Resolve compiler/architecture blockers exposed by the first native build.
Keep the baseline arithmetic and transform convention intact.

**Evidence so far:** unique ownership, device guards and input validation
are implemented. Host tests cover six allocations and four uploads; the older
Ada batch passed the native pre-call injection probe. That does not establish
actual driver OOM, partial-transfer or failed-context cleanup safety. The
[tooling guide](offline-test-tooling.md#shared-allocation-ownership-and-injected-failures)
separates these scopes.

**Acceptance:** native build and smoke run; injected failures at each allocation
and constant upload unwind all prior allocations; invalid inputs submit no work;
normal/destructor cleanup uses the owning device and does not throw. Update the
[D5 source-contract tests](../tests/docs/test_cuda_dataflow.py) and buffer map to
match the actual fix. Static source checks alone cannot complete this item.

### B03 — exact CUDA matrix

**Evidence so far:** the older Ada batch passed the direct OpenFHE matrix,
including 169,279,488 original-gate comparisons, the expanded benchmark matrix
and bounded sanitizer smoke. [The report](batch40-native-analysis.md) records
the tested workload and limitations. Repeat the current source on the primary
device; older results do not validate the new resident runtime.

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

**Evidence so far:** the [full-matrix report](batch40-native-analysis.md)
compares measured public-call and resident-event timings. Raw samples,
validated reports and a selected CUDA timeline are retained. Hardware counters
were unavailable, so there is no accepted first bottleneck decision.

The [collector](offline-test-tooling.md#collector-and-validation) now separates
GPU-only and local CPU collection, with hash-checked resume. Host control-flow
checks and a real Linux CPU run validate that split. Current GPU-only collector
and resident-runtime execution still need native verification. Use the
[first-run analysis](first-native-analysis.md) only for its earlier 46-bit sample.

Separate construction/table upload, device-resident CUDA-event timings,
transfer-inclusive wall time, and explicit synchronization. Add raw samples,
warm-up, median/p5/p95, workload metadata and `ptxas` resource reports. Profile
the strided column pass's sectors/requested byte, L2/DRAM traffic, bank conflicts,
occupancy, registers/spills and launch gaps.

The [session collector](../scripts/gpu_session.py), called by the
[AutoDL helper](../scripts/autodl_setup.sh), now requires CUDA/OpenFHE coverage,
retains logs and propagates failed stages, including benchmarks. Host
control-flow tests exercise those failure classifications. Expanded
batch tests cover partial completion, strict input/scope matching, and hashed
restart records. Optional bounded Nsight captures and compiler resource output
are prepared, not executed here. The old 46-bit wall-clock row does not complete B04.

**Acceptance:** reproducible raw results, exact-gate status and one evidence-backed
first bottleneck decision on named hardware. Follow the
[benchmark plan](benchmark-tuning-plan.md). Do not label the current wrapper's
transfer-inclusive wall time as kernel-only time.

## 3. P3: implement resident NTT safely

### B05 — arithmetic policies

Offline preparation on November 19 adds canonical Shoup multiplication,
independent bigint checks and a native CUDA intrinsic probe. That probe passed
on reported compute 8.9 in the November 19 batch. The current NTT
dispatch keeps its reference arithmetic while the candidate is validated.
See [conversion preparation](basis-conversion-preparation.md).


Keep the simple exact implementation for checking answers. Add Shoup
multiplication for precomputed constant multipliers and Barrett reduction
for two changing operands. Prove their valid input ranges and generate the
required tables before selecting either optimization.
Run independent bigint, maximum-width adversarial, complete-transform and
native OpenFHE comparisons before selecting optimized dispatch.

**Acceptance:** no hidden integer-remainder path in selected optimized kernels;
range/overflow tests and hardware-specific exactness evidence accompany each
policy. Alternative encodings need a superseding contract, not a changed label.

### B06 — resident runtime and registry

The November 19 batch passed the resident exactness and selected sanitizer
checks on reported compute 8.9. Driver/context failure handling and primary
RTX 5090 acceptance remain open.


Offline preparation 2026-11-16: the [resident-chain prototype](resident-operation-chain.md)
implements private handles, one-stream dependencies, immutable outputs, bounded
reservations, owned staging and failed-work quarantine. Its deferred host
transport passes OpenFHE comparisons and lifetime/error tests; the real CUDA
adapter compiles for sm_89/sm_120. Native CUDA execution, actual failed-context
cleanup, broader concurrency/cache behavior and production API acceptance are
still open. This is not a completed B06 or a P3 phase transition.

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

A plan contains reusable tables for an operation. Identify it with every
parameter that affects its values. Allow only one pending upload per plan,
and make later readers wait until it is ready. Keep it alive while callers
or queued work use it. Remove least-recently-used (LRU) entries only after
those users finish.
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

Offline preparation on November 19 adds a reusable coefficient-domain switch,
all-dimension CPU/OpenFHE fixtures through 40 towers and a prepared native CUDA
probe. The native conversion probe passed its full OpenFHE matrix on
reported compute 8.9 on November 19. Hybrid tests exercise the candidate's digit lifts and P-to-Q correction
against independent BigInteger answers. Resident GPU composition and primary-device
acceptance remain open.


Upstream entry points: [`ApproxSwitchCRTBasis`](https://github.com/openfheorg/openfhe-development/blob/b59e1f291965f14fbee5172cd83a20f5b856204f/src/core/include/lattice/hal/default/dcrtpoly-impl.h#L938-L982) and [`ApproxModDown`](https://github.com/openfheorg/openfhe-development/blob/b59e1f291965f14fbee5172cd83a20f5b856204f/src/core/include/lattice/hal/default/dcrtpoly-impl.h#L1020-L1055).

Broaden D8's model into independent Q→QP and QP→Q CPU fixtures across the intended
levels, tower counts and boundary coefficients. Freeze exact constants, integer
representatives, reduction order, and separate digit-extension/ModDown plan kinds.
Then implement conversion for batches of polynomials using views of arrays
already stored on the GPU, so intermediate results need not return to the CPU.

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
Store the reusable digit decomposition as an explicit read-only result
(a hoist). Record which input allocation and version produced it, so it
cannot be reused after the input changes.

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
  -DNTT_REQUIRE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=native \
  -DNTT_ENABLE_OPENFHE_COMPAT=ON \
  -DCMAKE_PREFIX_PATH="$NTT_OPENFHE_PREFIX"
cmake --build build/native-gpu -j2
ctest --test-dir build/native-gpu --output-on-failure
```

CMake auto-detection can produce a CPU-only build. B02/B03 require an actual
CUDA compiler, the CUDA targets and their registered CTests, a real GPU, and
successful native execution. The commands above are a handoff, not evidence of execution on that GPU.
Compiler-only builds are documented separately in the preparation guide.

For each completed item, save the source revision, test inputs and parameters,
machine and compiler details, exact commands, original logs, any failure
explanation, and the result of each required check. Update this backlog, the roadmap and that
calendar day's journal together. Leave unexecuted items visibly open.
