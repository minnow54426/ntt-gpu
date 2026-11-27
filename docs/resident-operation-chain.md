# Keep the NTT chain on the device

Upload two inputs, compose their operations on the GPU and download the final
result:

```text
upload(a, b) → forward(a), forward(b) → pointwise multiply → inverse → download
```

A session is one batch of work with its own resources. It owns a GPU stream
(an ordered work queue) and returns handles, which let callers refer to stored
polynomials. Operations leave their inputs unchanged. Queued work keeps input
buffers alive even if the caller discards a handle. An upload first copies the
caller’s array into a temporary CPU buffer owned by the session.

**Status:** the recorded Ada runs checked resident NTT and ModUp/ModDown.
The [November 24 report](paired-ntt-native-analysis.md#what-changed-in-the-november-24-ada-results)
contains the latest native measurements. Newer Hybrid adapters and runtime
changes pass offline checks but require another native run. Driver/context
failure handling, production ownership requirements and RTX 5090 acceptance
remain open.

## What is implemented

| Part | Concrete implementation |
|---|---|
| Public prototype | [`resident.hpp`](../include/ntt/resident.hpp): sessions, immutable polynomial handles, completions and owned host results |
| Runtime and registry | [`resident.cpp`](../src/resident.cpp): private allocation provenance, dependencies, memory reservations, operation journal and cleanup |
| Transport boundary | [`resident_driver.hpp`](../include/ntt/resident_driver.hpp): device buffers/events and a single-stream command interface |
| CUDA adapter | [`resident_adapter.inc`](../src/cuda/resident_adapter.inc), included by [`ntt_kernels.cu`](../src/cuda/ntt_kernels.cu), reuses the existing kernels |
| Deferred host transport | [`resident_host_driver.hpp`](../tests/support/resident_host_driver.hpp): queues CPU work for ownership/failure tests; never represents GPU execution |
| Opt-in OpenFHE bridge | [`openfhe_hybrid_adapter.hpp`](../include/ntt/openfhe_hybrid_adapter.hpp): explicit key-switch, relinearization and rotation entry points |

Each session fixes N and the exact ordered list of primes and roots. This
list is its basis. Uploaded arrays must identify whether they contain
coefficients in ordinary order or evaluations in the required bit-reversed
order. Each `uint64_t` stores a remainder from 0 to q−1, called a canonical
residue. A subview selects whole prime towers from an existing buffer; it
keeps that buffer and its completion information alive. Transforms allocate
new outputs instead of overwriting their inputs.

Use one host thread per session, including handle access and destruction.
The API is separate from the abstract [proposed Backend](../include/ntt/backend_api.hpp).
The OpenFHE bridge is opt-in; production dispatch, caching, automatic CPU
fallback, cross-device operations and multi-stream execution remain planned.

For conversion, an explicit-layout `upload(layout, values)` can select a
contiguous whole-tower subset of the session's basis. Coefficients use natural
order; an explicitly declared evaluation layout uses bit-reversed order.
The runtime validates parameter equality, tower order, length and every
uploaded residue before queuing the copy.

## Compose a product

On a real GPU, after creating the validated coefficient layout from the
[D9 contract](device-backend-contract.md):

```cpp
#include "ntt/resident.hpp"

auto session = ntt::resident::make_cuda_session(coefficient_layout, 0);
auto da = session.upload(a);
auto db = session.upload(b);
auto fa = session.forward(da);
auto fb = session.forward(db);
auto product = session.pointwise(fa, fb);
auto coefficients = session.inverse(product);
auto output = session.download(coefficients);
auto values = output.wait_values();
session.close();
```

`values` is a read-only view of memory owned by `output`. Keep `output` alive
while reading that view. `wait_values()` waits until the download finishes
and checks that every value is in the allowed range. Matching the values
against OpenFHE is a separate test.
`session.multiply(a, b)` composes the same chain and validates both external
arrays before the first command is submitted.

For an independent inverse input, `upload_evaluation(values)` explicitly
declares a bit-reversed evaluation array. It does not transform or reorder it.
Every input must use the session’s complete prime list in the same order,
with each value in range. `subview()` can then select adjacent whole towers.
It cannot remove towers from the middle: for example, an array with active
Q_l towers followed by P must be packed explicitly if unused Q towers lie
between them.

## What moves, and when

For the complete product, the runtime records seven operations and six fresh
operand/result allocations:

| Transfer or operation | Count |
|---|---:|
| Host-to-device input uploads | 2 |
| Device-to-device copies for immutable outputs | 4 |
| Device-to-host final download | 1 |
| NTT/pointwise kernel launches at N=65536 | 8 |

Twiddle tables, moduli and inverse lengths are uploaded once during session
setup in four additional constant-array copies. These are separate from the
input/output counts. Each forward or inverse first copies its input to fresh
device storage; pointwise multiplication likewise preserves both inputs.

Every CUDA copy, kernel and completion event in the adapter uses one explicit
nonblocking stream. The existing launcher helpers accept that stream while
their old callers keep their default behavior. There are no intermediate
host-array downloads or per-operation host scans of resident coefficients.

The prototype uses legacy `cudaMalloc`/`cudaFree` and owned pageable host
staging. These can introduce synchronization or staging costs. No copy/compute
overlap, allocation pool, cache reuse or performance improvement is claimed.
Measure those costs on named hardware before optimizing them.

## Ownership and failure rules

A private registry issues handles and validates their allocation/session IDs,
layout and view range. Session IDs are unique and allocation IDs are never
reused within a session. Copying a descriptor cannot create a trusted handle.
Wrong sessions, domains, extents and noncanonical uploads fail before enqueue.

Before submitting an operation, the runtime retains its input, output and
temporary buffers, along with any earlier work it depends on. These retained
references are called pins. Discarding a public handle cannot free memory
that queued work still uses. When a section of the queue finishes, the runtime
releases its pins. It keeps a record of a failed submission even when the
caller never received a completion handle.

In [`resident.cpp`](../src/resident.cpp), admission and submission have separate
jobs:

1. `open_state()` rejects a moved, closed or poisoned session.
2. The operation checks its input layout, reserves output capacity and calls
   `begin()` to retain every resource before native work.
3. Upload, transform, pointwise and join use `submit_output()`: create the event,
   allocate the reserved output, enqueue the operation, record completion and
   construct the result under one failure guard.
4. A throw at any submission step poisons the session and drains or retains
   its resources through the existing failure path.

Conversion constants and two-output Hybrid operations keep explicit submission
code because they pin additional resources. Diagnostic methods and `close()`
remain callable after admission has stopped.

| State | Required behavior |
|---|---|
| Preflight or admission refusal | Submit no command for that operation; report the error |
| Pending work | Retain allocation, staging, event and dependency leases |
| Successful completion | Release that prefix's in-flight pins; external handles keep their own storage |
| Submission, query, wait or output-validation failure | Reject new work, retain the failure and attempt to drain |
| Cannot prove queued work has stopped | Keep the driver and complete resource journal alive; do not free queued storage |
| Later successful drain | Reclaim pins; explicit close still reports the original failure |

`Completion::query()` reports a failed outcome when a transport error is found;
`wait()` propagates the saved error. `close()` stops submission, drains and
invalidates polynomial handles. An already completed owned host result can
outlive the session. Destruction is nonthrowing; call `close()` explicitly to
observe errors.

If work cannot be proved stopped, resources are deliberately retained until a
later successful close or process exit. This is conservative failure handling,
not a finished recovery service. Final deallocation errors are best-effort in
destructors; prior recorded cleanup errors are checked during explicit close.
Actual failed-context behavior still needs native verification.

## Bounded resources

Defaults are **256 MiB device storage**, including fixed CUDA tables,
**128 MiB owned host staging**, and **1,024 operation records** per session.
The CUDA setup also checks its temporary table staging against the host limit
and releases it after setup synchronization. Payload reservations exist before
native allocation and follow the resource's actual lifetime.

After releasing buffers, the operation log keeps a small record of each
completed operation. Completed work still counts toward the operation limit.
Start a new session for another batch. These are prototype session limits, not a device-wide
cache or a replacement for [B07 admission/cache work](implementation-backlog.md#b07-cache-and-memory-admission).

## Observed checks

- [Host control tests](../tests/resident_tests.cpp) cover independent negacyclic convolution,
  immutable inputs, owned uploads, explicit/implicit dependencies, nested views,
  foreign/closed handles, budgets, every synchronous transport step, failures
  after enqueue, asynchronous errors, abandoned outputs and quarantine recovery.
  The conversion additions check every transport failure point, every failure
  after an enqueue, parameter-table budgets and retention through a failed drain.
- The same control tests pass **AddressSanitizer, UndefinedBehaviorSanitizer
  and leak detection** on Linux.
- The original NTT/product fixture matches pinned OpenFHE on **4,456,448 residues**
  at N=65536 with an unsorted 55/50/60-bit basis. Cases cover zero, bounds,
  monomials and dense values, forward, independent inverse, roundtrip, product,
  input immutability and tower views.
- The extended [conversion fixture](../tests/support/resident_conversion_cases.hpp)
  checks both ModUp input formats, an independently generated ModDown input,
  composed operations, unchanged inputs and nonzero-offset tower views. It
  covers N=64/4096/65536, five active Q levels, mixed 30/50/55/60-bit primes,
  four coefficient patterns and an additional 40-Q/three-P dense case.
- Stock CUDA 13 compiles the adapter and the prepared native test for
  **sm_89 and sm_120**. Five invalid CUDA-linked factory inputs reject without
  requiring a driver. These checks execute no device arithmetic.

The [exact fixture](../tests/resident_openfhe_tests.cpp) uses real OpenFHE
`NativePoly` format conversion and multiplication. Its transform path reaches
[`NativePoly::SwitchFormat`](https://github.com/openfheorg/openfhe-development/blob/b59e1f291965f14fbee5172cd83a20f5b856204f/src/core/include/lattice/hal/default/poly-impl.h#L454-L474)
through the format API. Root/modulus selection comes from the pinned OpenFHE
implementation; complete arrays are compared, not approximate CKKS decryptions.
The recorded CUDA NTT/product and conversion runs used reported compute 8.9.
Their evidence is tied to the source versions in the journal. The current
runtime cleanup and newer Hybrid adapters need the next native run; compilation
for sm_89/sm_120 does not supply that evidence.

## Reproduce and verify next

With the pinned OpenFHE install available, on the CPU workspace:

```sh
cmake -S . -B build/native-cpu -DCMAKE_BUILD_TYPE=Release \
  -DNTT_ENABLE_CUDA=OFF -DNTT_REQUIRE_CUDA=OFF \
  -DNTT_ENABLE_OPENFHE_COMPAT=ON \
  -DCMAKE_PREFIX_PATH="$PWD/build/openfhe-compat/install"
cmake --build build/native-cpu -j2
ctest --test-dir build/native-cpu -R '^resident_' --output-on-failure
```

The [session collector](offline-test-tooling.md#collector-and-validation) now
requires `resident_runtime_host`, `resident_openfhe_host`, the CUDA-linked
preflight, and the GPU-labelled `resident_cuda_exact` in the applicable modes.
On a rental, it also runs a dense resident smoke case with each of the four
Compute Sanitizer tools. Skipping sanitizers remains an explicit limited run.

On the next RTX 5090 rental, run the current GPU answer and error-handling
tests, the new resident CUDA test, and the memory/threading sanitizer tools.
Then use a profiler to measure where execution time goes.
[D11/P2](openfhe-integration-plan.md) remains open for primary-device acceptance,
driver/context failure handling and hardware-counter profiling.
