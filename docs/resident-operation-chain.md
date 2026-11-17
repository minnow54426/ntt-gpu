# Keep the NTT chain on the device

Upload two inputs, compose their operations on the GPU and download the final
result:

```text
upload(a, b) → forward(a), forward(b) → pointwise multiply → inverse → download
```

A session owns the stream and returns handles to immutable polynomial buffers.
Queued work retains its inputs even if the caller drops a handle. Uploads copy
caller data into owned staging before returning.

**Status:** the host test transport passes exact OpenFHE and lifetime/failure
checks at N=65536. The CUDA adapter compiles for sm_89/sm_120 but has not run
on a GPU. This is bounded [B06 preparation](implementation-backlog.md#b06-resident-runtime-and-registry),
not a production backend or a measured speedup; D11/P2 remains open.

## What is implemented

| Part | Concrete implementation |
|---|---|
| Public prototype | [`resident.hpp`](../include/ntt/resident.hpp): sessions, immutable polynomial handles, completions and owned host results |
| Runtime and registry | [`resident.cpp`](../src/resident.cpp): private allocation provenance, dependencies, memory reservations, operation journal and cleanup |
| Transport boundary | [`resident_driver.hpp`](../include/ntt/resident_driver.hpp): device buffers/events and a single-stream command interface |
| CUDA adapter | [`resident_adapter.inc`](../src/cuda/resident_adapter.inc), included by [`ntt_kernels.cu`](../src/cuda/ntt_kernels.cu), reuses the existing kernels |
| Deferred host transport | [`resident_host_driver.hpp`](../tests/support/resident_host_driver.hpp): queues CPU work for ownership/failure tests; never represents GPU execution |

Each session fixes its N, exact parameter snapshot and ordered basis. Uploads
may declare coefficients in natural order or evaluations in the existing
bit-reversed order. Data use canonical `uint64_t` residues. Whole-tower subviews
retain the original allocation and readiness; transforms issue fresh outputs.

Use one host thread per session, including handle access and destruction.
The API is separate from the abstract [proposed Backend](../include/ntt/backend_api.hpp).
It supplies no production OpenFHE adapter, cache, CPU fallback, cross-device
operation or multi-stream execution.

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

`values` borrows immutable storage owned by `output`; keep that object alive
while reading the span. `wait_values()` waits for completion and checks canonical
residues before exposing storage. Exact agreement with OpenFHE is a separate test.
`session.multiply(a, b)` composes the same chain and validates both external
arrays before the first command is submitted.

For an independent inverse input, `upload_evaluation(values)` explicitly
declares a bit-reversed evaluation array. It does not transform or reorder it.
Every input must have the session's complete ordered basis and canonical values.
`subview()` can subsequently select contiguous whole towers. It cannot fabricate
reduced QlP by shortening a full QP array across inactive Q towers.

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

Every queued operation pins its inputs, output, staging and dependencies before
calling the transport. Dropping a public handle or completion therefore cannot
free an allocation still used by queued work. Successful completion retires the
completed stream prefix. A failed submission remains in the bounded journal
even if no completion handle could be returned.

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

The journal keeps compact completion tombstones after releasing pins. Its
operation limit does not reset when work completes; start a new session for
another bounded batch. These are prototype session limits, not a device-wide
cache or a replacement for [B07 admission/cache work](implementation-backlog.md#b07-cache-and-memory-admission).

## Observed checks

- **293 host control checks** cover independent negacyclic convolution,
  immutable inputs, owned uploads, explicit/implicit dependencies, nested views,
  foreign/closed handles, budgets, every synchronous transport step, failures
  after enqueue, asynchronous errors, abandoned outputs and quarantine recovery.
- The same control tests pass **AddressSanitizer, UndefinedBehaviorSanitizer
  and leak detection** on Linux.
- The deferred host transport matches pinned OpenFHE on **4,456,448 residues**
  at N=65536 with an unsorted 55/50/60-bit basis. Cases cover zero, bounds,
  monomials and dense values, forward, independent inverse, roundtrip, product,
  input immutability and tower views.
- Stock CUDA 13 compiles the adapter and the prepared native test for
  **sm_89 and sm_120**. Five invalid CUDA-linked factory inputs reject without
  requiring a driver. These checks execute no device arithmetic.

The [exact fixture](../tests/resident_openfhe_tests.cpp) uses real OpenFHE
`NativePoly` format conversion and multiplication. Its transform path reaches
[`NativePoly::SwitchFormat`](https://github.com/openfheorg/openfhe-development/blob/b59e1f291965f14fbee5172cd83a20f5b856204f/src/core/include/lattice/hal/default/poly-impl.h#L454-L474)
through the format API. Root/modulus selection comes from the pinned OpenFHE
implementation; complete arrays are compared, not approximate CKKS decryptions.
The CUDA variant uses the same fixture, but is **unrun** here.

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

The next paid session must run the current native exactness and failure gates,
the new resident CUDA test, sanitizer checks and profiling on an actual RTX 5090.
Keep [D11/P2](openfhe-integration-plan.md) open until those required observations
exist. A passing host transport or compiler-only build cannot supply them.
