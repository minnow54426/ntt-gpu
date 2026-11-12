# ADR 0001: accept the device backend architecture

**D10 · Accepted for implementation · 2026-11-12 (Asia/Shanghai)**

Use an explicit, device-resident backend with immutable polynomial handles,
checked representation metadata, and owned completion dependencies. Integrate
OpenFHE first through an opt-in DCRT format-conversion adapter with transactional
host updates and preflight-only CPU fallback.

This decision accepts the D7–D9 contracts and the
[implementation backlog](implementation-backlog.md). **P1 is complete. D11 is
next: provision the native GPU environment.** Acceptance establishes the design;
it does not establish CUDA correctness, runtime failure safety, or performance.

## 1. Review scope and evidence

The reviewed source baseline is
[f349e72](https://github.com/minnow54426/ntt-gpu/tree/f349e72a1053eaebd714232f14415a56c9c17c4b).
OpenFHE remains pinned to 1.6.1, commit
`b59e1f291965f14fbee5172cd83a20f5b856204f`, using NativeInteger 64.
The production target is N=65536 with canonical 50–60-bit RNS primes.
Small teaching examples and schedule boundary tests remain useful evidence;
they do not expand production support.

| P1 question | Accepted evidence | Remaining runtime gate |
|---|---|---|
| Does every butterfly/stage have the correct addresses and factors? | [D1–D4 proofs](ntt-65536-work-distribution.html), exhaustive shared-helper probes and CPU schedule checks | Native compilation, launch geometry, barriers and races |
| Are arithmetic bounds and reductions explicit? | [D6 contract](modular-arithmetic-contract.md), independent bigint checks and full-size CPU patterns | Native 50/55/60-bit adversarial matrix |
| Are OpenFHE formats, roots and tower arrays exact? | [D7 contract](openfhe-representation-contract.md) and real NativePoly/DCRTPoly checks | Same fixtures through CUDA and the adapter |
| Is the Hybrid graph exact? | [D8 graph](openfhe-hybrid-operation-graph.md), 18 full-size cases and intermediate-array comparisons | Broader parameter/key coverage and GPU composition |
| Are buffer identity and ownership explicit? | [D9 contract](device-backend-contract.md), 78 metadata checks and a real OpenFHE metadata bridge | Private registry, actual events, allocation and failure tests |

The existing [CUDA wrapper](cuda-buffer-dataflow.md) remains a synchronous
correctness baseline. The proposed interfaces in
[backend_api.hpp](../include/ntt/backend_api.hpp) have no runtime implementation
or factory. `ntt::proposal` stays in place until implementation tests justify
promotion. This ADR freezes behavioral invariants, not a stable public C++ ABI.

## 2. Freeze the representation and operation boundary

Accept these rules without reinterpretation:

- A deep immutable context carries N, the full ordered Q/P moduli and exact
  primitive roots, and the full-context digit width α.
- Each polynomial carries its ordered tower references, active Q count L,
  domain, order and encoding. Coefficients use natural order; evaluations use
  the D7 bit-reversed order. Residues are canonical `uint64_t` values.
- Forward, inverse and pointwise operations validate their input layouts.
  Inverse includes N-inverse normalization. Public inputs are immutable;
  results have fresh storage. Private scratch/output may use in-place kernels.
- Views alias only complete contiguous towers and retain allocation/readiness
  leases. A shortened full-QP span cannot stand in for reduced QlP. Reduced-level
  P key indices retain their full-context offsets.
- D8's digit/complement transform placement, unreduced approximate CRT,
  B-before-A output order, accumulation before ModDown, and delayed-QP scaling
  remain exact requirements. A decryption result alone cannot replace them.

Semantic validity and device capability remain separate. Mathematical validity
of an alternative primitive root does not establish OpenFHE cache compatibility.
The first adapter accepts only the pinned context's canonical minimum roots and
rejects known conflicting registrations before using the shared FTT cache.
The process must not populate that upstream cache with conflicting custom roots;
its entries do not retain enough root identity to detect such external usage
after the fact. Broader custom-root support requires a new reviewed contract
and fixtures.

Out-of-place results may require an additional device-to-device copy when using
the current kernels. Record that copy in memory and timing results. Public
in-place writes, partial-tower strides, peer transfers and cross-device handles
are excluded from the initial API.

## 3. Accept the first OpenFHE seam and fallback policy

The first adapter is an explicit opt-in DCRT operation with SetFormat-equivalent
behavior. It packs towers through `ConvertToInt()`, performs a backend transform,
imports a private candidate, and commits the candidate only after complete
success. It does not globally replace `SwitchFormat` or patch NativePoly storage.
The first adapter transaction is synchronous to its caller.

Validate the DCRT label, each tower's label, dimension, ordered q/ψ pairs,
length and canonical values, including on a same-format request. Verify both
tower labels and the outer DCRT label when importing; metadata overrides alone
do not convert data. Access to the same host object must be serialized by its
caller.

| Condition | Accepted behavior |
|---|---|
| Valid object already in the requested format | Return without conversion |
| Available, supported, verified device operation | Use the device route |
| Valid but unsupported layout, unavailable GPU, or unverified operation | CPU only under `AllowCpuBeforeSubmit`; otherwise error |
| Invalid object/metadata/residues | Error under either policy |
| Admission busy, budget exhausted or allocation failure | Report the resource error; do not silently change routes |
| First enqueue-capable call attempted, then any failure | Propagate failure; no automatic CPU retry |

`RequireDevice` is the explicit default for a caller opting into this adapter.
Applications choosing fallback must select `AllowCpuBeforeSubmit`. The existing
OpenFHE CPU APIs keep their existing behavior.

Commit the completed candidate through the pinned DCRT move assignment, which
is declared `noexcept` in [the reviewed source][dcrt-move]. All validation and
import work precedes that commit. Recoverable failures leave the original host
object unchanged. This promise does not cover process termination: some upstream
constructors/copy operations are declared `noexcept` while allocating. Adapter
fault tests must distinguish recoverable injected failures from upstream fatal
allocation paths; no universal out-of-memory rollback claim is accepted.

## 4. Close the partial-submission lifetime gap

D9 establishes leases through completion. D10 additionally requires those leases
to exist **before the first enqueue-capable call**, including when no completion
handle is ever returned.

| Stage | Required ownership and error behavior |
|---|---|
| Preflight | Validate handles, choose the route, reserve checked resource credits and create an operation record |
| Submitting | Mark the transaction as submitted before attempting the first async copy/kernel/wait; the operation record already owns all leases |
| Event recorded | Publish an immutable completion tied to that event/generation; retain all input/output/plan/staging/scratch pins |
| Submission or event-recording failure | Keep the record privately, mark it failed, drain the owning stream; do not release memory merely because no handle was returned |
| Completed successfully | Make readiness visible and release operation pins; external handles retain their own leases |
| Failed, quiescence not established | Poison the backend session, reject further submission and quarantine affected resources until safe teardown |

A producer on the same stream uses FIFO order. A pending producer on another
stream needs the exact event wait. Only earlier submitted operation IDs are
valid dependencies; IDs/generations must never wrap into reusable identities.
A failed producer fails its consumers. Event failure is not proof of device
quiescence, and dropping a completion is not cancellation.

The registry validates all allocation, stream and event identities against its
own session; user-supplied structs are not authority. Constructor unwinding uses
RAII, each runtime call uses the owning device guard, and `close()` drains and
reports errors. Destructors do not throw. The [D5 raw-Impl cleanup gap](cuda-buffer-dataflow.md#open-ownership-and-integration-issues)
is a release blocker tracked as B02, not a fix delivered by this ADR.
Subsequent [offline preparation](offline-gpu-preparation.md) repairs baseline
ownership in source and compiles it; B02's native fault verification stays open.

Do not automatically call `cudaDeviceReset` to recover a session in a shared
process: it can invalidate allocations owned by other users of the device.
An unrecoverable session reports failure and retains quarantine until safe
owner-controlled teardown, potentially at the worker-process boundary.

## 5. Choose initial cache and admission limits

Use these conservative, configurable defaults for the first backend/adapter
implementation. They are engineering limits, not measured optimal settings.
MiB means 2²⁰ bytes.

| Setting | Initial decision |
|---|---|
| Device cache budget | 256 MiB per backend session |
| Device operand/output/scratch budget | Separate 256 MiB per backend session |
| Owned host staging budget | Separate 128 MiB per backend session |
| Initial executor | One explicit nonblocking CUDA stream per backend session |
| Initial adapter concurrency | One host adapter transaction at a time per session |
| Admission when limits cannot be met | Fail immediately with a specific error; no hidden waiting or automatic budget growth |

These are ceilings, not eager allocations. A second initial stream or concurrent
adapter transaction reports a resource-limit/busy error. The accepted dependency
model supports later multi-stream execution, but that capability remains disabled
until its race/failure tests pass. Pure metadata helpers remain portable.

Charge the allocator's requested allocation extents, including explicit padding,
before allocating. These budgets bound accounted allocations, not CUDA's internal
physical-memory reservation or driver overhead; measure those separately.
Pending cache uploads count against the cache budget; an entry's pinned bytes
are counted once. Operand/output/scratch allocations remain charged while any
operation or external polynomial handle owns them. Host staging includes owned
upload snapshots and retained download results. Register table-build/host-object
memory separately in measurements; it is not covered by the staging cap.

The initial allocator uses individually owned allocations, so `trim_cache()`
reports actual released device bytes. A later pooling allocator must distinguish
reusable suballocations from memory returned to CUDA and revise accounting before
claiming the same statistic.

Before the adapter's first submission, reserve credits for its complete checked
transaction plan. Individual asynchronous backend operations reserve their own
requirements before enqueue; a later operation can still fail after an earlier
operation has submitted, which forbids automatic fallback for that transaction.
Preserve the original host object in either case.

Cache lookup uses D9's complete semantic keys. One mutex-protected pending entry
coordinates each upload. Evict least-recently-used eligible entries only after
external leases and operation pins are zero and quiescence is proven. If eviction
cannot admit a request, return `CacheBudgetExceeded`; if output/staging credits
are unavailable, report the corresponding resource limit. Budget reduction is
atomic: if the new limit cannot be met, retain the previous limit.

The D8 full-context key payload is 16 MiB and one eight-tower forward/inverse
plan has approximately 8 MiB of payload. Those explain why a 256 MiB cache is a
reasonable initial ceiling; they do not bound a production key set. No peak
memory or throughput claim follows from these defaults.

## 6. Alternatives reviewed

| Alternative | Review result |
|---|---|
| Raw pointer plus length | Rejected: omits basis, format, generation and producer ownership |
| Add global format hooks immediately | Deferred: broadens the first seam before packing and rollback are tested |
| Expose public in-place writes first | Deferred: adds aliasing and concurrent-reader hazards before a native baseline |
| Keep the current host-vector wrapper as the production backend | Rejected: forces transfers and hides stream/device ownership |
| Retry CPU after a CUDA failure | Rejected: submission can be partial and dependencies may still be live |
| Unlimited caching or automatic cache growth | Rejected: defeats predictable admission and live-reader safety |
| Replace approximate CRT with canonical CRT | Rejected: changes pinned OpenFHE intermediate arrays |
| Tune layouts/arithmetic before measuring the baseline | Deferred: native correctness and named-hardware evidence come first |

The existing wrapper is retained as the baseline comparison during migration.
No algorithm rewrite or runtime adapter is introduced by D10.

## 7. Accept P1 and track the implementation

| P1 exit requirement | D10 disposition |
|---|---|
| NTT/DCRT and Hybrid representation, ordering and active towers are explicit | Accepted: D7/D8 source and CPU exactness gates |
| Pinned approximate-CRT and reduction order are identified | Accepted: D8 independent intermediate-array checks |
| Buffer/device/stream/event ownership is specified | Accepted: D9 metadata model plus D10 submission-failure rules |
| First integration seam and CPU fallback are frozen | Accepted: explicit DCRT adapter and the policy in section 3 |

The [backlog](implementation-backlog.md) gives each follow-up a phase,
dependency and observable acceptance criterion. Native build/safety and exactness
block benchmark claims. Runtime lifecycle and cache gates block promotion from
`ntt::proposal`. The transactional adapter and application tests block production
OpenFHE integration. D11–D95 retain their phase order; completing D10 does not
mark any later implementation gate complete.

Review validation reruns the 16-test CPU/OpenFHE Release suite, documentation
unit tests, interactive mathematics, route/fragment checks, and desktop/mobile
browser checks. Exact observed results are appended to the
[daily journal](journal/2026-11-12.md#d10-accept-architecture-and-close-p1).
At D10 acceptance, no CUDA compiler or GPU was available in the review environment.
A compiler-only toolkit was installed during later
[offline preparation](offline-gpu-preparation.md); real GPU evidence is still pending.

Any later change to residue encoding, root conventions, public aliasing,
submission/fallback boundaries, ownership, or resource-accounting semantics
requires a superseding ADR and exact regression evidence. Budget values may
be explicitly configured within the accepted admission rules; benchmark-based
default changes must record the hardware and workload that justify them.

[dcrt-move]: https://github.com/openfheorg/openfhe-development/blob/b59e1f291965f14fbee5172cd83a20f5b856204f/src/core/include/lattice/hal/default/dcrtpoly.h#L131
