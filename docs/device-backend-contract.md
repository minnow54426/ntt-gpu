# Device buffers and the backend boundary

**D9 · API proposal with executable metadata checks · N=65536**

A polynomial on the GPU needs more than a pointer and a length. Its ordered
primes, roots, active level, representation, producing operation, and lifetime
must travel together. D9 defines that boundary and proposes an opt-in OpenFHE
format-conversion adapter as the first integration seam.

Read [D7's representation contract](openfhe-representation-contract.md) for the
array convention and [D8's Hybrid graph](openfhe-hybrid-operation-graph.md) for
Q/P transitions. The [D5 buffer map](cuda-buffer-dataflow.md) describes the
existing synchronous implementation.

## 1. What D9 delivers

| Artifact | State after D9 |
|---|---|
| Immutable parameter snapshot, layouts, views, cache identities and dependency checks | Implemented as a portable host metadata prototype |
| Backend, immutable polynomial, stream and completion interfaces | Compile-checked abstract proposal; no runtime factory |
| OpenFHE metadata bridge | Checked against actual D8 Hybrid outputs and evaluation-key parameters |
| CUDA allocations, streams, events, asynchronous copies and cache eviction | Proposed lifecycle; not implemented by these headers |
| OpenFHE adapter and device dispatch | Proposed first seam; implementation follows acceptance and native GPU gates |

The prototype lives in [backend_contract.hpp](../include/ntt/backend_contract.hpp),
with interfaces in [backend_api.hpp](../include/ntt/backend_api.hpp) and tests in
[backend_contract_tests.cpp](../tests/backend_contract_tests.cpp). The
`ntt::proposal` namespace makes its status explicit. These headers introduce
no CUDA or OpenFHE dependency into the standalone build. The current CUDA
wrapper and its algorithms are unchanged.

**D9 completes the proposal. D10 must accept the architecture and backlog.**
P1 remains in progress; a metadata check cannot prove a physical allocation,
an event wait, or native GPU correctness.

## 2. Carry the complete polynomial identity

The immutable `Parameters` object owns a deep copy of:

- ring dimension N;
- full, ordered Q and P lists, each containing the exact modulus and primitive
  2N-th root;
- full-context digit width α, which must not be recomputed after dropping Q towers.

Its factory rejects invalid dimensions, composite or duplicate moduli, invalid
roots, and invalid digit geometry. For these power-of-two rings, the root test
is ψᴺ = −1 mod q; together with prime q and q ≡ 1 mod 2N, it establishes order
2N. The semantic bound is q < 2⁶³. The initial device shape is narrower:
N=65536 and canonical 50–60-bit primes. A valid shape still needs an available
backend and an operation verified on the selected GPU.

Alternative primitive roots can be mathematically valid metadata. The first
OpenFHE adapter must import the exact existing roots and respect D7's shared
FTT-cache constraint; support for conflicting custom-root contexts is not implied.

Each `PolynomialLayout` adds the active Q count L, an explicit ordered list of
`TowerRef {Q or P, index}`, domain, order, and residue encoding. M is the full
Q count; K is the P count. The full parameter snapshot remains attached at
reduced levels.

| Field | Initial supported contract |
|---|---|
| Coefficient domain | Natural coefficient order |
| Evaluation domain | OpenFHE-compatible bit-reversed slot order |
| Residues | Canonical `uint64_t`, 0 ≤ x < q for each tower |
| Packing | Tower-major; each selected tower has exactly N consecutive residues |
| Device identity | Device ordinal plus backend-session identity |
| Allocation identity | Registry ID plus generation; never an OpenFHE object address |

Montgomery and lazy residues are explicit enum values but rejected by this
prototype. They cannot enter the canonical kernels under a different label.
The host-value validator checks the exact element count and every residue.
An adapter must verify the outer DCRT format and each tower's format, dimension,
modulus and root before constructing this layout; relabeling data is not conversion.

Forward requires coefficient input and returns evaluation layout. Inverse
requires evaluation input and returns coefficient layout, including the existing
N-inverse normalization. Pointwise multiplication requires identical evaluation
layouts. Parameter equality compares complete values, not hash equality alone.

## 3. Make views explicit and bounded

A future private registry owns allocation records. A public view carries an
allocation ID/generation, first tower, tower count, and active Q count. It exposes
no raw device pointer. Validation checks a live matching generation, device and
session, allocation capacity, legal whole-tower extent, and layout consistency.
Byte counts use checked multiplication.

\[
\operatorname{offset}(t,i)=8(tN+i),\qquad
\operatorname{bytes}(T)=8NT.
\]

At N=65536, one tower is 524,288 bytes, or **0.5 MiB**. Whole-tower subviews are
immutable aliases that retain the allocation and its readiness completion.
Arbitrary coefficient strides and partial-tower views are deferred.

A reduced QP basis is a particularly important boundary:

| Basis | Logical order | Indices in the full five-Q/three-P key |
|---|---|---|
| Full QP, L=5 | q₀ q₁ q₂ q₃ q₄ p₀ p₁ p₂ | 0 1 2 3 4 5 6 7 |
| Active QP, L=4 | q₀ q₁ q₂ q₃ p₀ p₁ p₂ | 0 1 2 3 5 6 7 |

Shortening the full allocation to seven towers produces q₀…q₄,p₀,p₁, which is
wrong. The validator rejects that attempted L=4 view. Materialize active QP in
a fresh contiguous allocation using the explicit tower map. A Q prefix alone
can use a zero-copy prefix view; the full key's P suffix can also be viewed
without copying. For initial Hybrid execution, gathering the active key into
a packed immutable snapshot is the simple proposed option; strided key access
requires a separately accepted kernel contract.

The proposal returns fresh output allocations for forward, inverse and
pointwise operations. Inputs are immutable, and output aliasing is rejected.
Existing kernels may still mutate a private output or private scratch buffer.
An initial implementation may therefore need an extra device-to-device copy;
that cost must appear in benchmarks. Public in-place operations wait for a
separate alias and concurrency design.

## 4. Submit work with an explicit completion

`Backend` proposes `create_stream`, `upload`, `forward`, `inverse`, `pointwise`,
`subview`, `download`, and `close`. `Submit` carries an owned stream lease and
owned dependency handles. `PendingPolynomial` couples an immutable result
handle with its completion. This is the intended usage, not a runnable backend:

```cpp
Submit queue{backend.create_stream(), {}};
auto resident = backend.upload(coefficient_layout, host_values, queue);
auto transformed = backend.forward(resident.value, queue);
auto copy = backend.download(transformed.value, queue);
auto evaluation_values = copy->wait_values();
```

The upload must validate and copy the borrowed host span into owned staging
before returning. Download owns its host destination and exposes values only
after successful completion. The same stream orders this example; a consumer
on another stream must explicitly depend on the producer's completion.

| Producer state | Consumer rule |
|---|---|
| Pending, same stream | FIFO stream order establishes the dependency |
| Pending, different stream | Queue a wait for the exact producer event and generation |
| Succeeded | Consumption is allowed after actual event success is established |
| Failed | Reject the consumer and propagate the producer error |
| Different device or backend session | Reject; peer transfer is not part of this API |

Every submitted operation receives a monotonically increasing ID in its backend
session. Only earlier submitted completions can be dependencies; callers cannot
manufacture future events or introduce cyclic waits. Each completion owns one
immutable event instance. Re-recording an event behind an existing completion
handle is forbidden.

The host prototype checks metadata supplied as trusted registry records. The
runtime must resolve every handle through its private registry, refresh event
status, and issue actual `cudaStreamWaitEvent` calls. Passing the prototype a
fabricated record does not create a CUDA event or prove synchronization.

## 5. Keep resources alive through completion

Each submitted operation pins its input allocations, output allocation,
stream, plans, keys, host staging, and scratch leases. The backend retains these
pins even when the caller drops the returned completion. Dropping a handle is
not cancellation. Private scratch is reusable only after every operation that
can touch it has completed successfully or device quiescence is proven.

The proposed lifecycle is:

1. Validate all metadata, capability and dependency handles before submission.
2. Reserve output/scratch and cache leases under the appropriate locks.
3. Submit work and record a completion event; retain all pins in an operation record.
4. On success, publish readiness and release operation pins.
5. On failure, mark the result failed and quarantine affected resources until
   the stream/device is drained or torn down safely.

**A failed event is not evidence that the device has stopped using memory.**
The `can_retire` prototype permits retirement only with zero external leases,
zero in-flight pins, and either successful readiness or proven quiescence.
Runtime pin counts are registry facts, not guesses from `shared_ptr::use_count`.
A budget exhausted by pinned entries produces backpressure or a reported
allocation failure; it never permits eviction of a live entry.

The backend state should use `unique_ptr<Impl>`, move-only allocation owners,
and a device guard on every runtime call. Constructors must unwind partial
allocation safely. `close()` stops submission, drains work, reports errors and
invalidates further use; destructors are nonthrowing. Outstanding leases retain
backend state until safe release. An unrecoverable context failure needs a
quarantine/teardown policy rather than immediate pool reuse.

D5 identified a raw-`Impl*` constructor-failure leak in the existing wrapper.
This proposal specifies the replacement discipline; **it does not fix or test
that CUDA failure path**. Fault injection and real runtime teardown tests belong
in the accepted implementation backlog.

## 6. Cache semantic identities, then pin leases

Caches belong to one backend session and device. A content hash can accelerate
lookup, but a hit must compare the complete semantic key. Object pointers,
ring size alone, or OpenFHE key tags alone are insufficient.

| Cache | Required identity |
|---|---|
| NTT plans/tables | Device/session, N, exact ordered q/ψ pairs, encoding, table convention, arithmetic revision |
| Digit conversion plans | Full parameter context including α, L, digit index, ordered source/complement bases, conversion-rule revision |
| Evaluation-key material | Device/session, full context, immutable material ID/generation, key tag, key kind and index, packing revision |

The NTT plan bundles forward/inverse powers and normalization constants. Its
current table convention is natural ψ powers including the N endpoint. It
cannot reuse an OpenFHE bit-reversed FTT table as if the two layouts were equal.
A forward/inverse plan may share identity because both directions are included;
the operand's domain still controls which operation is valid.

For digit d, the conversion-plan validator derives its contiguous Q partition
using the full α, then verifies the exact ordered active-Q/P complement. Its
rule revision includes D8's uncentered, **unreduced approximate CRT sum**. A
canonical-CRT plan is a different operation. ModDown needs a distinct plan kind
for its P→active-Q conversion and P-inverse scale; the prototype's
`ConversionPlanKey` covers digit extension only.

Evaluation keys are deep immutable A/B snapshots; secret keys are not uploaded.
A material ID/generation is assigned to that snapshot. Mutating the original
OpenFHE key requires a new snapshot and identity. A rotation index or
relinearization component is part of the identity; equal tags do not establish
equal key data. Packing validates digit count, full QP order and every residue.

On a cache miss, a future implementation publishes one pending entry under a
mutex and shares its readiness event with concurrent requests. This single
upload must be waited on across streams. Readers pin the entry through their
own completion. A budgeted LRU can evict only unpinned, quiescent entries.
The proposal contains equality/validation rules and retirement checks, not an
implemented cache, uploader, hash table or LRU policy.

The abstract API exposes `cache_stats()`, `set_cache_budget(bytes)`, and
`trim_cache()`. Statistics count allocated cache bytes including pending uploads,
with pinned bytes counted once per entry; operand/scratch pools are separate.
Budget is reserved before allocation. Lowering the budget evicts eligible entries
first and fails if live pins prevent the requested bound. Trimming returns only
bytes actually freed. D10 must select the initial budget and admission policy.

Hoisted digits are explicit immutable results, not an implicit global cache.
Their provenance includes the complete parameter snapshot, source allocation
ID/generation, complete active-Q view, domain/order and rule revision. The
prototype rejects stale or incompatible provenance. Reuse across rotation keys
is permitted only when that input/context is unchanged and each key matches
the same Q/P/α context; the digits do not depend on a particular key's A/B data.
A private registry must also validate that the referenced source exists and
has the claimed layout.

## 7. Budget the Hybrid lifetimes

D8 fixes the graph; D9 assigns the ownership boundaries. Digit coefficient
scratch lives through its complement conversion. Each assembled evaluation
digit lives through every consuming B/A MAC. QP accumulators live through
ModDown or an explicitly delayed QP consumer. P coefficient scratch and Q
correction buffers live through the final correction transforms and subtraction.
Keys and tables remain pinned by all readers.

For the full D8 fixture M=L=5, K=3, α=3, β=2 at N=65536:

| Storage | Logical size | Lifetime |
|---|---:|---|
| Input Q polynomial | 2.5 MiB | Upload through last decomposition reader |
| Two assembled QP digits | 8 MiB | Hoist result through last MAC reader |
| Two QP accumulators | 8 MiB | MAC through last ModDown/delayed reader |
| Two final Q outputs | 5 MiB | Returned immutable outputs |
| Full two-digit A/B key | 16 MiB | Cached snapshot plus in-flight reader pins |

The sizes follow 8NT bytes per T-tower polynomial. They are logical payloads,
not a measured or proven peak allocation bound. Streaming digits can release
scratch earlier; preserving a hoist for several rotations keeps all digits
alive. Allocator alignment, twiddle tables, host staging, concurrent streams,
conversion scratch and out-of-place copies add memory. D10 must choose the
initial budget/backpressure policy; P2/P3 must measure actual live bytes.

## 8. Integrate OpenFHE through a transactional adapter

The proposed first seam is an **explicit opt-in DCRT format-conversion adapter**
with SetFormat-equivalent behavior. It does not replace OpenFHE's global
`SwitchFormat` or patch every NativePoly call. This small boundary tests exact
tower packing, transform ordering, ownership and failure handling before Hybrid
GPU integration adds conversion and key-material lifetimes.

The host operation is synchronous even if its internal backend is asynchronous:

1. Snapshot the DCRT context and validate all tower metadata, labels and values.
   If the requested format already matches, return without conversion.
2. During preflight, select the device route or an explicitly permitted CPU route.
3. Pack each tower using `ConvertToInt()` into canonical tower-major storage.
   Do not reinterpret OpenFHE native integer objects as a flat `uint64_t*`.
4. Upload, transform, download and wait. Import into a private candidate object,
   set each tower's actual format and the DCRT format, and verify metadata.
5. Replace the caller's object only after the entire candidate succeeds.

The caller must serialize access to the same host object during the operation.
The candidate preserves strong exception safety: a failed upload, launch,
completion, download or import leaves the original host object unchanged.
This is a proposed guarantee; it requires adapter and fault-injection tests.

| Situation | `RequireDevice` | `AllowCpuBeforeSubmit` |
|---|---|---|
| Invalid metadata, residues or format labels | Error | Error |
| Already in the requested format, valid input | No-op | No-op |
| Available, supported and verified device operation | Device | Device |
| Device unavailable, valid unsupported shape, or unverified operation | Error | CPU before submission |
| Any failure after submission | Propagate error | Propagate error |

CPU fallback uses a private host candidate too. No automatic CPU retry occurs
after device submission. A caller may explicitly retry the unchanged original
after handling the backend error. An unverified CUDA operation cannot advertise
production support merely because a kernel compiles; a diagnostic harness can
run the native exactness gates to establish that capability.

## 9. Verification and D10 acceptance

`backend_contract_check` runs in the normal CPU build. It checks valid and
invalid roots/parameters, immutable snapshots, domain transitions, canonical
host values, byte overflow, reduced-QP mistakes, view ranges, stale generations,
aliasing, stream dependencies, failed-resource retirement, cache identities,
hoist provenance, and preflight-only fallback.

The opt-in `openfhe_hybrid_graph` additionally imports real pinned OpenFHE
parameters at L=5,4,3,1 and checks the proposed active QP layout and key indices
against actual Hybrid outputs and full-context key towers. It retains D8's
18 cases and exact residue-array comparisons.

```sh
.venv/bin/cmake --build build/openfhe-compat/ntt-build -j2
.venv/bin/ctest --test-dir build/openfhe-compat/ntt-build --output-on-failure -j2
```

D10's acceptance artifact should decide and record:

- the immutable, out-of-place initial API and its device-to-device copy cost;
- the DCRT format adapter as the first seam and its preflight-only fallback;
- private-registry provenance, resource pins and error quarantine;
- cache upload coordination, memory budget and backpressure;
- the ordered implementation backlog: RAII/failure cleanup, native exactness,
  resident NTT backend, adapter tests, then conversion/Hybrid integration.

Required runtime tests include partial-constructor failure, wrong device,
concurrent cache misses, missing event waits, completion abandonment, cache
pressure with live readers, asynchronous failure and drain, and host rollback.
None is replaced by metadata simulation. Real CUDA execution, complete Hybrid
GPU support, CKKS application integration and performance remain later gates.
