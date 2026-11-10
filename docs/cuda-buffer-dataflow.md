# CUDA buffers and host/device dataflow — D5

This is a map of the current implementation, not a proposed device-resident
API. Read it alongside the [public interface](../include/ntt/cuda_ntt.hpp),
[implementation and launchers](../src/cuda/ntt_kernels.cu), and
[kernel schedule](ntt-4096-work-distribution.html).

## The big picture

One `CudaNttTransform` represents one fixed ring size and a fixed list of RNS
moduli/roots. It owns two reusable operand buffers and four constant buffers.
The public interface accepts **host vectors**, not device pointers.

```text
construct once
  validate parameters → build host tables → allocate six device buffers
  → upload constants once

each public operation
  host inputs → upload to reusable scratch → kernels → download result

destroy after successful construction
  delete Impl → free six device buffers
```

Constants and scratch allocations persist across calls. Operand contents are
scratch state and are overwritten by the next call. Only the internal chain
inside `multiply()` remains device-resident between its arithmetic operations.
Calling public `forward()`, `pointwise_multiply()`, and `inverse()` separately
does **not** preserve a transfer-free pipeline.

## Constructor: what happens, in order

1. Allocate `Impl` with `new Impl{}`; all six CUDA pointers start as null.
2. Reject empty moduli or a moduli/root count mismatch. Store N and limb count K.
3. Set tile length `min(256,N)` and column length `N/256`.
4. For every limb, construct a `NegacyclicNtt` CPU reference and build powers
   of ψ and ψ-inverse for exponents **0 through N inclusive**. Compute N-inverse.
5. Allocate the six persistent CUDA buffers listed below. Host allocations
   have already occurred; parameter checks precede **device** allocation.
6. Upload both power tables, the moduli, and inverse lengths: four H2D copies.
7. Record row-grid count `max(1,N/256)` and elementwise-grid count
   `ceil(K*N/256)`. All kernels launch 256 threads per block.

The CPU constructor checks power-of-two N≥2, `(q−1) % (2N) == 0`,
`ψ^N mod q == q−1`, and q>N. It assumes prime q; it does not itself run a
primality test. The parameter generator is a separate source of checked
primes. See [parameter generation](../include/ntt/param_gen.hpp).

After construction, temporary host power-table/inverse-length vectors are
released. `Impl` retains the moduli vector and per-limb CPU references, not a
second permanent host copy of every power table.

## Persistent allocation and ownership map

Define S=`8*K*N` bytes, T=`8*K*(N+1)` bytes, R=`8*K` bytes.
Every stored element is `uint64_t`.

| Device allocation | Elements / bytes | Layout and role | Writer / reader |
|---|---|---|---|
| `device_operand_a` | K*N / S | `[limb][coefficient or evaluation position]`; working input/output | Public input copy and kernels write; result copy reads |
| `device_operand_b` | K*N / S | Second operand; separate from A | Input copy and its forward transform write; pointwise kernel reads |
| `device_twiddle_powers` | K*(N+1) / T | `[limb][exponent]`, ψ^e | Constructor uploads; forward kernels read |
| `device_inverse_twiddle_powers` | K*(N+1) / T | `[limb][exponent]`, (ψ-inverse)^e | Constructor uploads; inverse kernels read |
| `device_moduli` | K / R | q for each limb | Constructor uploads; arithmetic kernels read |
| `device_inverse_transform_lengths` | K / R | N-inverse modulo each q | Constructor uploads; final scale kernel reads |

`CudaNttTransform` owns `Impl*`; `Impl` owns the CUDA pointers. They are private:
callers cannot retain a borrowed device-buffer view. Copy construction and
copy assignment are deleted. No move operations are declared; this class is
not currently movable either. Do not use one instance concurrently: public
operations share and overwrite A/B and there is no locking or per-call scratch.

On normal destruction, `~CudaNttTransform()` deletes `Impl`, whose destructor
calls `cudaFree` on each pointer. Free errors are deliberately ignored. Do not
infer safe concurrent destruction or synchronization with unrelated GPU work.

### Exact byte counts (requested allocations, not measured device usage)

```text
two operand buffers = 2S            = 16KN bytes
constant buffers    = 2T + 2R       = 16KN + 32K bytes
total               = 2S + 2T + 2R = 32K(N+1) bytes
```

For **three limbs**, the source formulas give:

| N | One operand S | Both operands | Constants / constructor upload | All six allocations |
|---|---:|---:|---:|---:|
| 4096 | 98,304 B (96 KiB) | 196,608 B | 196,704 B | 393,312 B |
| 65536 | 1,572,864 B (1.5 MiB) | 3,145,728 B | 3,145,824 B | 6,291,552 B |

The extra 96 bytes comprise 48 bytes for the two power tables' final entries
and 48 bytes for the moduli and inverse lengths.
CUDA context/allocator overhead, registers, per-block shared memory, host
vectors, and CPU-reference objects are **not** included. These are formulas,
not a VRAM measurement or a memory-capacity benchmark.

## Public API contracts

For operands, flat index `k*N+i` belongs to limb k. Power-table index
`k*(N+1)+e` belongs to exponent e of that limb. Transform kernels select limbs
with `blockIdx.y`. Pointwise and scale kernels instead use a flat grid over
K*N entries and recover the limb by integer division by N.

| Public call | Input → output representation | Host-visible effect | Copies / launches at N≥512 |
|---|---|---|---|
| `forward(values)` | Natural coefficients → bit-reversed evaluations | Mutates host vector | H2D A, column + row, D2H A: 2 copies, 2 launches |
| `inverse(values)` | Bit-reversed evaluations → natural coefficients | Mutates host vector | H2D A, row + column + scale, D2H A: 2 copies, 3 launches |
| `pointwise_multiply(left,right)` | Matching bit-reversed evaluations → same order | Replaces left; right is a const input | H2D A+B, pointwise, D2H A: 3 copies, 1 launch |
| `multiply(left,right)` | Natural coefficients → natural negacyclic product | Left is passed by value; returns its result buffer | H2D A+B, 2 forwards + pointwise + inverse, D2H A: 3 copies, 8 launches |

Inputs must contain K*N residues in their corresponding `[0,q[k])` ranges.
Methods check lengths, but do not scan residue ranges or store a runtime
coefficient/evaluation tag. The caller is responsible for representation and
parameter compatibility. A vector from another modulus set with the same
length is not rejected merely for having different provenance.

`n()` and `limbs()` return scalar geometry without a device copy. `q(k)` and
`ref(k)` use bounds-checked host vectors; `ref(k)` is a borrowed const reference
valid only while the transform object exists. These getters do not expose the
GPU allocation or transfer data.

## Follow a complete multiplication

```text
HOST left (by-value local) ──H2D──> A: coefficients
HOST right (const input) ──H2D──> B: coefficients

A ──column DIF──row DIF──> A: bit-reversed evaluations
B ──column DIF──row DIF──> B: bit-reversed evaluations
A,B ──pointwise (write A only)──> A: product evaluations
A ──row DIT──column DIT──N-inverse scale──> A: product coefficients

A ──D2H──> returned HOST left
```

The four constant buffers are read throughout this chain; they are not
re-uploaded per operation. B is scratch after the call, not a cached operand
that the next public operation is entitled to reuse. Every new `multiply()`
uploads both input operands again.
Passing an lvalue as left creates a host copy; passing a moved vector can
transfer its host allocation into the by-value parameter instead.

Each public method uses `cudaMemcpy`, not `cudaMemcpyAsync`, and returns the
downloaded host result. Launches specify no explicit stream; there is no public
stream/event parameter, asynchronous completion handle, or device-resident
buffer API. `cudaGetLastError()` follows launches; copies are also checked by
`CHECK_CUDA`. There is no automatic CPU fallback on a CUDA error.

## Transfer boundary: fused versus separate calls

Excluding construction, each forward/inverse transfers 2S bytes and pointwise
or fused multiply transfers 3S bytes. For a fair comparison of **two inputs**:

```text
forward(a) + forward(b) + pointwise_multiply(a,b) + inverse(a)
    2S     +    2S     +          3S             +   2S = 9S

multiply(a,b) = 3S
```

Both paths perform eight kernels at the target sizes. Fusion removes six
operand-sized transfer copies (9S→3S), not six kernels. At N=65536/K=3 this is
14,155,776 B versus 4,718,592 B of requested H2D+D2H payload. It is not a
measured latency speedup. Host vector copies, allocator costs and PCIe/cache
effects are separate. At N≤256 forward uses one kernel, inverse two and
multiply five; the host-copy counts are unchanged.

The benchmark wraps public calls in wall-clock timing, so its GPU columns
include operand transfers. Construction/table uploads occur outside that
per-call region; see `run_bench` in the implementation. It is not kernel-only
timing.

## Open ownership and integration issues

- **Constructor exception cleanup:** `impl_` is a raw pointer initialized with
  `new Impl{}`. If the constructor subsequently throws, the enclosing object's
  destructor is not run and nothing in the constructor deletes that pointer.
  Invalid parameters can leak host state; allocation/upload failure can also
  leak previously allocated CUDA buffers. This is a source-inspected cleanup
  gap, not a fault-injection result. Introduce RAII in a dedicated implementation
  change and verify allocation/upload failure paths before production use.
- No device identifier is captured in the object and no device guard is used
  around operations/destruction. Multi-device ownership is not specified.
- No stream/event lifetime contract, buffer view, representation/level tag,
  evaluation-key cache, or OpenFHE Q/P base-switch integration exists in this
  interface. Reusable allocations are not the final integration API.
- Native CUDA compilation and launch/barrier behavior are still unverified
  here. CPU schedule tests do not compile or run this CUDA wrapper.

D9 should freeze these ownership/API decisions; the production device API and
RAII implementation must follow explicit failure/lifetime tests. D6 next
examines arithmetic bounds, parameter assumptions, and adversarial exactness.

## Verification and limits

[Source-contract tests](../tests/docs/test_cuda_dataflow.py) check that this map
matches current allocation sizes, frees, public copy directions, operation
ordering and host-only interface shape. They also pin the endpoint arithmetic
above. These are deliberately **static source checks**, not CUDA-runtime tests
or proof of constructor failure safety. Existing CTests verify CPU arithmetic
and shared stage helpers. No new claim of actual GPU support/performance is
made by completing D5.
