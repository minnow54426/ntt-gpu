# CUDA buffers and host/device dataflow — D5

This is a map of the current implementation, not a proposed device-resident
API. Read it alongside the [public interface](../include/ntt/cuda_ntt.hpp),
[implementation and launchers](../src/cuda/ntt_kernels.cu), and
[kernel schedule](ntt-65536-work-distribution.html).

Updated after [offline GPU preparation](offline-gpu-preparation.md): ownership,
validation and device guards now compile with nvcc; real GPU behavior is pending.

## The big picture: N=65536, three towers

One `CudaNttTransform` represents one fixed ring size and a fixed list of RNS
moduli/roots. It owns two reusable operand buffers and four constant buffers.
The public interface accepts **host vectors**, not device pointers.

For this page, **N=65536 and K=3**. One polynomial contains 196,608
64-bit residues: **1,572,864 bytes (1.5 MiB)**. Each tower occupies 512 KiB.

<div class="dataflow-grid">
<div><strong>① Construct once</strong><p>Allocate six GPU buffers: 6,291,552 B total.</p><p>Upload four constant arrays: 3,145,824 B, four copies.</p></div>
<div><strong>② Reuse for each call</strong><p>Host inputs → GPU A/B → kernels → host result.</p><p>A and B are each 1.5 MiB. Constants stay on the GPU.</p></div>
<div><strong>③ Destroy</strong><p>Free all six allocations after successful construction.</p><p>CPU vectors and GPU allocations have separate ownership.</p></div>
</div>


Constants and scratch allocations persist across calls. Operand contents are
scratch state and are overwritten by the next call. Only the internal chain
inside `multiply()` remains device-resident between its arithmetic operations.
Calling public `forward()`, `pointwise_multiply()`, and `inverse()` separately
does **not** preserve a transfer-free pipeline.

## Constructor: what happens, in order

1. Create `Impl` with `make_unique`; all six CUDA pointers start as null and the
   owning-device sentinel is −1. Constructor exceptions destroy this partial state.
2. Validate N in [2,65536], matching nonempty tower lists, prime q<2⁶³,
   canonical primitive roots and checked buffer extents. Store N and limb count K.
3. Set tile length `min(256,N)` and column length `N/256`.
4. For every limb, construct a `NegacyclicNtt` CPU reference and build powers
   of ψ and ψ-inverse for exponents **0 through N inclusive**. Compute N-inverse.
5. Capture the current CUDA device, then allocate the six persistent buffers. Host allocations
   have already occurred; parameter checks precede **device** allocation.
6. Upload both power tables, the moduli, and inverse lengths: four H2D copies.
7. Record row-grid count `max(1,N/256)` and elementwise-grid count
   `ceil(K*N/256)`. All kernels launch 256 threads per block.

The CPU constructor checks power-of-two N≥2, `(q−1) % (2N) == 0`,
`ψ^N mod q == q−1`, and q>N. It assumes prime q; it does not itself run a
primality test. The wrapper now adds that prime/range/extent validation in
[cuda_validation.hpp](../include/ntt/cuda_validation.hpp) before any CUDA call.

After construction, temporary host power-table/inverse-length vectors are
released. `Impl` retains the moduli vector and per-limb CPU references, not a
second permanent host copy of every power table.

## Persistent allocation and ownership map

For N=65536/K=3, define S=1,572,864 B, T=1,572,888 B, R=24 B.
In general, \(S=8KN,\ T=8K(N+1),\ R=8K\) bytes.
Every stored element is `uint64_t`.

| Device allocation | Elements / bytes | Layout and role | Writer / reader |
|---|---|---|---|
| `device_operand_a` | 196,608 / 1,572,864 B | `[limb][coefficient or evaluation position]`; working input/output | Public input copy and kernels write; result copy reads |
| `device_operand_b` | 196,608 / 1,572,864 B | Second operand; separate from A | Input copy and its forward transform write; pointwise kernel reads |
| `device_twiddle_powers` | 196,611 / 1,572,888 B | `[limb][exponent]`, \(\psi^e\) | Constructor uploads; forward kernels read |
| `device_inverse_twiddle_powers` | 196,611 / 1,572,888 B | `[limb][exponent]`, \((\psi^{-1})^e\) | Constructor uploads; inverse kernels read |
| `device_moduli` | 3 / 24 B | q for each limb | Constructor uploads; arithmetic kernels read |
| `device_inverse_transform_lengths` | 3 / 24 B | N-inverse modulo each q | Constructor uploads; final scale kernel reads |

`CudaNttTransform` owns `unique_ptr<Impl>`; `Impl` owns the CUDA pointers. They are private:
callers cannot retain a borrowed device-buffer view. Copy construction and
copy assignment are deleted. No move operations are declared; this class is
not currently movable either. Do not use one instance concurrently: public
operations share and overwrite A/B and there is no locking or per-call scratch.

On normal destruction or constructor unwinding, `unique_ptr` destroys `Impl`.
Its destructor selects the owning device, calls `cudaFree` on each pointer and
restores the previous device. Host-preflight failure skips CUDA entirely.
Device-selection/free errors cannot be reported by this nonthrowing destructor;
a failed device selection skips freeing rather than using a different device. Do not
infer safe concurrent destruction or synchronization with unrelated GPU work.

### Exact byte counts (requested allocations, not measured device usage)

\[
\begin{aligned}
\text{Operands}&=2S=3{,}145{,}728\ \mathrm{B},\\
\text{Constants}&=2T+2R=3{,}145{,}824\ \mathrm{B},\\
\text{All six}&=2S+2T+2R=6{,}291{,}552\ \mathrm{B}.
\end{aligned}
\]

For **three limbs**, the source formulas give:

| N | One operand S | Both operands | Constants / constructor upload | All six allocations |
|---|---:|---:|---:|---:|
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
Methods now check lengths and scan canonical residue ranges before upload, then
select the owning device. They do not store a runtime coefficient/evaluation tag.
The caller is responsible for representation and
parameter compatibility. A vector from another modulus set with the same
length is not rejected merely for having different provenance.

`n()` and `limbs()` return scalar geometry without a device copy. `q(k)` and
`ref(k)` use bounds-checked host vectors; `ref(k)` is a borrowed const reference
valid only while the transform object exists. These getters do not expose the
GPU allocation or transfer data.

## Follow a complete multiplication

<div class="multiply-flow">
<div class="flow-host"><strong>HOST · input</strong><p>left: 196,608 residues<br>right: 196,608 residues</p></div>
<div class="flow-arrow">↓ H2D · 2 copies · 3 MiB total</div>
<div class="flow-device"><strong>GPU · reusable A and B</strong>
<div class="dataflow-grid">
<div><strong>A · 1.5 MiB</strong><p>Coefficients<br>↓ column DIF · 256 blocks/tower<br>↓ row DIF · 256 blocks/tower<br>Bit-reversed evaluations</p></div>
<div><strong>B · 1.5 MiB</strong><p>Coefficients<br>↓ column DIF · 256 blocks/tower<br>↓ row DIF · 256 blocks/tower<br>Bit-reversed evaluations</p></div>
</div>
<div class="flow-arrow">↓ A × B, write A · 1 kernel · 768 blocks</div>
<div><strong>A · product evaluations</strong><p>↓ row DIT → column DIT → inverse scale<br>3 kernels; result: 196,608 coefficient residues</p></div>
<p class="flow-constants">Constants stay here: forward powers 1,572,888 B + inverse powers 1,572,888 B + moduli 24 B + inverse lengths 24 B.</p>
</div>
<div class="flow-arrow">↓ D2H · 1 copy · 1.5 MiB</div>
<div class="flow-host"><strong>HOST · returned left</strong><p>196,608 residues · 1.5 MiB · natural coefficient order</p></div>
</div>

For this three-tower example, each transform launch has a 256×3 grid,
256 threads per block, and 2 KiB shared memory per block. The pointwise and
scale grids each have 768 blocks. A complete multiply uses **eight launches
and three operand copies, totaling 4,718,592 B (4.5 MiB)**.


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

| Route for two input polynomials | Copies | Requested transfer payload | Kernel launches |
|---|---:|---:|---:|
| Four public calls: forward A, forward B, pointwise, inverse | 9 | 14,155,776 B (13.5 MiB) | 8 |
| One `multiply(A,B)` call | 3 | 4,718,592 B (4.5 MiB) | 8 |


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

- **Constructor cleanup:** the original raw-Impl ownership gap is repaired with
  `unique_ptr`. Partial device state now reaches the cleanup destructor during
  unwinding. Real allocation/upload failure injection remains required; host
  rejection tests do not prove CUDA cleanup under a failed runtime/context.
- The wrapper captures its owning device and guards operations/destruction.
  Native wrong-device, restore-failure and cleanup-error tests are still open.
- No stream/event lifetime contract, buffer view, representation/level tag,
  evaluation-key cache, or OpenFHE Q/P base-switch integration exists in this
  interface. Reusable allocations are not the final integration API.
- The wrapper compiles and links for sm_89/sm_120 with stock CUDA 13.0.
  Launch/barrier, arithmetic and driver behavior remain unverified on hardware.

[D9's backend proposal](device-backend-contract.md), accepted by
[D10's ADR](architecture-decision.md), specifies these ownership/API decisions;
the production device API and baseline RAII changes still require explicit
native failure/lifetime tests. The
[D6 arithmetic review](modular-arithmetic-contract.md) examines arithmetic
bounds, parameter assumptions, and adversarial exactness.

## Verification and limits

[Source-contract tests](../tests/docs/test_cuda_dataflow.py) check that this map
matches current allocation sizes, frees, public copy directions, operation
ordering and host-only interface shape. They also pin the N=65536 arithmetic
above. These are deliberately **static source checks**, not CUDA-runtime tests
or proof of constructor failure safety. Existing CTests verify CPU arithmetic
and shared stage helpers. No new claim of actual GPU support/performance is
made by completing D5.
