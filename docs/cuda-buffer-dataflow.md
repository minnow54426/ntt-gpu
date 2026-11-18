<div id="cuda-buffers-and-hostdevice-dataflow-d5" aria-hidden="true"></div>

# Where the CUDA wrapper stores and moves data

The synchronous `CudaNttTransform` accepts host vectors. Each call uploads
inputs, runs kernels and downloads the result. Its fused `multiply()` keeps
intermediate operations on the device; separate public calls repeat transfers.

This page maps the [interface](../include/ntt/cuda_ntt.hpp) and
[implementation](../src/cuda/ntt_kernels.cu). It describes the baseline wrapper,
not the newer [resident-handle API](resident-operation-chain.md).
Use the [work-distribution explorer](ntt-65536-work-distribution.html) for the
kernel schedule. The [Ada report](batch40-native-analysis.md) records native
baseline evidence; later runtime changes need their own verification.

## The big picture: N=65536, three towers

One `CudaNttTransform` fixes the polynomial length N and the ordered pairs
\((q_k,\psi_k)\): a prime modulus and an NTT root for each tower. The caller
supplies these parameters and packed **host vectors**. One instance can then
process different polynomials with that same layout.

### What are the three towers?

Write a polynomial as \(a(x)=a_0+a_1x+\cdots+a_{N-1}x^{N-1}\).
In the Residue Number System (RNS), each coefficient modulo
\(Q=q_0q_1q_2\) is represented by three remainders:

\[
a_i\quad\longrightarrow\quad
\begin{pmatrix}
a_i\bmod q_0\\
a_i\bmod q_1\\
a_i\bmod q_2
\end{pmatrix}.
\]

A **tower** collects all N remainders for one prime. Three towers therefore
represent **one polynomial under three primes**, not three ciphertexts or three
unrelated messages. Each prime fits in a machine word even when their product
Q does not. The NTT runs independently in each tower, using that tower's q and ψ.

In a CKKS pipeline, a ciphertext commonly contains polynomial components such
as c₀ and c₁; each component has its own towers. These buffers hold polynomial
residues, not the user's decoded complex slots. This standalone wrapper takes
the packed values from its caller; it does not encrypt data or unpack an
OpenFHE ciphertext automatically.

### See the actual numbers in a small example

For illustration only, take **N=4**, primes **17, 41, 73**, and
\(a(x)=1-2x+20x^2+100x^3\). Reduce each coefficient separately. On small screens, scroll wide tables horizontally.

<div class="site-table-scroll" role="region" aria-label="Coefficient residues under three primes" tabindex="0" markdown="1">
<div style="min-width:22rem" markdown="1">

| \(a_i\) | Mod 17 | Mod 41 | Mod 73 |
|---|---:|---:|---:|
| 1 | 1 | 1 | 1 |
| −2 | 15 | 39 | 71 |
| 20 | 3 | 20 | 20 |
| 100 | 15 | 18 | 27 |

</div>
</div>

For example, −2 becomes q−2 in each tower, while \(100\bmod41=18\).
The wrapper receives only these nonnegative, canonical residues. Casting −2
to an unsigned integer is not the same operation as reducing it modulo q.

Read down each prime column in the table, then concatenate the columns.
The input is flattened **tower by tower**, not coefficient by coefficient:

```cpp
std::vector<uint64_t> values{
    1, 15,  3, 15,   // tower 0: all four coefficients modulo 17
    1, 39, 20, 18,   // tower 1: all four coefficients modulo 41
    1, 71, 20, 27    // tower 2: all four coefficients modulo 73
};
```

This array has \(3\times4=12\) words, or \(12\times8=96\) bytes.
The example's roots can be `[2, 3, 10]`, respectively; each satisfies
\(\psi_k^4=-1\pmod{q_k}\). These small parameters illustrate storage, not the
N=65536 workload. Actual primes and roots come from the chosen context; see
[the OpenFHE representation contract](openfhe-representation-contract.md).

In the first tower, forward NTT changes `[1, 15, 3, 15]` into `[10, 16, 3, 9]`.
The first array contains coefficient residues; the second contains evaluations
in bit-reversed order. Both still occupy four 8-byte words. The
[evaluation and inverse formulas](openfhe-representation-contract.md#3-exact-evaluation-and-inverse-formulas)
explain what those new values mean.

### Scale the same layout to N=65536

Now use **N=65536 and K=3**. Here K counts prime towers in one polynomial.
For tower k and position i:

\[
\begin{aligned}
\text{word index}&=kN+i,\\
\text{byte offset}&=8(kN+i).
\end{aligned}
\]

<div class="site-table-scroll" role="region" aria-label="Tower word ranges and byte offsets" tabindex="0" markdown="1">
<div style="min-width:28rem" markdown="1">

| Tower | Word range | Byte offset |
|---|---|---:|
| 0 (q₀) | 0…65535 | 0 |
| 1 (q₁) | 65536…131071 | 524288 |
| 2 (q₂) | 131072…196607 | 1048576 |

</div>
</div>

In coefficient format, `values[N + 5]` is coefficient 5 modulo q₁.
After forward NTT, the same location holds evaluation position 5 of that
tower in bit-reversed order. **The values change meaning, but the array still
has K×N entries.** Inverse returns coefficient format. Ring multiplication
also returns N coefficients per tower because the result is reduced modulo
\(x^N+1\).

Every entry uses `uint64_t`: **64 bits ÷ 8 bits per byte = 8 bytes**. A 50- or 60-bit modulus
does not shrink this implementation's storage word. Therefore:

\[
\begin{aligned}
\text{One tower}&=65536\times8\\
&=524288\ \text{bytes}=512\ \text{KiB},\\[0.4em]
\text{One polynomial}&=3\times65536\times8\\
&=1572864\ \text{bytes}=1.5\ \text{MiB}.
\end{aligned}
\]

KiB means 1024 bytes; MiB means \(1024^2=1048576\) bytes. The polynomial
occupies one contiguous buffer containing all three towers. A host vector and
its GPU copy have the same numeric payload, but live in separate memory.

### What else is stored besides the polynomial?

The wrapper allocates **two operand buffers**, A and B, for two different
polynomial inputs. Each buffer contains all three towers and is 1.5 MiB.
A forward-only call uses A; the instance still owns both allocations.
During fused multiplication, A and B hold inputs, then evaluations; the
pointwise product and final inverse result overwrite A.

Four more arrays contain parameters and precomputed constants:

- **Forward twiddles:** powers of each tower's ψ, used by the butterflies.
  The default build stores the first N entries in bit-reversed exponent order.
- **Inverse twiddles:** corresponding powers of ψ-inverse for the reverse transform.
- **Moduli:** one q per tower, so every arithmetic operation reduces by the
  correct prime. This is `[q₀, q₁, q₂]`, not the composite product Q.
- **Inverse lengths:** one \(N^{-1}\bmod q_k\) per tower for final normalization.
  These are modular inverses, not floating-point `1/N` values.

These constants depend on N, primes, roots and the selected transform schedule,
not on the input coefficients. The [table builder](../include/ntt/transform_stage.hpp)
and [constructor](../src/cuda/ntt_kernels.cu) show their actual generation and
uploads. There is no separate GPU root array; the roots are already encoded
in the twiddle tables.

For the small N=4 example, the moduli array is `[17, 41, 73]` and the inverse
lengths are `[13, 31, 55]`, since each value times 4 is 1 modulo its prime.
The first tower, with q=17 and ψ=2, has:

```text
forward table: [1,  4, 2,  8, 16]
inverse table: [1, 13, 9, 15, 16]
```

The first four forward exponents are `0, 2, 1, 3`. The last entry is
\(\psi^4\bmod17=16\). This shows why these arrays contain **constants**, not
another copy of the coefficient data.

### Derive the six-buffer total

The current [table allocation](../include/ntt/transform_stage.hpp) retains
**N+1 words per tower** in both build variants: N indexed entries plus the
endpoint \(\psi^N\) (or its inverse-root counterpart). That extra entry is a
table convention, not an extra coefficient. Each complete twiddle array uses:

\[
\begin{aligned}
3\times(65536+1)&=196611\ \text{words},\\
196611\times8&=1572888\ \text{bytes}.
\end{aligned}
\]

<div class="site-table-scroll" role="region" aria-label="Six persistent GPU buffers and their sizes" tabindex="0" markdown="1">
<div style="min-width:26rem" markdown="1">

| Buffer | Count | Bytes each |
|---|---:|---:|
| Operands A/B | 2 | 1,572,864 |
| Twiddle tables | 2 | 1,572,888 |
| Moduli | 1 | 24 |
| Inverse lengths | 1 | 24 |

</div>
</div>

Each small array has three words: \(3\times8=24\) bytes.

Add those six allocations:

\[
\begin{aligned}
\text{Operands}&=2\times1572864=3145728\ \mathrm B,\\
\text{Twiddles}&=2\times1572888=3145776\ \mathrm B,\\
\text{Small constants}&=24+24=48\ \mathrm B,\\
\text{Total}&=6291552\ \mathrm B.
\end{aligned}
\]

That is **6 MiB plus 96 bytes**, not the size of one polynomial. At setup,
only the four constant arrays are uploaded: \(3145776+48=3145824\) bytes.
A and B receive coefficient/evaluation data later, when an operation uploads
its host inputs. The sizes below count requested GPU buffer payloads; host
objects, CUDA context/allocator overhead and per-block shared memory are separate.

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

1. Create the uniquely owned `Impl`; pointers start null. Validate the supported
   size, prime/root pairs and checked buffer extents before CUDA calls.
2. Build CPU references, bit-reversed forward and reciprocal twiddle tables, and
   N-inverse for each tower. Set tile length `min(256,N)` and column length `N/256`.
3. Capture the current device, allocate six persistent buffers and upload four
   constant arrays. Record launch geometry: 256 threads per block,
   `max(1,N/256)` row blocks per tower and `ceil(K*N/256)` elementwise blocks.

The raw CPU NTT constructor assumes prime q. The CUDA wrapper adds prime/range,
root and extent checks in [cuda_validation.hpp](../include/ntt/cuda_validation.hpp).
Temporary host power tables are released after setup; the moduli and CPU
references remain.

## Persistent allocation and ownership map

For N=65536/K=3, define S=1,572,864 B, T=1,572,888 B, R=24 B.
In general, \(S=8KN,\ T=8K(N+1),\ R=8K\) bytes.
Every stored element is `uint64_t`.

| Device allocation | Elements / bytes | Layout and role | Writer / reader |
|---|---|---|---|
| `device_operand_a` | 196,608 / 1,572,864 B | `[limb][coefficient or evaluation position]`; working input/output | Public input copy and kernels write; result copy reads |
| `device_operand_b` | 196,608 / 1,572,864 B | Second operand; separate from A | Input copy and its forward transform write; pointwise kernel reads |
| `device_twiddle_powers` | 196,611 / 1,572,888 B | `[limb][table index]`, \(\psi^{\operatorname{rev}(i)}\) | Constructor uploads; forward kernels read |
| `device_inverse_twiddle_powers` | 196,611 / 1,572,888 B | `[limb][table index]`, \((\psi^{-1})^{\operatorname{rev}(i)}\) | Constructor uploads; inverse kernels read |
| `device_moduli` | 3 / 24 B | q for each limb | Constructor uploads; arithmetic kernels read |
| `device_inverse_transform_lengths` | 3 / 24 B | N-inverse modulo each q | Constructor uploads; final scale kernel reads |

`CudaNttTransform` owns `unique_ptr<Impl>`; `Impl` owns a shared-logic
`SetupResources<CudaSetupRuntime>` allocation owner. The six named pointers
are private aliases into that owner's allocation list:
callers cannot retain a borrowed device-buffer view. Copy construction and
copy assignment are deleted. No move operations are declared; this class is
not currently movable either. Do not use one instance concurrently: public
operations share and overwrite A/B and there is no locking or per-call scratch.

On destruction or constructor failure, the allocation owner releases acquired
pointers in reverse order while selecting/restoring their owning device.
Preflight rejection has no allocations to free. A cleanup trace records
failures, but the nonthrowing destructor cannot report them to its caller.
Failed device selection skips the free. Concurrent use or destruction is not
supported.

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
`k*(N+1)+e` selects table position e of that limb. For e<N, its exponent is
bit-reversed in the default build and e in the legacy build; position N holds
the retained endpoint. Transform kernels select limbs
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
<div><strong>A · 1.5 MiB</strong><p>Coefficients<br>↓ column stages · 256 blocks/tower<br>↓ row stages · 256 blocks/tower<br>Bit-reversed evaluations</p></div>
<div><strong>B · 1.5 MiB</strong><p>Coefficients<br>↓ column stages · 256 blocks/tower<br>↓ row stages · 256 blocks/tower<br>Bit-reversed evaluations</p></div>
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
  unwinding. Shared ownership logic now has host failure tests and a prepared
  native pre-call injection probe; neither establishes real driver OOM, DMA or
  context-error safety. Native execution of the new probe remains required; host
  rejection tests do not prove CUDA cleanup under a failed runtime/context.
- The wrapper captures its owning device and guards operations/destruction.
  Native wrong-device, restore-failure and cleanup-error tests are still open.
- No stream/event lifetime contract, buffer view, representation/level tag,
  evaluation-key cache, or OpenFHE Q/P base-switch integration exists in this
  interface. Reusable allocations are not the final integration API.
- The wrapper compiles and links for sm_89/sm_120 with stock CUDA 13.0.
  The older Ada baseline passed its selected native checks; current source and
  failed-context behavior still require the recorded native acceptance gates.

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
