# GPU NTT implementations

**Date:** 2026-11-05  
**Project target:** CKKS, OpenFHE 1.6.1, Hybrid RNS key switching, NVIDIA RTX 5090 (`sm_120`), ring dimension `65536`<br>
**Historical source snapshot:** `363f972`; see the [current GPU report](batch40-native-analysis.md) for later measurements.<br>
**Evidence policy:** implementation facts below come only from author papers, official repositories/source code, or NVIDIA documentation. Recommendations and inferences are explicitly labelled.

## Executive conclusions

The aim is to produce exactly the same arrays as OpenFHE, then make that
work faster. Three costs are worth measuring: reducing wide integer products
modulo a prime, reading values far apart in memory, and repeatedly copying
arrays between CPU and GPU. A profiler must establish which cost dominates.

- **Arithmetic:** test Shoup multiplication for fixed NTT multipliers, called
  twiddles. Test Barrett or Montgomery reduction for products of two changing
  values. These methods replace expensive remainder operations with bounded
  integer calculations.
- **Layout:** test a transpose/Four-Step schedule that places neighboring
  threads’ data together. The GPU can combine these accesses (coalescing).
- **Thread work:** let each thread keep four or eight values in its registers,
  and test whether this reduces waits between threads.
- **Integration:** keep buffers and keys on the GPU through Hybrid key switching:
  extend from Q primes to Q plus auxiliary P primes, compute, then return to Q.
- **Evidence:** use FIDESlib as the closest CKKS/OpenFHE comparison; measure
  complete operations as well as individual NTTs.

These are recommendations from the sources below, not measured speedups for
this repository.

<a id="1-current-implementation-what-it-does-well-and-where-it-differs-from-the-literature"></a>

## 1. Current implementation

### 1.1 Sourced facts from this repository

This subsection describes the historical source snapshot named above. The
current default butterfly has since changed; see the
[butterfly invariant](modular-arithmetic-contract.md#3-the-invariant-through-every-butterfly).
The historical CUDA code computes products modulo x^N+1. It folds the root
factors, or twist, into its butterflies rather than making a separate pass.
Its schedule is:

- Forward: DIF column pass followed by a DIF contiguous-tile pass.
- Inverse: DIT contiguous-tile pass followed by a DIT column pass and a separate scale kernel.
- Fixed split width and block size of 256.
- Limb-major storage `[limb][coefficient]`; `blockIdx.y` selects the limb.
- One full table of powers of `psi` and `psi^-1`, each `[limb][N+1]`.
- Two launches per forward transform for `N > 256`; three for inverse.
- Host upload and download on every public `forward`, `inverse`, and `pointwise_multiply` call; `multiply` at least keeps its two forward transforms, pointwise product, and inverse together on device.
- Device multiplication is currently `(__uint128_t(a) * b) % q`.
- CPU simulation shares the exact stage bodies with CUDA, and CUDA tests compare exact arrays to a CPU reference and Python big-integer oracle.

Primary code: [`ntt_kernels.cu`](../src/cuda/ntt_kernels.cu), [`dev_core.hpp`](../include/ntt/dev_core.hpp), [`mod_arith.hpp`](../include/ntt/mod_arith.hpp), and [`cuda_ntt.hpp`](../include/ntt/cuda_ntt.hpp).

### 1.2 Strengths worth retaining

**Recommendation/inference:** keep these properties through optimization:

- The exact mathematical contract: natural-order input, bit-reversed evaluation output, exact inverse, and exact equality to the CPU oracle.
- The shared simulator/device stage definitions where practical. Optimized arithmetic can sit behind a second policy while the reference policy remains unchanged.
- Limb-major RNS storage as the baseline. It maps naturally to one modulus per independent transform, pointwise CKKS arithmetic, and OpenFHE's [`DCRT-tower model`](https://github.com/openfheorg/openfhe-development/blob/b59e1f291965f14fbee5172cd83a20f5b856204f/src/core/include/lattice/hal/default/dcrtpoly.h#L879-L889).
- Precomputation at context construction rather than in a hot operation.
- Named-hardware benchmark provenance and automatic correctness gates.

### 1.3 Immediate limitations

| Observed source behavior | Recommended experiment |
|---|---|
| `mul_mod` uses 128-bit remainder | Division-free Shoup and Barrett policies |
| Column threads load `a[r + 256*t]`: 2,048-byte spacing | Coalesced transpose/Four-Step; measure sectors per requested byte |
| Full power table and a barrier after every radix-2 stage | Packed twiddles, Shoup quotients, radix-4/8 register work |
| Fixed 256-thread split | Tune launch shape for N=65536 and real tower/batch counts |
| Host vectors and synchronous copies | Add device views, stream ownership, and reusable scratch |

Keep the simple reference implementation as a way to check answers.
Reading and writing an array only a few times does not prove that the GPU
uses memory efficiently: widely spaced requests may fetch many unused bytes.

<a id="2-representative-implementations-and-transferable-techniques"></a>

## 2. Libraries and techniques

| Reference | Technique to evaluate |
|---|---|
| FIDESlib | Exact OpenFHE interoperability, device state, transform-boundary fusion |
| GPU-NTT / HEonGPU | Merge versus Four-Step schedules, batches, streams, memory pools |
| CKKS-GPU-CORE | Eight values per thread and padded shared memory |
| PhantomFHE | Radix-8, lazy Shoup ranges, key-switch fusion |
| TensorFHE | Operation batching; tensor arithmetic is a separate research path |
| ICICLE | Explicit ordering, device domains, and asynchronous APIs |
| cuFHE / cuHE | Resident keys and stream-aware operations in a different FHE scheme |
| Theodosian | RTX 5090 cache/write-bandwidth limits and fusion tradeoffs |

<details markdown="1">
<summary>Read the implementation evidence and source links</summary>

### 2.1 FIDESlib: the closest match to the project target

FIDESlib is a server-side GPU library for CKKS that can exchange data with OpenFHE. The cited release states support for OpenFHE 1.6.1, Hybrid key switching, CUDA 12/13 and GPU architectures sm_70 through sm_120. It says its GPU arrays are compared bit for bit with a patched OpenFHE CPU reference. These claims make it a close reference for our encryption scheme, library version and target GPU. See the [official README](https://github.com/CAPS-UMU/FIDESlib/tree/597a0b3ab072dbbe061d8cacea216411208359d2#readme) and its [bit-compatibility design record](https://github.com/CAPS-UMU/FIDESlib/blob/597a0b3ab072dbbe061d8cacea216411208359d2/BITCOMPAT.md).

Its forward transform uses Cooley–Tukey (CT) butterflies; inverse uses Gentleman–Sande (GS) butterflies. The code loads groups into shared memory, coordinates nearby threads within a warp, and reads/writes 16 bytes at a time. Templates select Shoup or Barrett modular arithmetic. Other template choices combine an NTT with the next operation, such as rescaling, plaintext multiplication, ModDown or key-product accumulation. This can avoid storing and reloading an intermediate array. See [`NTT_1D.cu`](https://github.com/CAPS-UMU/FIDESlib/blob/597a0b3ab072dbbe061d8cacea216411208359d2/src/NTT_1D.cu), [`INTT_1D.cu`](https://github.com/CAPS-UMU/FIDESlib/blob/597a0b3ab072dbbe061d8cacea216411208359d2/src/INTT_1D.cu), [`NTTfusions.cuh`](https://github.com/CAPS-UMU/FIDESlib/blob/597a0b3ab072dbbe061d8cacea216411208359d2/src/NTTfusions.cuh), and [`ModMult.cuh`](https://github.com/CAPS-UMU/FIDESlib/blob/597a0b3ab072dbbe061d8cacea216411208359d2/src/ModMult.cuh).

Its Hybrid pipeline has separate kernels for digit decomposition, extending the prime basis (ModUp), and reducing it (ModDown). It assigns prime towers to devices and tracks dependencies with CUDA streams and completion events. The scheduling code can record repeated work as a CUDA Graph. The high-level polynomial object records whether auxiliary primes are present and offers combined key-switch operations. See [`Conv.cu`](https://github.com/CAPS-UMU/FIDESlib/blob/597a0b3ab072dbbe061d8cacea216411208359d2/src/CKKS/Conv.cu), [`LimbPartition.cu`](https://github.com/CAPS-UMU/FIDESlib/blob/597a0b3ab072dbbe061d8cacea216411208359d2/src/CKKS/LimbPartition.cu), and [`RNSPoly.cpp`](https://github.com/CAPS-UMU/FIDESlib/blob/597a0b3ab072dbbe061d8cacea216411208359d2/src/CKKS/RNSPoly.cpp).

**For this project:** Use FIDESlib to study exact OpenFHE interoperability and transform-boundary fusion. Preserve QP towers and compare the complete Hybrid operation, not only NTT latency.

**Inference:** FIDESlib is both a design reference and a competitor baseline. Before designing a new OpenFHE patch layer, evaluate whether this project's intended contribution is (a) independent kernels integrated directly into stock OpenFHE, (b) an alternative backend with a stable ABI, or (c) improvements that can be contributed to FIDESlib/OpenFHE. Duplicating its entire compatibility layer without a differentiator is high effort.

### 2.2 GPU-NTT / HEonGPU: Merge NTT and Four-Step NTT

Özcan and Savaş compare an iterative “Merge NTT” with a Four-Step NTT. Their homomorphic-encryption tests include lengths 2^12 through 2^16; proving workloads use larger lengths. They choose kernel counts and block shapes to make neighboring memory accesses combine efficiently. Their results distinguish time for one transform from transforms completed per second. The paper reports cases where iterative Merge has higher throughput despite Four-Step’s more local accesses. See the author paper, [Two Algorithms for Fast GPU Implementation of NTT](https://eprint.iacr.org/2023/1410), especially its design and evaluation sections.

The cited GPU-NTT source provides Barrett arithmetic for 32- and 64-bit integers, with modulus limits of 30 and 61 bits respectively. Transform lengths are specialized at compile time, and configuration tables choose kernels for each size. The library supports prime towers, batches and caller-selected CUDA streams. It uses different execution paths for smaller rings and larger transforms. See the [official repository](https://github.com/Alisah-Ozcan/GPU-NTT), [`ntt_launch_impl.cuh`](https://github.com/Alisah-Ozcan/GPU-NTT/blob/eaeed82aaac869f7548ad1c743b3d7c08e0f6a40/src/lib/ntt_merge/ntt_launch_impl.cuh), and [`modular_arith.cuh`](https://github.com/Alisah-Ozcan/GPU-NTT/blob/eaeed82aaac869f7548ad1c743b3d7c08e0f6a40/src/include/gpuntt/common/modular_arith.cuh).

HEonGPU uses GPU-NTT inside a GPU-resident BFV/CKKS library. Its paper describes a GPU memory pool, pinned host pool, and multi-stream execution in addition to the NTT itself. See [HEonGPU](https://eprint.iacr.org/2024/1543) and the [official repository](https://github.com/enCRYPTON-labs/HEonGPU).

**For this project:** Compare Merge and Four-Step schedules with matched modulus, order and batching. Tune N=65536 across tower counts rather than copying another GPU's launch table.

### 2.3 CKKS-GPU-CORE: memory-centric CKKS and 8-point-per-thread NTT

Jung et al. keep up to eight NTT values in each thread’s private registers. Threads exchange values through shared memory between their small local transforms. Extra padding changes shared-memory addresses to reduce bank conflicts, where accesses compete for the same memory bank. For basis conversion, threads cover output primes and coefficients, add input-tower products, and then apply Barrett reduction. See [Over 100x Faster Bootstrapping in Fully Homomorphic Encryption through Memory-Centric Optimization with GPUs](https://scale.snu.ac.kr/papers/2021-09-Conference-CHES-100x.pdf) and the [official CKKS-GPU-CORE repository](https://github.com/scale-snu/ckks-gpu-core).

The source implements two NTT phases, register arrays of eight coefficients per thread, Shoup butterflies, padded shared memory, batched ModUp, and fused ModDown/NTT epilogues. See [`NttImple.cu`](https://github.com/scale-snu/ckks-gpu-core/blob/1eb16b667fc27684d82957c52b09777a8e350c0e/src/public/NttImple.cu) and [`Context.cu`](https://github.com/scale-snu/ckks-gpu-core/blob/1eb16b667fc27684d82957c52b09777a8e350c0e/src/public/Context.cu).

**For this project:** Test eight register-held values per thread, padded shared memory and conversion/NTT fusion as separate measured changes.

### 2.4 PhantomFHE: radix-8 NWT, lazy Shoup butterflies, and scheme-level fusion

Phantom supports the BFV, BGV and CKKS encryption schemes on CUDA. Small transforms run in one block. Larger transforms use two phases, eight values per thread and padded shared-memory exchanges. Its CT forward butterflies use Shoup multiplication and temporarily allow residues in [0,2q) or [0,4q). This is lazy reduction: postponing some reductions rather than keeping every intermediate below q. Its GS inverse also postpones final reduction and scaling. See the [official repository](https://github.com/encryptorion-lab/phantom-fhe), [`ntt_1d.cu`](https://github.com/encryptorion-lab/phantom-fhe/blob/1f4a198443b3af77118e51f53d5b8f332154b875/src/ntt/ntt_1d.cu), [`fntt_2d.cu`](https://github.com/encryptorion-lab/phantom-fhe/blob/1f4a198443b3af77118e51f53d5b8f332154b875/src/ntt/fntt_2d.cu), and [`butterfly.cuh`](https://github.com/encryptorion-lab/phantom-fhe/blob/1f4a198443b3af77118e51f53d5b8f332154b875/include/butterfly.cuh).

The Phantom paper describes intra-kernel fusion that reuses register temporaries and inter-kernel fusion across operations such as NWT plus scaling and ModDown plus correction. Its key-switch path connects ModUp, evaluation-key inner product, and ModDown on a supplied stream. See [Phantom: A CUDA-Accelerated Word-Wise Homomorphic Encryption Library](https://eprint.iacr.org/2023/049) and [`eval_key_switch.cu`](https://github.com/encryptorion-lab/phantom-fhe/blob/1f4a198443b3af77118e51f53d5b8f332154b875/src/eval_key_switch.cu).

**For this project:** Study radix-8 staging and stream-aware key switching. Prove each lazy-residue range before removing reductions; a different butterfly convention is not automatically array-compatible.

### 2.5 TensorFHE: tensor-core NTT and operation-level batching

TensorFHE breaks a 32-bit NTT into matrix multiplications for Tensor Cores. Its aim is high throughput across many CKKS operations, rather than minimum time for one operation. It changes the layout from (batch, limb, N) to (limb, batch, N), placing the same prime tower from different operations together in memory. Independent matrix products use separate streams. See the author paper, [TensorFHE](https://arxiv.org/abs/2212.14191), especially Sections IV-C and IV-D.

**For this project:** Measure operation batching and limb/batch layouts for throughput. Tensor arithmetic is a separate experiment with different width and exactness requirements.

**Inference/caution:** do not make tensor-core NTT the first implementation path. TensorFHE's arithmetic decomposition and throughput objective differ from OpenFHE's usual 50–60-bit RNS towers and this project's immediate exact-compatibility goal. Establish a fast integer-core Shoup/Barrett backend first; treat tensor cores as a separate experimental backend.

### 2.6 ICICLE: useful API ideas, not a direct CKKS backend

ICICLE's official NTT API exposes stream, batch size, row/column batch layout, input/output ordering, device-resident flags, async execution, reusable per-device domains, fast-twiddle mode, and an explicit mixed-radix selection. See the [official NTT API](https://github.com/ingonyama-zk/icicle/blob/main/docs/versioned_docs/version-4.0.0/api/cpp/ntt.md).

**For this project:** Borrow the API ideas: explicit order, batches, device state and readiness. Verify transform semantics and licensing before treating ICICLE as a CKKS backend.

**Inference/caution:** ICICLE is primarily a proof-system field library, not an OpenFHE CKKS/Hybrid implementation. Its API concepts transfer; its field choices and transform semantics must not be assumed compatible. Its CUDA backend also has source-availability/licensing conditions that should be checked before any reuse.

### 2.7 cuFHE/cuHE: historically relevant but not the integration model

cuFHE implements TFHE gates and credits cuHE's GPU NTT; its API supports CUDA streams and keeps a public bootstrapping key initialized on the GPU. See the [official cuFHE repository](https://github.com/vernamlab/cuFHE) and [official cuHE repository](https://github.com/vernamlab/cuHE).

**For this project:** Keep reusable keys resident and expose stream dependencies. TFHE gate bootstrapping is a different workload from CKKS Hybrid switching.

**Inference/caution:** TFHE gate bootstrapping has different polynomial sizes, dataflow, and key-switch structure from CKKS Hybrid RNS. cuFHE is not a suitable algorithmic baseline for the OpenFHE integration; it is mainly historical evidence for GPU-resident keys and stream-level concurrency.

### 2.8 Theodosian: RTX 5090 changes the optimization priority

Theodosian profiles CKKS on RTX 5090. It reports that its optimized NTT and basis-conversion kernels are limited by how fast data move through the L2 cache. Its NTT uses two kernels and can generate twiddles during execution, trading extra arithmetic for fewer table reads. Small tower counts leave GPU capacity unused, so it batches work while the data still fit effectively in L2. For Hybrid key switching, it reuses multiple polynomials in cache and schedules complementary work. CUDA Graphs reduce repeated launch setup for fixed operation sequences. See [Theodosian](https://arxiv.org/abs/2512.18345).

The paper reports that batching can hurt after the footprint exceeds the 5090's effective L2 capacity, and that key switching benefits more from cache-aware multi-polynomial execution and graph scheduling than from additional local kernel fusion once the baseline is already heavily fused.

**For this project:** Profile the actual RTX 5090's L2 traffic and working-set size. Test twiddle generation, cache-aware batching and graphs only after measuring their tradeoffs.


</details>

<a id="3-algorithm-and-arithmetic-choices-for-this-project"></a>

## 3. Algorithms and arithmetic

### 3.1 CT/GS versus Stockham and Four-Step

**Sourced observation:** Phantom and FIDESlib use CT forward plus GS inverse; GPU-NTT compares iterative Merge and Four-Step schedules; ICICLE exposes mixed-radix and ordering choices.

**Recommendation:** keep CT/DIF forward and GS/DIT inverse as the first optimized path because the current repository already has the equivalent ordering contract and no bit-reversal pass. Add a Four-Step/transpose prototype only as a benchmarked alternative for the strided column pass. A Stockham autosort path may offer regular accesses and natural ordering, but it normally ping-pongs global buffers; for the target sizes, its extra full-array traffic must beat the cost of the current uncoalesced gather before adoption.

### 3.2 Mixed radix and register staging

**Sourced observation:** CKKS-GPU-CORE and Phantom use up to eight values per thread and radix-8 local transforms; GPU-NTT notes higher-radix butterflies as a further optimization.

**Recommendation:** implement radix-4 first, then radix-8. Radix describes
how many values a local transform combines at once. Registers are private
storage for one thread; shared memory lets threads in a block exchange values.
Test this sequence:

1. Load 4/8 values per thread using coalesced vectorized transactions.
2. Execute the local stages in registers.
3. Store into padded shared memory for the permutation between local groups.
4. Use warp shuffles where partner lanes remain within a warp.
5. Store once at the end of the phase.

This schedule can reduce multiplier-table reads, block-wide waits (barriers),
and shared-memory accesses compared with assigning only one butterfly to each
thread per stage. Measure register use and speed to check the benefit.

### 3.3 Modular multiplication

**Sourced observation:** GPU-NTT uses Barrett with explicit modulus-bit bounds; Phantom and CKKS-GPU-CORE use Shoup values for twiddles and lazy butterflies; FIDESlib provides Shoup and Barrett paths.

**Recommendation:** use three explicit primitives rather than one universal `mul_mod`:

- `mul_const_shoup(x, w, w_shoup, q)` for twiddles, `N^-1`, basis constants, and evaluation-key constants when precomputation is useful.
- `mul_var_barrett(a, b, q, mu)` or a benchmarked Montgomery alternative for pointwise variable products.
- `reduce_wide(acc, q, mu)` for basis-conversion/key-switch accumulators with a proved accumulator bound.

Retain the current `__uint128_t % q` only as host/device test oracle. Test each optimized primitive exhaustively around `0`, `q-1`, lazy bounds, and the maximum supported OpenFHE prime size.

### 3.4 Twiddle representation

**Recommendation/inference:** replace `[psi^0, ..., psi^N]` as the only device representation with:

- stage/phase-major twiddles in the order consumed by warps;
- a parallel Shoup quotient table;
- optional compact “seed” twiddles for on-the-fly expansion in the phase with `O(N)` table traffic;
- a metadata record containing transform convention, direction, modulus, ordering, and supported lazy range.

The CPU reference can continue to build a simple power table; a packing step should produce the device format and be independently cross-checked.

<a id="4-rns-data-layout-and-hybrid-key-switch-implications"></a>

## 4. RNS layout and Hybrid state

### 4.1 Baseline layout

**Recommendation:** keep coefficient data limb-major:

```text
poly[component][digit-or-batch][limb][coefficient]
```

Here adjacent coefficients are adjacent in memory. A limb is one prime
tower. Neighboring threads can then load nearby coefficients for the same
prime, and all threads in a block can use the same modulus constants. Views
should describe the spacing between dimensions so batches can be interpreted
without copying the arrays.

### 4.2 Extended-basis state

**Recommendation:** make these states explicit and non-interchangeable:

- `Q, coefficient`
- `Q, evaluation`
- `QP, coefficient`
- `QP, evaluation`

Q denotes the ciphertext primes; P denotes extra primes used during key
switching. “Coefficient” and “evaluation” identify what the stored numbers
mean. A buffer view must record the active Q and P counts, current level,
array order, device and queued-work dependencies. Otherwise the adapter could
drop needed towers or transform values that are already transformed.

### 4.3 Target Hybrid pipeline

**Recommendation/inference:** design the first end-to-end kernel graph around OpenFHE's actual Hybrid formulation, approximately:

```text
input c2 in Q/evaluation
  -> digit partition / inverse NTT as required
  -> Q-digit to QP basis extension
  -> forward NTT of extended digits
  -> evaluation-key multiply-accumulate into two QP outputs
  -> approximate ModDown QP -> Q
  -> add/fold c0 as required by the exact OpenFHE operation
```

Read the selected OpenFHE 1.6.1 source to determine exactly where transforms
and rounding occur, then test those boundaries. Two formulas can describe
the same modular result yet select different intermediate integers. Exact
array compatibility requires OpenFHE’s choices at each checked step.

### 4.4 Fusion candidates

High-value, low-ambiguity experiments:

- INTT final scaling + decomposition pre-scale.
- Basis-extension output + first forward-NTT load/prologue.
- Forward-NTT final store + evaluation-key dot product when register and occupancy costs permit.
- INTT or NTT boundary + ModDown correction.
- Pointwise accumulation across Hybrid digits, deferring reduction only under a proved bound.

Combining operations in one kernel (fusion) can save memory traffic. It can
also use more registers or shared memory, leaving room for fewer active
threads, or make cache reuse worse. Theodosian reports that more fusion is
not always faster on RTX 5090. Measure each combined version.

<a id="5-streams-transfers-allocation-and-launch-strategy"></a>

## 5. Streams and memory

NVIDIA documents that asynchronous host/device copies require pinned host memory and can overlap with kernels in other streams. It also documents global-to-shared asynchronous copies and pipelines, and states that CUDA Graphs reduce repeated launch setup cost for fixed workflows. See the [CUDA Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/#asynchronous-and-overlapping-transfers-with-computation), [asynchronous copy documentation](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/async-copies.html), and [CUDA Graphs documentation](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html).

Recommended API split:

- `NttContext`: immutable moduli, roots, packed twiddles, reduction constants, launch table, and allocator handle.
- `DeviceRnsView`: pointer(s), dimensions/strides, basis/domain/order metadata, device, and readiness event.
- `enqueue_forward/inverse(view, stream)`: no allocation, no synchronization, no host copy.
- convenience host methods: pinned staging plus explicit or documented synchronization.
- reusable scratch arena sized for maximum supported level/digit count.
- captured CUDA graphs for fixed ring dimension, level, digit count, and operation sequence.

**Recommendation/inference:** try `cuda::memcpy_async`/pipeline only after the coalesced phase layout is fixed. The current column gather has no useful sequential “next tile” per block to hide without first changing data ownership; async copy cannot repair uncoalesced addresses.

<a id="6-correctness-and-benchmark-methodology"></a>

## 6. Correctness and measurement

### 6.1 Correctness ladder

Check primitives, butterflies, full transforms and polynomial products before
composing basis conversion and Hybrid operations. Compare complete arrays with
the same parameters and keys. [The arithmetic guide](modular-arithmetic-contract.md)
explains the independent oracles; [the benchmark plan](benchmark-tuning-plan.md)
defines the full matrix and required record.

### 6.2 Required performance views

Separate setup, resident-device events, transfer-inclusive calls and complete
operations. Report latency and throughput across ring sizes, widths and tower
counts. Retain raw samples, named hardware, peak memory and profiler counters;
[the collector guide](offline-test-tooling.md) explains the implemented scopes.
A kernel-only time excludes CPU–GPU copies and other call costs. Compare
it only with a measurement that excludes the same work.

### 6.3 Baselines

- This repository's current pre-Barrett kernel, labelled clearly as a correctness baseline.
- Optimized kernel with identical transfers.
- GPU-NTT or another standalone GPU NTT with matched modulus/order/batch where license and semantics permit.
- FIDESlib 2.2/OpenFHE 1.6.1 for NTT microbenchmarks and, more importantly, full Hybrid key switch/rotation.
- Stock OpenFHE 1.6.1 CPU, with build flags and thread count recorded.

Do not compare one implementation's kernel-only time with another's transfer-inclusive time, nor one-limb latency with an entire RNS polynomial.

<a id="7-prioritized-points-to-absorb"></a>

## 7. Priorities

| Priority | Deliverable |
|---|---|
| P0 | Real-GPU exact checks, arithmetic policies, device-buffer/stream contract |
| P1 | Shoup/Barrett kernels, register staging, coalesced layout, measured dispatch |
| P2 | Explicit Q/QP state, exact basis conversion, resident keys, safe fusion |
| P3 | Hoisting, operation batching, graph replay, whole-inference profiling |

The [integration plan](openfhe-integration-plan.md) tracks completion; the
[benchmark plan](benchmark-tuning-plan.md) defines the workload and acceptance
record. The priorities here do not mark a roadmap gate complete.

<a id="8-licensing-and-clean-room-implementation"></a>

## 8. Licensing and provenance

This section is informational, not legal advice.

- **FIDESlib:** MIT license in its official repository. Its OpenFHE interoperability currently depends on patches, which should be reviewed separately for upstream/license provenance.
- **GPU-NTT:** Apache-2.0.
- **HEonGPU:** check the repository's current license before copying code; concepts described by the paper can be independently implemented.
- **PhantomFHE:** GPLv3. Copying its source into a differently licensed project can impose copyleft obligations. Use the paper and behavior as design evidence unless the project intentionally adopts compatible licensing.
- **CKKS-GPU-CORE:** CC BY-NC-SA 4.0 in the repository. The non-commercial/share-alike terms make direct code reuse unsuitable for many projects. Independently implement paper-level ideas and preserve clean-room provenance.
- **ICICLE CUDA backend:** source availability and the backend license have changed across releases. Verify the exact version's terms before any reuse.
- **cuFHE/cuHE:** their official repositories state their licenses; verify individual files before reuse.

**Recommendation:** continue the repository's clean-room policy. Cite papers and inspect behavior/interfaces, but write original kernels. If code is adapted from MIT/Apache sources, preserve required notices and record the exact file/commit. Do not copy GPL or CC BY-NC-SA kernel code unless the project's license decision explicitly permits it.

<a id="9-decision-summary-for-the-next-code-update"></a>

## 9. Next implementation steps

Follow the [current roadmap](openfhe-integration-plan.md) and
[backlog](implementation-backlog.md), rather than treating this November 5
survey as the live tracker:

1. Validate the current source and resident chain on RTX 5090; collect profiling.
2. Test Shoup/Barrett arithmetic and coalesced layouts independently against the
   reference path. Compare radix-4/8 variants under the same timing boundary.
3. Select measured configurations for real tower/batch counts.
4. Extend resident buffers into exact base conversion and Hybrid key switching,
   with FIDESlib as a relevant complete-operation comparison.

Keep explanations tied to the selected implementation. Four logical array
movements do not prove coalesced traffic, and more fusion or larger batches can
hurt when they increase register pressure or exceed cache capacity.

## Primary sources

<details markdown="1">
<summary>Papers, repositories, and NVIDIA documentation</summary>

- Özcan and Savaş, [Two Algorithms for Fast GPU Implementation of NTT](https://eprint.iacr.org/2023/1410); [GPU-NTT source](https://github.com/Alisah-Ozcan/GPU-NTT).
- Özcan and Savaş, [HEonGPU](https://eprint.iacr.org/2024/1543); [HEonGPU source](https://github.com/enCRYPTON-labs/HEonGPU).
- Yang et al., [Phantom](https://eprint.iacr.org/2023/049); [PhantomFHE source](https://github.com/encryptorion-lab/phantom-fhe).
- Jung et al., [Over 100x Faster Bootstrapping through Memory-Centric Optimization with GPUs](https://scale.snu.ac.kr/papers/2021-09-Conference-CHES-100x.pdf); [CKKS-GPU-CORE source](https://github.com/scale-snu/ckks-gpu-core).
- Fan et al., [TensorFHE](https://arxiv.org/abs/2212.14191).
- Agulló-Domingo et al., [FIDESlib source and documentation](https://github.com/CAPS-UMU/FIDESlib).
- Choi et al., [Theodosian](https://arxiv.org/abs/2512.18345).
- Ingonyama, [ICICLE NTT API](https://github.com/ingonyama-zk/icicle/blob/main/docs/versioned_docs/version-4.0.0/api/cpp/ntt.md).
- Vernam Lab, [cuFHE](https://github.com/vernamlab/cuFHE) and [cuHE](https://github.com/vernamlab/cuHE).
- NVIDIA, [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/), [CUDA C++ Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/), and [Nsight Compute documentation](https://docs.nvidia.com/nsight-compute/).

</details>
