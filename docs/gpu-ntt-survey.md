# GPU NTT survey for CKKS, Hybrid key switching, OpenFHE, and RTX 5090

**Date:** 2026-11-05  
**Project target:** CKKS, OpenFHE 1.6.1, Hybrid RNS key switching, NVIDIA RTX 5090 (`sm_120`), ring dimensions `4096..65536`  
**Current repository revision inspected:** `363f972`  
**Evidence policy:** implementation facts below come only from author papers, official repositories/source code, or NVIDIA documentation. Recommendations and inferences are explicitly labelled.

## Executive conclusions

1. **The current two-pass structure is a reasonable correctness scaffold, not yet a competitive GPU NTT.** It has a valuable exact-reference contract, only two forward launches, shared-memory stage execution, and RNS-limb batching. Its principal performance blockers are device-side `__uint128_t % q`, severely strided global accesses in the column pass, a host-only synchronous API, and one fixed launch shape for all sizes and limb counts.
2. **FIDESlib is the most important comparison.** It implements CKKS Hybrid key switching, targets OpenFHE 1.6.1, supports `sm_120`, keeps CKKS objects on the GPU, fuses NTT/INTT boundaries with rescale, ModDown, and key-switch dot products, and tests exact GPU/CPU ciphertext equality. It is substantially closer to this project's end goal than a standalone NTT benchmark.
3. **Use Shoup multiplication for precomputed twiddles first.** Every butterfly has a known multiplier, making Shoup's precomputed quotient a natural replacement for division. Use a separate Barrett/Montgomery-style path for variable-by-variable pointwise products. Introduce lazy ranges only with explicit range proofs and boundary tests.
4. **Replace the fixed radix-2-per-thread inner loop with multi-butterfly thread work.** The strongest CKKS implementations hold 4 or 8 values in registers per thread, perform radix-4/radix-8 local work, exchange through padded shared memory, and reduce synchronization and twiddle traffic.
5. **Treat global layout and key-switch scheduling as first-class design decisions.** On RTX 5090, recent analysis finds optimized NTT and basis-conversion kernels limited by L2/write bandwidth. The current column gather uses a 2,048-byte stride between neighboring threads' 64-bit loads at `N >= 512`, so nominal byte counts understate the actual memory-transaction cost.
6. **Do not optimize NTT in isolation.** The target unit is the GPU-resident Hybrid key-switch sequence: inverse NTT/decomposition, basis extension, forward NTT, evaluation-key multiply-accumulate, and basis reduction. Fuse safe producer/consumer boundaries and retain the extended `QP` representation when the OpenFHE operation sequence allows it.
7. **Benchmark both latency and throughput.** A single ciphertext may underfill a 5090. Report kernel-only and end-to-end latency, batched tower/ciphertext throughput, memory traffic, achieved bandwidth, occupancy, and the complete key-switch/rotation latency. Always attach device, clocks, toolkit, driver, modulus bits, tower count, ordering, and transfer policy.

## 1. Current implementation: what it does well and where it differs from the literature

### 1.1 Sourced facts from this repository

The CUDA backend uses a negacyclic transform with folded twisting and the following schedule:

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
- Limb-major RNS storage as the baseline. It maps naturally to one modulus per independent transform, pointwise CKKS arithmetic, and OpenFHE's DCRT-tower model.
- Precomputation at context construction rather than in a hot operation.
- Named-hardware benchmark provenance and automatic correctness gates.

### 1.3 Immediate limitations

**Repository fact:** `mul_mod` performs 128-bit remainder on every device multiply. The source itself marks this as a correctness-first, pre-Barrett implementation.

**Recommendation/inference:** this must be replaced before NTT timing says anything useful. A radix-2 butterfly currently performs two modular products in this formulation, so division emulation dominates any memory-layout comparison.

**Repository fact:** in `dif_col_kernel`, neighboring threads access `a[r + 256*t]`. For 64-bit coefficients, adjacent threads are separated by 2,048 bytes. The scatter has the same pattern.

**Recommendation/inference:** this is the largest structural memory concern. Although the algorithm moves each coefficient through global memory only once per pass, the column pass is not warp-coalesced. Measure actual DRAM sectors and L2 transactions; do not describe the implementation as “four round-trips” without also reporting transaction efficiency.

**Repository fact:** the tile and column kernels fetch twiddles from a full global table inside each stage, and synchronize the whole block after every radix-2 stage.

**Recommendation/inference:** stage-major twiddle packing, precomputed Shoup quotients, register-held radix-4/8 work, warp shuffles for the smallest distances, and padded shared-memory exchange are the highest-value inner-kernel experiments.

**Repository fact:** every transform uses 256 threads and a 256-way split, independent of `N`, modulus count, register pressure, shared-memory use, and GPU architecture.

**Recommendation/inference:** replace the single hard-coded shape with a small, benchmarked dispatch table for `N = 2^12..2^16` and for low/high tower counts. Compile-time specialization for five supported sizes is reasonable and permits unrolling and constant launch geometry.

**Repository fact:** the public interface accepts host `std::vector` objects and uses synchronous `cudaMemcpy`; it has no stream, device-buffer, external allocator, or graph-capture contract.

**Recommendation/inference:** retain this API as a convenience/reference facade, but add a device-resident API before OpenFHE integration. Otherwise host transfers and forced synchronization will erase key-switch gains.

## 2. Representative implementations and transferable techniques

### 2.1 FIDESlib: the closest match to the project target

FIDESlib describes itself as a server-side CKKS GPU library interoperable with OpenFHE. Its current release states support for OpenFHE 1.6.1, Hybrid key switching, CUDA 12/13, and `sm_70` through `sm_120`; it also states that GPU operations are compared bit-for-bit to a patched OpenFHE CPU reference. These are precisely this project's scheme, integration, key-switch, and GPU targets. See the [official README](https://github.com/CAPS-UMU/FIDESlib/tree/597a0b3ab072dbbe061d8cacea216411208359d2#readme) and its [bit-compatibility design record](https://github.com/CAPS-UMU/FIDESlib/blob/597a0b3ab072dbbe061d8cacea216411208359d2/BITCOMPAT.md).

Its current 1D NTT uses CT butterflies for forward and GS butterflies for inverse, shared-memory staging, warp-level synchronization below a warp, vectorized 16-byte loads/stores, and Shoup/Barrett arithmetic selected by templates. More importantly, the NTT mode is templated to fuse downstream work including rescale, plaintext multiplication, ModDown, and key-switch dot/accumulate operations. See [`NTT_1D.cu`](https://github.com/CAPS-UMU/FIDESlib/blob/597a0b3ab072dbbe061d8cacea216411208359d2/src/NTT_1D.cu), [`INTT_1D.cu`](https://github.com/CAPS-UMU/FIDESlib/blob/597a0b3ab072dbbe061d8cacea216411208359d2/src/INTT_1D.cu), [`NTTfusions.cuh`](https://github.com/CAPS-UMU/FIDESlib/blob/597a0b3ab072dbbe061d8cacea216411208359d2/src/NTTfusions.cuh), and [`ModMult.cuh`](https://github.com/CAPS-UMU/FIDESlib/blob/597a0b3ab072dbbe061d8cacea216411208359d2/src/ModMult.cuh).

FIDESlib's Hybrid pipeline has dedicated decomposition/ModUp and ModDown kernels, per-device limb partitions, CUDA streams/events, and graph capture in its multi-GPU/scheduling layer. Its high-level RNS object tracks whether a polynomial is in the extended basis and exposes fused key-switch operations. See [`Conv.cu`](https://github.com/CAPS-UMU/FIDESlib/blob/597a0b3ab072dbbe061d8cacea216411208359d2/src/CKKS/Conv.cu), [`LimbPartition.cu`](https://github.com/CAPS-UMU/FIDESlib/blob/597a0b3ab072dbbe061d8cacea216411208359d2/src/CKKS/LimbPartition.cu), and [`RNSPoly.cpp`](https://github.com/CAPS-UMU/FIDESlib/blob/597a0b3ab072dbbe061d8cacea216411208359d2/src/CKKS/RNSPoly.cpp).

What to absorb:

- Make exact OpenFHE ciphertext compatibility—not only decrypt-level CKKS tolerance—the integration acceptance target where the same algorithmic formulation is used.
- Model transform state explicitly: coefficient/evaluation order, normal/extended RNS basis, level, and owning device/stream.
- Create fusion points at transform boundaries. A generic `forward()` that always stores a complete intermediate is insufficient for high-performance key switching.
- Use Shoup constants beside twiddle and key constants. Keep generic Barrett reduction for operands whose multiplier is not precomputed.
- Test the complete Hybrid formulations. FIDESlib's compatibility record shows that two mathematically valid rotation/key-switch formulations need not produce bit-identical ciphertexts because rounding and noise enter in different orders.
- Preserve full `QP` towers during export/import. FIDESlib documents a real compatibility bug caused by truncating extended-basis ciphertexts during materialization.

**Inference:** FIDESlib is both a design reference and a competitor baseline. Before designing a new OpenFHE patch layer, evaluate whether this project's intended contribution is (a) independent kernels integrated directly into stock OpenFHE, (b) an alternative backend with a stable ABI, or (c) improvements that can be contributed to FIDESlib/OpenFHE. Duplicating its entire compatibility layer without a differentiator is high effort.

### 2.2 GPU-NTT / HEonGPU: Merge NTT and Four-Step NTT

Özcan and Savaş present two GPU algorithms: an iterative “Merge NTT” and a Four-Step NTT. Their target ranges include `2^12..2^16` for HE and larger sizes for proving. They optimize kernel count and block shape for coalesced global access, and report both single-transform latency and concurrent-transform throughput. The paper reports that the iterative form can outperform Four-Step for HE throughput despite Four-Step's locality. See the author paper, [Two Algorithms for Fast GPU Implementation of NTT](https://eprint.iacr.org/2023/1410), especially its design and evaluation sections.

The current official GPU-NTT repository supports 32- and 64-bit Barrett arithmetic, rejecting moduli above 30 and 61 bits respectively. It specializes ring sizes at compile time, uses per-size kernel configuration tables, supports RNS transforms and batches, accepts CUDA streams, and has different low-ring and multi-stage paths. See the [official repository](https://github.com/Alisah-Ozcan/GPU-NTT), [`ntt_launch_impl.cuh`](https://github.com/Alisah-Ozcan/GPU-NTT/blob/eaeed82aaac869f7548ad1c743b3d7c08e0f6a40/src/lib/ntt_merge/ntt_launch_impl.cuh), and [`modular_arith.cuh`](https://github.com/Alisah-Ozcan/GPU-NTT/blob/eaeed82aaac869f7548ad1c743b3d7c08e0f6a40/src/include/gpuntt/common/modular_arith.cuh).

HEonGPU uses GPU-NTT inside a GPU-resident BFV/CKKS library. Its paper describes a GPU memory pool, pinned host pool, and multi-stream execution in addition to the NTT itself. See [HEonGPU](https://eprint.iacr.org/2024/1543) and the [official repository](https://github.com/enCRYPTON-labs/HEonGPU).

What to absorb:

- Compare at least two schedules: an improved two-pass/Merge path and a Four-Step or explicit transpose path. Do not assume the current fixed split is optimal.
- Build a per-size launch table. The supported range is only five powers of two, so exhaustive offline tuning on the 5090 is tractable.
- Treat batches as part of the NTT API, not a benchmark wrapper. CKKS naturally supplies many modulus towers and often multiple ciphertext components.
- Report throughput as well as latency; the best shape changes when enough independent transforms exist to saturate the GPU.
- Add an allocator and stream contract at the library boundary.

### 2.3 CKKS-GPU-CORE: memory-centric CKKS and 8-point-per-thread NTT

Jung et al.'s memory-centric CKKS work keeps up to eight NTT values per thread in registers, exchanges through shared memory between local subtransforms, and pads shared memory to reduce bank conflicts. Its basis conversion launches across output limbs and coefficients and accumulates input-limb products before Barrett reduction. See [Over 100x Faster Bootstrapping in Fully Homomorphic Encryption through Memory-Centric Optimization with GPUs](https://scale.snu.ac.kr/papers/2021-09-Conference-CHES-100x.pdf) and the [official CKKS-GPU-CORE repository](https://github.com/scale-snu/ckks-gpu-core).

The source implements two NTT phases, register arrays of eight coefficients per thread, Shoup butterflies, padded shared memory, batched ModUp, and fused ModDown/NTT epilogues. See [`NttImple.cu`](https://github.com/scale-snu/ckks-gpu-core/blob/1eb16b667fc27684d82957c52b09777a8e350c0e/src/public/NttImple.cu) and [`Context.cu`](https://github.com/scale-snu/ckks-gpu-core/blob/1eb16b667fc27684d82957c52b09777a8e350c0e/src/public/Context.cu).

What to absorb:

- Evaluate radix-8-per-thread as a direct evolution of this repository's two-pass structure: keep eight coefficients in registers, use shared memory only for cross-thread permutations, and reduce full-block barriers.
- Add padding and profile bank conflicts rather than assuming the `[256]` shared tile is conflict-free for every access pattern.
- Co-design basis conversion and NTT buffers. The same extended-basis layout should flow directly into forward NTT and evaluation-key multiplication.
- Fuse inverse-transform epilogues and forward-transform prologues where the Hybrid formula permits.

### 2.4 PhantomFHE: radix-8 NWT, lazy Shoup butterflies, and scheme-level fusion

Phantom implements BFV, BGV, and CKKS on CUDA. Its 1D path handles small transforms in one block; its larger 2D path performs radix-8 work with eight register-held values per thread, padded shared-memory exchanges, and two phases. Forward CT butterflies use Shoup multiplication and lazy `[0, 2q)`/`[0, 4q)` ranges; inverse GS butterflies defer final reduction/scaling. See the [official repository](https://github.com/encryptorion-lab/phantom-fhe), [`ntt_1d.cu`](https://github.com/encryptorion-lab/phantom-fhe/blob/1f4a198443b3af77118e51f53d5b8f332154b875/src/ntt/ntt_1d.cu), [`fntt_2d.cu`](https://github.com/encryptorion-lab/phantom-fhe/blob/1f4a198443b3af77118e51f53d5b8f332154b875/src/ntt/fntt_2d.cu), and [`butterfly.cuh`](https://github.com/encryptorion-lab/phantom-fhe/blob/1f4a198443b3af77118e51f53d5b8f332154b875/include/butterfly.cuh).

The Phantom paper describes intra-kernel fusion that reuses register temporaries and inter-kernel fusion across operations such as NWT plus scaling and ModDown plus correction. Its key-switch path connects ModUp, evaluation-key inner product, and ModDown on a supplied stream. See [Phantom: A CUDA-Accelerated Word-Wise Homomorphic Encryption Library](https://eprint.iacr.org/2023/049) and [`eval_key_switch.cu`](https://github.com/encryptorion-lab/phantom-fhe/blob/1f4a198443b3af77118e51f53d5b8f332154b875/src/eval_key_switch.cu).

What to absorb:

- Use CT forward and GS inverse so compatible orderings can eliminate explicit permutations between NTT, pointwise multiplication, and INTT.
- Define and prove lazy ranges per butterfly. Reductions are removed only when the next consumer's bound remains safe.
- Prefer register reuse and boundary fusion over simply increasing shared-memory tile size.
- Expose the CUDA stream all the way through key switching.

### 2.5 TensorFHE: tensor-core NTT and operation-level batching

TensorFHE maps a decomposed 32-bit NTT to tensor-core matrix multiplications and emphasizes throughput rather than single-operation latency. It batches CKKS operations and changes layout from `(batch, limb, N)` to `(limb, batch, N)` so equal-limb data from different operations is contiguous. It also assigns independent matrix multiplications to separate streams. See the author paper, [TensorFHE](https://arxiv.org/abs/2212.14191), especially Sections IV-C and IV-D.

What to absorb:

- Operation-level batching and a layout selectable between `[limb][batch][N]` and `[batch][limb][N]` are worth measuring for inference serving.
- Throughput-oriented inference can benefit from batching several ciphertexts even if one ciphertext already batches all RNS towers.

**Inference/caution:** do not make tensor-core NTT the first implementation path. TensorFHE's arithmetic decomposition and throughput objective differ from OpenFHE's usual 50–60-bit RNS towers and this project's immediate exact-compatibility goal. Establish a fast integer-core Shoup/Barrett backend first; treat tensor cores as a separate experimental backend.

### 2.6 ICICLE: useful API ideas, not a direct CKKS backend

ICICLE's official NTT API exposes stream, batch size, row/column batch layout, input/output ordering, device-resident flags, async execution, reusable per-device domains, fast-twiddle mode, and an explicit mixed-radix selection. See the [official NTT API](https://github.com/ingonyama-zk/icicle/blob/main/docs/versioned_docs/version-4.0.0/api/cpp/ntt.md).

What to absorb:

- Separate an algorithm-independent NTT contract from backend-specific tuning extensions.
- Make ordering explicit in the type/configuration rather than relying on comments.
- Initialize and cache a maximum-size domain once, then reuse it for smaller transforms where mathematically compatible.
- Allow both row-batched and column-batched transforms and device-resident async operation.

**Inference/caution:** ICICLE is primarily a proof-system field library, not an OpenFHE CKKS/Hybrid implementation. Its API concepts transfer; its field choices and transform semantics must not be assumed compatible. Its CUDA backend also has source-availability/licensing conditions that should be checked before any reuse.

### 2.7 cuFHE/cuHE: historically relevant but not the integration model

cuFHE implements TFHE gates and credits cuHE's GPU NTT; its API supports CUDA streams and keeps a public bootstrapping key initialized on the GPU. See the [official cuFHE repository](https://github.com/vernamlab/cuFHE) and [official cuHE repository](https://github.com/vernamlab/cuHE).

What to absorb:

- Keep large evaluation/bootstrapping keys resident and permit concurrent independent operations through streams.

**Inference/caution:** TFHE gate bootstrapping has different polynomial sizes, dataflow, and key-switch structure from CKKS Hybrid RNS. cuFHE is not a suitable algorithmic baseline for the OpenFHE integration; it is mainly historical evidence for GPU-resident keys and stream-level concurrency.

### 2.8 Theodosian: RTX 5090 changes the optimization priority

Theodosian profiles CKKS on RTX 5090 and reports that optimized NTT and basis-conversion kernels are L2-bandwidth-bound. Its NTT uses two kernels; on-the-fly twiddle generation trades arithmetic for fewer twiddle loads. It finds small limb counts underutilize the GPU, uses L2-aware batching until the working set approaches cache capacity, applies multi-polynomial caching and complementary pipelining to Hybrid key-switch stages, and uses CUDA Graphs to reduce launch overhead for static FHE graphs. See [Theodosian](https://arxiv.org/abs/2512.18345).

The paper reports that batching can hurt after the footprint exceeds the 5090's effective L2 capacity, and that key switching benefits more from cache-aware multi-polynomial execution and graph scheduling than from additional local kernel fusion once the baseline is already heavily fused.

What to absorb:

- Profile L2 read/write throughput and SM-to-L2 traffic on the actual 5090, not only DRAM bandwidth and occupancy.
- Generate high-volume twiddles on the fly when the arithmetic cost is cheaper than their cache traffic; retain a precomputed mode for low-volume stages.
- Batch according to measured working-set size. “More batches” is not monotonic.
- Capture stable key-switch/rotation/inference sequences in CUDA Graphs.
- Pipeline stages with complementary DRAM/L2 behavior only after dependencies and cache interference are measured.

## 3. Algorithm and arithmetic choices for this project

### 3.1 CT/GS versus Stockham and Four-Step

**Sourced observation:** Phantom and FIDESlib use CT forward plus GS inverse; GPU-NTT compares iterative Merge and Four-Step schedules; ICICLE exposes mixed-radix and ordering choices.

**Recommendation:** keep CT/DIF forward and GS/DIT inverse as the first optimized path because the current repository already has the equivalent ordering contract and no bit-reversal pass. Add a Four-Step/transpose prototype only as a benchmarked alternative for the strided column pass. A Stockham autosort path may offer regular accesses and natural ordering, but it normally ping-pongs global buffers; for the target sizes, its extra full-array traffic must beat the cost of the current uncoalesced gather before adoption.

### 3.2 Mixed radix and register staging

**Sourced observation:** CKKS-GPU-CORE and Phantom use up to eight values per thread and radix-8 local transforms; GPU-NTT notes higher-radix butterflies as a further optimization.

**Recommendation:** implement radix-4 first, then radix-8:

1. Load 4/8 values per thread using coalesced vectorized transactions.
2. Execute the local stages in registers.
3. Store into padded shared memory for the permutation between local groups.
4. Use warp shuffles where partner lanes remain within a warp.
5. Store once at the end of the phase.

This directly reduces table loads, barriers, and shared-memory operations relative to one butterfly per thread per stage.

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

## 4. RNS data layout and Hybrid key-switch implications

### 4.1 Baseline layout

**Recommendation:** keep coefficient data limb-major:

```text
poly[component][digit-or-batch][limb][coefficient]
```

with `coefficient` contiguous. This gives coalesced coefficient loads for one limb and makes modulus-specific constants uniform across a block. Provide views/strides so operation-level batching can switch the middle dimensions without copying.

### 4.2 Extended-basis state

**Recommendation:** make these states explicit and non-interchangeable:

- `Q, coefficient`
- `Q, evaluation`
- `QP, coefficient`
- `QP, evaluation`

Each buffer view should carry active Q tower count, P tower count, level, ordering, device, and stream/event ownership. This prevents accidental truncation or redundant transforms at the OpenFHE boundary.

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

The exact transform placement and rounding order must be derived from the pinned OpenFHE 1.6.1 source and frozen in compatibility tests. Do not substitute an algebraically equivalent formulation if bit equality is the goal.

### 4.4 Fusion candidates

High-value, low-ambiguity experiments:

- INTT final scaling + decomposition pre-scale.
- Basis-extension output + first forward-NTT load/prologue.
- Forward-NTT final store + evaluation-key dot product when register and occupancy costs permit.
- INTT or NTT boundary + ModDown correction.
- Pointwise accumulation across Hybrid digits, deferring reduction only under a proved bound.

**Caution:** fusion can lower occupancy by increasing registers/shared memory and can prevent cache-friendly scheduling. Theodosian shows additional fusion is not automatically beneficial on RTX 5090. Require profiler evidence for every fused variant.

## 5. Streams, transfers, allocation, and launch strategy

NVIDIA documents that asynchronous host/device copies require pinned host memory and can overlap with kernels in other streams. It also documents global-to-shared asynchronous copies and pipelines, and states that CUDA Graphs reduce repeated launch setup cost for fixed workflows. See the [CUDA Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/#asynchronous-and-overlapping-transfers-with-computation), [asynchronous copy documentation](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/async-copies.html), and [CUDA Graphs documentation](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html).

Recommended API split:

- `NttContext`: immutable moduli, roots, packed twiddles, reduction constants, launch table, and allocator handle.
- `DeviceRnsView`: pointer(s), dimensions/strides, basis/domain/order metadata, device, and readiness event.
- `enqueue_forward/inverse(view, stream)`: no allocation, no synchronization, no host copy.
- convenience host methods: pinned staging plus explicit or documented synchronization.
- reusable scratch arena sized for maximum supported level/digit count.
- captured CUDA graphs for fixed ring dimension, level, digit count, and operation sequence.

**Recommendation/inference:** try `cuda::memcpy_async`/pipeline only after the coalesced phase layout is fixed. The current column gather has no useful sequential “next tile” per block to hide without first changing data ownership; async copy cannot repair uncoalesced addresses.

## 6. Correctness and benchmark methodology

### 6.1 Correctness ladder

1. Arithmetic primitive versus `__uint128_t % q`, including adversarial bounds and every OpenFHE modulus.
2. Butterfly and single-stage exact tests for CT/DIF and GS/DIT, including lazy-range invariants.
3. Complete forward/inverse exact equality against the existing CPU and Python oracle for every `N` in `{4096, 8192, 16384, 32768, 65536}`.
4. RNS batches across realistic CKKS tower counts, not only three limbs.
5. Negacyclic polynomial multiplication exact equality.
6. Basis conversion exact equality to the pinned OpenFHE 1.6.1 implementation.
7. Hybrid relinearization and rotation: exact ciphertext comparison where formulation and code generation permit; decrypt-error comparison as a secondary semantic test.
8. Long operation sequences and inference circuits, including level/scale/noise metadata and repeated rotations.

### 6.2 Required performance views

For each ring dimension, modulus bit-width, and tower count:

- kernel-only forward and inverse latency using CUDA events;
- end-to-end latency with pinned and pageable host transfer reported separately;
- transforms/second for batches of towers, ciphertext components, and independent ciphertexts;
- full polynomial multiply latency;
- Hybrid key switch, relinearization, and rotation latency;
- complete inference latency and peak device memory;
- bytes read/written, achieved DRAM bandwidth, L2 read/write throughput, global load/store efficiency, bank conflicts, register count, occupancy, and launch count from Nsight Compute/Systems;
- cold setup, warm steady state, and CUDA Graph replay as distinct measurements.

### 6.3 Baselines

- This repository's current pre-Barrett kernel, labelled clearly as a correctness baseline.
- Optimized kernel with identical transfers.
- GPU-NTT or another standalone GPU NTT with matched modulus/order/batch where license and semantics permit.
- FIDESlib 2.2/OpenFHE 1.6.1 for NTT microbenchmarks and, more importantly, full Hybrid key switch/rotation.
- Stock OpenFHE 1.6.1 CPU, with build flags and thread count recorded.

Do not compare one implementation's kernel-only time with another's transfer-inclusive time, nor one-limb latency with an entire RNS polynomial.

## 7. Prioritized points to absorb

### P0: required before meaningful performance work

- Add Shoup twiddle multiplication and a variable-product Barrett/Montgomery path.
- Add exact arithmetic gates for the maximum OpenFHE modulus width.
- Add a device-resident, stream-aware API.
- Benchmark the current column pass's memory-transaction efficiency on RTX 5090.
- Pin OpenFHE 1.6.1 and extract its exact Hybrid transform/order/basis contract into tests.

### P1: first competitive NTT

- Implement per-size compile-time dispatch for `2^12..2^16`.
- Implement radix-4/8 register staging with padded shared memory.
- Pack stage-major twiddle and Shoup tables.
- Prototype a coalesced transpose/Four-Step alternative to the strided column pass.
- Benchmark latency and RNS/ciphertext throughput; choose by size and batch, not one universal kernel.

### P2: base conversion and Hybrid key switching

- Freeze the limb-major `Q/QP` device layout and explicit domain/basis metadata.
- Implement basis conversion with output-limb × coefficient parallelism and bounded wide accumulators.
- Fuse selected NTT/basis/key-dot boundaries.
- Keep evaluation keys and scratch buffers resident.
- Add CUDA Graph capture for relinearization and rotation.

### P3: inference-oriented optimization

- Add operation-level batching with an L2-capacity guard.
- Hoist decomposition for groups of rotations.
- Reuse extended-basis results across compatible operations.
- Profile complementary stream pipelines and whole-model memory traffic.
- Consider 32-bit/tensor-core research only as an optional backend because it changes OpenFHE parameter representation and compatibility assumptions.

## 8. Licensing and clean-room implementation

This section is informational, not legal advice.

- **FIDESlib:** MIT license in its official repository. Its OpenFHE interoperability currently depends on patches, which should be reviewed separately for upstream/license provenance.
- **GPU-NTT:** Apache-2.0.
- **HEonGPU:** check the repository's current license before copying code; concepts described by the paper can be independently implemented.
- **PhantomFHE:** GPLv3. Copying its source into a differently licensed project can impose copyleft obligations. Use the paper and behavior as design evidence unless the project intentionally adopts compatible licensing.
- **CKKS-GPU-CORE:** CC BY-NC-SA 4.0 in the repository. The non-commercial/share-alike terms make direct code reuse unsuitable for many projects. Independently implement paper-level ideas and preserve clean-room provenance.
- **ICICLE CUDA backend:** source availability and the backend license have changed across releases. Verify the exact version's terms before any reuse.
- **cuFHE/cuHE:** their official repositories state their licenses; verify individual files before reuse.

**Recommendation:** continue the repository's clean-room policy. Cite papers and inspect behavior/interfaces, but write original kernels. If code is adapted from MIT/Apache sources, preserve required notices and record the exact file/commit. Do not copy GPL or CC BY-NC-SA kernel code unless the project's license decision explicitly permits it.

## 9. Decision summary for the next code update

The survey supports the following immediate implementation sequence:

1. Introduce arithmetic policies and implement `ShoupTwiddle` plus `BarrettVariable`, with exact tests.
2. Add a device-buffer/stream API without removing the current host-vector facade.
3. Instrument and benchmark the existing two-pass kernel on RTX 5090 using CUDA events and Nsight.
4. Implement one radix-8, padded-shared-memory variant and one coalesced transpose/Four-Step variant.
5. Select per-size/per-batch winners through a dispatch table.
6. Build the `Q/QP` state/layout and OpenFHE 1.6.1 compatibility harness before implementing base conversion.
7. Optimize and report the full Hybrid key-switch graph, with FIDESlib as the central GPU baseline.

Benchmarking and autotuning across GPU parameters and implementation variants
are a permanent project workstream, not a one-time comparison. The concrete
matrix, required counters, reproducibility record, dispatch rules, and
acceptance gates are defined in
[`benchmark-tuning-plan.md`](benchmark-tuning-plan.md).

The HTML explainer should then be updated to show both the current correctness schedule and the selected production schedule. It should no longer imply that low nominal global byte count alone means efficient VRAM access; it should visualize coalescing, register-held radix-8 work, padded shared memory, and the complete `Q -> QP -> Q` Hybrid dataflow.

## Primary sources

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
