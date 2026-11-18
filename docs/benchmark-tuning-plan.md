# GPU NTT benchmark and tuning plan

**Primary target:** NVIDIA RTX 5090 (`sm_120`)  
**Cross-architecture control:** NVIDIA RTX 4090 (`sm_89`)  
**Workload target:** OpenFHE 1.6.1 CKKS with Hybrid key switching  
**Primary ring dimension:** `65536`; regression/benchmark sweep: `4096`–`65536`

A kernel is a function executed by many GPU threads. Choose between kernel
versions by checking their answers and measuring them at N=65536. Latency
means time for one operation; throughput means operations completed per
second. Record the GPU, encryption parameters, number of polynomials and
whether CPU–GPU transfers are included.

The [full-matrix report](batch40-native-analysis.md) records the temporary Ada
baseline. The [earlier diagnostic analysis](first-native-analysis.md) explains
why its single averaged row was insufficient. Neither supplies an RTX 5090
result or an accepted hardware-counter bottleneck diagnosis.

## 1. Implementations to compare

Every version must produce the same array values in the same order and pass
the same exact-answer tests. A “gate” below means a check that must pass
before the next step. Some proposed optimizations are named here before
they are implemented:

1. Current radix-2 two-pass implementation using reference modular arithmetic.
2. Two-pass implementation with Shoup twiddle multiplication and Barrett or
   Montgomery variable multiplication.
3. Radix-4 register-staged implementation.
4. Radix-8 register-staged implementation with padded shared memory.
5. Coalesced Four-Step/transpose implementation.
6. Fused variants used by base conversion and Hybrid key switching.

Compare with the GPU-NTT and FIDESlib libraries only when they compute the
same transform, accept the same prime sizes, process the same number of
prime towers, and include the same work in their timings.

## 2. Workload matrix

The minimum matrix is:

| Axis | Values |
|---|---|
| Ring dimension | `4096`, `8192`, `16384`, `32768`, `65536`; optimize primarily for `65536` |
| Modulus width | OpenFHE-compatible **30, 50, 55, 60-bit** primes; retain actual production 59-bit fixtures too |
| Active RNS towers | representative shallow, medium, and deep CKKS levels |
| Polynomial count | one component, full ciphertext, key-switch digits |
| Batch size | `1`, `2`, `4`, `8`, then L2-capacity-limited sweep |
| Direction | forward, inverse, forward–pointwise–inverse |
| Objective | single-operation latency and steady-state throughput |
| Transfer policy | device-resident, pinned end-to-end, pageable end-to-end |
| GPU | RTX 5090 primary; RTX 4090 control |

Take tower and Hybrid digit counts from the pinned OpenFHE parameter sets:
[`PrecomputeCRTTables`](https://github.com/openfheorg/openfhe-development/blob/b59e1f291965f14fbee5172cd83a20f5b856204f/src/pke/lib/schemerns/rns-cryptoparameters.cpp#L88-L134) defines the full-context partitions,
and [`EvalKeySwitchPrecomputeCore`](https://github.com/openfheorg/openfhe-development/blob/b59e1f291965f14fbee5172cd83a20f5b856204f/src/pke/lib/keyswitch/keyswitch-hybrid.cpp#L300-L366) derives the active-level count.
Three-limb synthetic fixtures alone are insufficient.

The next rental's bounded baseline batch uses all five dimensions, all four
prime widths, and K=1/3/8/16/24/32/40: **140 workloads**, with separate public
GPU, resident event/pinned-copy, and OpenFHE CPU jobs (**420 combined jobs**).
The rental helper defaults to 280 GPU-only jobs; collect the 140 CPU jobs
locally with the split collector. Select any K from 1 to 40; K counts primes
per polynomial, not ciphertexts. Defaults are five warmups and 100 samples.

This baseline does not sweep production CKKS parameter sets, ciphertext
batches or alternative implementations. Thirty-bit primes use the same
`uint64_t` storage/arithmetic; there is no specialized `uint32_t` kernel or
four-byte transfer path yet.
See the [batch protocol](offline-test-tooling.md) for restart and timing limits.

## 3. GPU and kernel parameters to tune

Group experiments by the cost they change:

| Area | Variables to test or record |
|---|---|
| Thread work | Block size, butterflies/values per thread, radix, register staging and spills |
| Memory | Split/transpose shape, shared-memory padding, bank conflicts, warp exchange and coalescing |
| Arithmetic | Shoup/Barrett policy, lazy range, reduction frequency and twiddle representation |
| Workload | RNS tower layout, ciphertext/digit batch layout and active blocks/warps |
| Runtime | Streams, allocation/scratch reuse, asynchronous copies and CUDA Graph replay |

Change one major factor at a time before combining winners.

Save `ptxas -v` compiler reports and the Nsight occupancy measurement.
Occupancy describes how many warps can remain active on a GPU processor.
A lower register count alone does not prove a faster program; measure the
complete operation.

## 4. Measurements

### Timing

- CUDA-event kernel-only latency for every kernel and complete transform;
- transfer-inclusive latency, separating pinned and pageable host memory;
- transforms per second and coefficients/towers processed per second;
- full polynomial multiplication latency;
- Hybrid key switch, relinearization, rotation, and hoisted-rotation latency;
- complete encrypted-inference latency and peak device memory;
- cold setup, warm steady state, and CUDA Graph replay separately.

Run warmups before collecting samples. Report the median (middle sample),
p5 (the fifth percentile) and p95 (the ninety-fifth percentile) to show both
typical time and variation. Save the individual samples so summaries can be
recomputed without another rental.

### Hardware counters

- achieved DRAM and L2 bandwidth;
- global load/store transaction efficiency;
- L2 hit rate and read/write sectors;
- shared-memory bank conflicts;
- registers per thread, spills, occupancy, and eligible warps;
- SM and memory-pipeline utilization;
- kernel-launch count and synchronization gaps.

In the current column pass, neighboring threads request coefficients 2,048
bytes apart. The GPU may fetch much more memory than the useful values.
Record the number of memory sectors fetched for each requested byte to
measure that overhead.

## 5. Reproducibility record

Every result row must include:

- GPU name, compute capability, clocks, power mode, and memory size;
- driver, CUDA toolkit/runtime, compiler, CMake flags, and git commit;
- implementation/variant ID and all tuned launch parameters;
- `N`, modulus bits, actual moduli or parameter-set ID, tower count, digit count,
  batch shape, transform ordering, and transfer policy;
- correctness-gate result and profiler command/version;
- whether setup, allocation, transfers, and synchronization are included.

Without these details, a result cannot support a reproducible comparison.
Keep it as an exploratory observation rather than putting it in the published
comparison tables.

## 6. Tuning procedure

1. Run exact arithmetic and transform gates for the candidate variant.
2. Reject variants that spill unexpectedly, exceed resource limits, or violate
   lazy-reduction bounds.
3. Sweep launch and staging parameters at N=65536.
4. Profile the best latency and throughput candidates with Nsight Compute.
5. Repeat using realistic RNS tower and Hybrid digit batches.
6. Validate finalists on RTX 4090 to identify architecture-specific choices.
7. Record the winner in a dispatch table keyed by GPU capability, ring
   dimension, tower/batch range, and latency/throughput mode.
8. Re-run the complete Hybrid key-switch and inference benchmarks. A locally
   faster NTT variant is accepted only if it improves the target pipeline or
   supplies a clearly justified fallback.

## 7. Acceptance gates

- Exact forward/inverse agreement for every supported `N` and OpenFHE modulus.
- No silent fallback to device-side integer remainder in optimized kernels.
- No benchmark mixes kernel-only and end-to-end timing boundaries.
- A selected latency and throughput implementation exists for N=65536 on RTX 5090.
- RTX 4090 results either validate the same dispatch or document a separate one.
- Full Hybrid key-switch and inference results accompany NTT microbenchmarks.
- The HTML explainer identifies which schedule is the correctness baseline and
  which implementation the tuner selected for each workload class.

### Compare the two butterfly schedules

Build the default `group_twiddle_ct` and a separate
`-DNTT_CUDA_LEGACY_BUTTERFLY=ON` build on the same GPU. Run exact checks and
sanitizers first, then run `events-jsonl` on matched N, roots, primes and
operands. Compare resident forward and inverse medians before comparing public
calls; keep both builds' raw data and `transform_schedule` labels. The pass
split and reduction are unchanged, but arithmetic and twiddle access differ.
Prior Ada timings used the old schedule; no speedup is assumed.
