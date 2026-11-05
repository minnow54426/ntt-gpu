# GPU NTT benchmark and tuning plan

**Primary target:** NVIDIA RTX 5090 (`sm_120`)  
**Cross-architecture control:** NVIDIA RTX 4090 (`sm_89`)  
**Workload target:** OpenFHE 1.6.1 CKKS with Hybrid key switching  
**Ring dimensions:** `4096`, `8192`, `16384`, `32768`, `65536`

Benchmarking and tuning are deliverables of the project. A kernel is not
considered complete merely because it is correct or fast for one transform
size. Selection must account for the GPU, CKKS parameters, batch shape, and
whether the objective is latency or throughput.

## 1. Implementations to compare

Each implementation must preserve the same transform convention and pass the
same exact correctness gates.

1. Current radix-2 two-pass implementation using reference modular arithmetic.
2. Two-pass implementation with Shoup twiddle multiplication and Barrett or
   Montgomery variable multiplication.
3. Radix-4 register-staged implementation.
4. Radix-8 register-staged implementation with padded shared memory.
5. Coalesced Four-Step/transpose implementation.
6. Fused variants used by base conversion and Hybrid key switching.

External libraries such as GPU-NTT and FIDESlib are comparison baselines when
their transform semantics, modulus limits, and timing boundaries can be matched.
Their numbers must not be compared directly when one result includes transfers
or an entire RNS polynomial and the other does not.

## 2. Workload matrix

The minimum matrix is:

| Axis | Values |
|---|---|
| Ring dimension | `2^12`, `2^13`, `2^14`, `2^15`, `2^16` |
| Modulus width | OpenFHE-compatible 50, 55, 59/60-bit primes |
| Active RNS towers | representative shallow, medium, and deep CKKS levels |
| Polynomial count | one component, full ciphertext, key-switch digits |
| Batch size | `1`, `2`, `4`, `8`, then L2-capacity-limited sweep |
| Direction | forward, inverse, forward–pointwise–inverse |
| Objective | single-operation latency and steady-state throughput |
| Transfer policy | device-resident, pinned end-to-end, pageable end-to-end |
| GPU | RTX 5090 primary; RTX 4090 control |

The exact tower counts and Hybrid digit counts must come from the pinned
OpenFHE parameter sets used by the inference examples, rather than synthetic
three-limb inputs alone.

## 3. GPU and kernel parameters to tune

Tune or record all of the following:

- threads per block and blocks per transform;
- values and butterflies per thread;
- radix and number of stages fused per kernel;
- tile/split dimensions and transpose shape;
- register count per thread and register spilling;
- static/dynamic shared memory per block and bank conflicts;
- active blocks and warps per SM;
- warp-shuffle versus shared-memory exchange boundaries;
- twiddle layout, Shoup table layout, and on-the-fly twiddle generation;
- lazy-reduction range and reduction frequency;
- RNS limb, ciphertext, and operation batch layout;
- CUDA stream count, asynchronous copies, and CUDA Graph replay;
- scratch-buffer reuse and allocator behavior.

Compiler resource reports (`ptxas -v`) and measured Nsight occupancy must both
be retained. A lower register count is not automatically better: the decision
criterion is end-to-end performance without spills or unacceptable occupancy.

## 4. Measurements

### Timing

- CUDA-event kernel-only latency for every kernel and complete transform;
- transfer-inclusive latency, separating pinned and pageable host memory;
- transforms per second and coefficients/towers processed per second;
- full polynomial multiplication latency;
- Hybrid key switch, relinearization, rotation, and hoisted-rotation latency;
- complete encrypted-inference latency and peak device memory;
- cold setup, warm steady state, and CUDA Graph replay separately.

Report median, p5, and p95 after warm-up. Store raw samples so later summaries
can be regenerated without rerunning the GPU.

### Hardware counters

- achieved DRAM and L2 bandwidth;
- global load/store transaction efficiency;
- L2 hit rate and read/write sectors;
- shared-memory bank conflicts;
- registers per thread, spills, occupancy, and eligible warps;
- SM and memory-pipeline utilization;
- kernel-launch count and synchronization gaps.

The current column pass must specifically record memory sectors per requested
byte because its neighboring-thread coefficient addresses are 2,048 bytes
apart.

## 5. Reproducibility record

Every result row must include:

- GPU name, compute capability, clocks, power mode, and memory size;
- driver, CUDA toolkit/runtime, compiler, CMake flags, and git commit;
- implementation/variant ID and all tuned launch parameters;
- `N`, modulus bits, actual moduli or parameter-set ID, tower count, digit count,
  batch shape, transform ordering, and transfer policy;
- correctness-gate result and profiler command/version;
- whether setup, allocation, transfers, and synchronization are included.

Results without this header are exploratory and must not appear in published
comparison tables.

## 6. Tuning procedure

1. Run exact arithmetic and transform gates for the candidate variant.
2. Reject variants that spill unexpectedly, exceed resource limits, or violate
   lazy-reduction bounds.
3. Sweep launch and staging parameters for each supported ring dimension.
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
- A selected implementation exists for all five ring dimensions on RTX 5090.
- RTX 4090 results either validate the same dispatch or document a separate one.
- Full Hybrid key-switch and inference results accompany NTT microbenchmarks.
- The HTML explainer identifies which schedule is the correctness baseline and
  which implementation the tuner selected for each workload class.
