# CUDA kernel performance optimization

Start with a timeline to find where the application spends time, then profile the
kernel that dominates that workload. For this project's CUDA number-theoretic
transform (NTT), the first two kernel experiments are **fixed-twiddle modular
multiplication** and **coalesced access in the strided pass**. The source shows
both opportunities. Hardware counters will determine which matters more on
the RTX 5090.

## Start with the right timing

<div class="site-table-scroll" role="region" aria-label="Benchmark timing scopes" tabindex="0" markdown="1">

| Timing scope | What to measure | What it helps decide |
| --- | --- | --- |
| Individual kernel | Time one strided pass, contiguous pass, normalization or pointwise product on the device | Which kernel and resource to optimize |
| GPU operation | CUDA-event interval around the transform or multiplication sequence, using resident data | Whether the change pays for any added pass, copy or gap on the stream |
| Host-visible operation | Wall time through completion, with allocation, orchestration and input/output transfers where applicable | Whether the caller sees the gain |
| Steady-state throughput | Transforms per second at a stated batch size and tower count | Whether batching improves utilization without hurting the target latency |

</div>

At fixed ring dimension and prime count, compare the same variants in alternating
order. Warm up, save individual timings and report the median and p95. Record the
GPU, driver, CUDA version, compiler flags, clocks, power limit, input seeds, primes,
and the timing boundary. Keep profiler runs separate from uninstrumented timings.
A CUDA-event interval can include idle gaps between kernels submitted to its stream.
Use the timeline to see those gaps. [CUDA Best Practices Guide][cuda] · [Nsight Systems User Guide][nsys]

On the November 19 RTX 4080 SUPER run, the median legacy/group-twiddle ratio
across eight workloads was **1.52×** for GPU-event forward NTT and **1.01×**
for the public forward call. Resident-interface improvements were smaller than
the standalone event gains. Nsight Compute could not collect hardware counters
on that host. The RTX 5090 still needs native measurements. [Paired Ada NTT results](paired-ntt-native-analysis.md)

## Diagnose the limit before tuning

<div class="site-table-scroll" role="region" aria-label="Profiling signals and experiments" tabindex="0" markdown="1">

| Signal to look for | Experiment | What would support it |
| --- | --- | --- |
| Nsight Systems shows transfers, allocation or idle gaps around short kernels | Reuse scratch and parameter buffers. Keep operands resident. Try CUDA Graph replay for a stable sequence with repeated launch gaps | Lower wall time at the relevant boundary. Time graph creation and warm replay separately |
| Nsight Compute shows excess global sectors per request or high DRAM/L2 throughput | Map adjacent lanes to adjacent coefficients. Compare tiled multi-column loading with a transpose-based layout | Lower sector traffic and time for the full transform, including any transpose |
| Shared-memory accesses show excess bank conflicts or barriers dominate when schedulers struggle to issue work | Change shared-memory layout or padding. Try warp shuffles for warp-local exchanges or register-staged radix-4 | Fewer excess shared-memory transactions or barriers, with a net timing gain after register and spill costs |
| Few blocks per SM or too few eligible warps to issue instructions | Sweep block size and butterflies per thread. Batch independent transforms when latency allows | More useful work per cycle and lower latency or higher throughput, with the batch size stated |
| Integer issue or dependency chains dominate while memory pipelines have headroom | Specialize fixed-operand reduction, tune instruction-level parallelism, and inspect generated machine code | Shorter kernel time and a corresponding drop in expensive instructions or dependency pressure |

</div>

Begin with **Launch Statistics**, **Occupancy**, **Scheduler Statistics** and
**Memory Workload Analysis**. Check eligible warps and issue activity before
interpreting stall counters. A high L2 hit rate can coexist with saturated L2
bandwidth, and a lightly loaded DRAM interface can coexist with expensive
global requests. Measure registers and spills with `-Xptxas=-v` and read the
compiled machine instructions (SASS) with `cuobjdump --dump-sass`. The usual
floating-point roofline does not characterize this exact integer workload. For
a first comparison, plot butterflies per second against measured L2 and DRAM
traffic, or compare to a matched modular-multiply microbenchmark. [Nsight Compute Profiling Guide][ncu] · [CUDA Binary Utilities][sass] · [Roofline paper][roofline]

## What the current NTT gives us to test

### Make modular multiplication cheaper

The group-twiddle butterfly uses one modular multiplication per pair. That
product still takes the generic `__uint128_t` remainder path. Twiddles and
the inverse normalization factor are known at setup, so the first arithmetic
experiment is to use the existing, separately tested Shoup multiply with
precomputed constants. Compare the extra table footprint and traffic against
the instruction savings. A full 64-bit quotient alongside every twiddle
would roughly double table storage and add about 20 MiB **per direction**
at N=65,536 with 40 prime towers. Benchmark Barrett reduction separately for
pointwise products, where both operands vary. Keep the first A/B comparison
on canonical residues before testing lazy reduction between stages. [Butterfly code](../include/ntt/transform_stage.hpp) · [Shoup helper](../include/ntt/shoup.hpp) · [Reference arithmetic](../include/ntt/mod_arith.hpp)

### Coalesce the strided pass

The two-pass schedule groups nearby butterflies in shared memory. In the
strided pass, adjacent lanes load coefficients **2,048 bytes apart** because
the column stride is 256 eight-byte values. For one aligned, fully active
warp instruction, a contiguous load of 32 values addresses eight 32-byte
sectors. The strided load addresses 32 sectors for the same 256 useful bytes.
This is an address calculation. Cache reuse between blocks and write behavior
determine how much traffic reaches DRAM. A block that loads several adjacent
columns cooperatively can improve global access, but the shared-memory layout
also needs to serve the butterfly accesses. Record sector counts and conflict
metrics for loads and stores, then time the complete transform. [CUDA kernels](../src/cuda/ntt_kernels.cu) · [CUDA Best Practices Guide][cuda]

### Match the launch shape to the amount of work

The current launch uses 256 threads per block. At N=4,096 the far group has
16 coefficients and eight butterflies per stage. In the contiguous pass
there are only 16 blocks per tower, or 48 blocks at three towers. At
N=65,536, both passes have 256 blocks per tower and 128 butterflies per block
per stage. Sweep 64, 128 and 256 threads per block as separate launch
parameters for each pass. Measure block count and useful lane work together.
A smaller block alone will leave the contiguous pass at 48 blocks in the
small case unless the tile or batch mapping also changes.

The RTX 5090 has 170 SMs, 96 MiB of L2 and a quoted DRAM bandwidth of
1,792 GB/s. Compute capability 12.0 permits up to 48 resident warps and
65,536 32-bit registers per SM. A 256-thread block occupies eight warps,
giving a ceiling of six such blocks per SM before register, shared-memory
and other limits. A launch with many active warps can still lose to one that
does more useful work per warp. Start with the existing shape and let timings
and resource measurements guide changes. [RTX Blackwell architecture whitepaper][5090] · [Blackwell Tuning Guide][blackwell] · [Compute capability list][cc]

### Fuse where the data is already local

The inverse transform launches a separate normalization kernel. Applying
the inverse-length factor as the final pass stores its results would remove
one launch and one read/write of the coefficient array. At N=65,536 and
40 towers this is **40 MiB of logical array traffic** per inverse, before
cache effects. The scale arithmetic remains, so measure register use and
inverse time in both versions. Try asynchronous global-to-shared copies once
there is a coalesced access pattern and useful independent work to overlap.
Double buffering raises shared-memory usage and can reduce residency.
[CUDA kernels](../src/cuda/ntt_kernels.cu) · [CUDA asynchronous copies][async]

## What published results establish

- **Kim et al.** identified a main-memory bandwidth problem at large
  homomorphic-encryption parameter sets. Their full optimization stack,
  including on-the-fly twiddle generation, achieved 4.2× over their radix-2
  baseline. Recompute or compress twiddles if measured table traffic dominates
  after arithmetic changes. [Kim et al., Accelerating NTTs on GPUs][kim]
- **Özcan, Javeed and Savaş** evaluate radix-2 Cooley–Tukey and Four-Step GPU
  transforms. A Four-Step candidate is worth comparing if tiled loads alone
  leave high sector costs. Include the permutation and any conversion to
  OpenFHE's bit-reversed evaluation layout in the timing. [IEEE Access, 2025][ozcan]
- **Theodosian** found SM-to-L2 write bandwidth constrained its optimized
  NTT and basis conversion on RTX 5090. It also found low tower counts left
  hardware underutilized and used cache-aware batching and CUDA Graphs at
  workload level. Test the batch sweep on this implementation. Its current
  64-bit storage and generic modular product create a different balance of
  compute and traffic. [Choi et al., Theodosian, §§III–V][theodosian] · [NVIDIA CUDA Graph benchmark][graphs]

## First RTX 5090 session

1. **Capture the baseline.** Reuse the [rental queue and profiling commands](next-gpu-rental.md#capture-the-resident-timeline-and-export-evidence).
   Begin with single-polynomial N=4,096 and N=65,536, K=3 and K=40 towers,
   and 30- and 60-bit primes. Keep the saved input vectors. Verify counter
   access, capture representative strided and contiguous kernels by name,
   and save a separate timeline so it survives any counter-access failure.
2. **Run independent A/B tests.** Compare Shoup twiddles and coalesced strided
   loading independently against the group-twiddle baseline. Use inverse
   scale fusion if the standalone pass is significant. Let the measured
   bottleneck guide which experiment runs first. Follow a promising result
   with a block-size sweep, then a batch sweep across cache footprints.
3. **Check the whole operation.** Require exact OpenFHE 1.6.1 values, inverse
   normalization, ordering and bounds at the public interface. Run native
   exactness and representative Compute Sanitizer checks. Compare changes
   at the transform and resident polynomial-operation boundaries, then at
   the complete Hybrid key-switch boundary once available. Use the full
   dimension and prime-width sweep for finalists. [OpenFHE representation contract](openfhe-representation-contract.md) · [Benchmark and tuning plan](benchmark-tuning-plan.md)

[cuda]: https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html
[nsys]: https://docs.nvidia.com/nsight-systems/UserGuide/index.html
[ncu]: https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html
[sass]: https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html
[roofline]: https://zenodo.org/records/1236156
[5090]: https://images.nvidia.com/aem-dam/Solutions/geforce/blackwell/nvidia-rtx-blackwell-gpu-architecture.pdf
[blackwell]: https://docs.nvidia.com/cuda/blackwell-tuning-guide/
[cc]: https://developer.nvidia.com/cuda/gpus
[async]: https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/advanced-kernel-programming.html
[kim]: https://arxiv.org/abs/2012.01968
[ozcan]: https://research.sabanciuniv.edu/id/eprint/51889/
[theodosian]: https://arxiv.org/html/2512.18345v1
[graphs]: https://developer.nvidia.com/blog/constant-time-launch-for-straight-line-cuda-graphs-and-other-performance-enhancements/
