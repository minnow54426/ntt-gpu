# Full-matrix GPU run: analysis

**2026-11-14 · temporary reported RTX 4080 SUPER / sm_89 · CPU: local M1 Pro**

The implementation passes the executed exactness and sanitizer gates. Its
largest performance opportunity is currently the boundary around GPU work:
keeping intermediate arrays resident already provides a substantial benefit.
This is a measured baseline, not a finished OpenFHE accelerator or an RTX 5090
result. No new rental is needed to analyze or prepare the next experiments.

## Evidence and measurement boundaries

The verified local evidence is under:

```text
build/ntt-evidence-20261114-ada-batch40/
  record/                     original mixed collector, cancelled
  gpu-only/                   complete fresh GPU-only collection
  profiling/selected-stats.log Nsight Systems CUDA kernel/copy summary
build/cpu-m1-pro-20261114/      separate complete CPU collection
build/analyze-split-session.py local analysis helper
```

Archive SHA-256:
`d7cdbe24ea9e87270c248011408d983f8760ef451dc72abba2be87ccdca5a1b6`.
The prior download check verified 1,087 per-file hashes. This analysis again
checks each fresh GPU/CPU log against its manifest hash, recomputes its
statistics with [benchmark_report.py](../scripts/benchmark_report.py), and
requires identical N/q/roots/seed and repetition counts across platforms/scopes.

| Collection | Accepted scope |
|---|---|
| Original mixed batch | 366 retained completed jobs; cancelled with exit 143, **not** a full successful 420-job batch |
| Fresh GPU-only batch | 140 workloads × two scopes = **280 passing jobs**, 154,000 timed samples |
| Local CPU baseline | 140 passing jobs × four operations = **56,000 timed samples** |
| Matrix | N=4096/8192/16384/32768/65536; K=1/3/8/16/24/32/40; primes=30/50/55/60 bits |
| Repetitions | Five warmups and 100 samples per operation; fixed seeded inputs, reset outside timing |

K is the number of RNS prime towers for one polynomial; it is **not** a batch
of independent ciphertexts. No independent-ciphertext batch sweep was measured.

Fresh GPU results use base commit `bbfa6db` plus an archived benchmark-only
source patch (SHA-256
`f844a963391c9d8ae9bc753bed9ec05a8d67175bd05e3321291e5532217764c1`).
The changed host benchmark skips repeated scalar CPU timing; device kernels
are unchanged. Its binary hash is recorded in the fresh manifest. Do not merge
old and new timing series as if they had identical binary provenance.

Hardware reports compute capability 8.9 and approximately 32 GiB under the
RTX 4080 SUPER name. That capacity is unusual for this model; the report is
not independent verification of the physical card or its retail configuration.
CUDA is 12.8.93. Nothing here measures the primary RTX 5090/sm_120 target.

## 1. Correctness: a strong baseline, not complete FHE

The saved native run passes **29/29 CTests**, including six GPU-labelled tests.
The original OpenFHE comparison matrix records **169,279,488** residue
comparisons; the expanded 140-cell × seven-pattern matrix records
**3,967,746,048**. These are comparison counts, not billions of independent
random test cases. Independent inverse checks avoid relying only on a forward/
inverse roundtrip.

All eight sanitizer executions pass: memcheck, racecheck, initcheck and
synccheck, each on legacy plus expanded matrix smoke. Both racechecks report
zero warnings/hazards. Expanded smoke covers N=4096/65536, 30/60-bit primes,
K=3/40 and zero/dense inputs: 107,790,336 comparisons per tool. It does not
cover every possible input, concurrent context, allocation failure or stream
ownership scenario. Setup-fault injection covers the existing ten pre-call
injection points, not a guarantee about every real driver/OOM failure.

The fresh benchmark also validates its GPU outputs against the reference.
The conclusion is exactness for the executed NTT/product cases against the
pinned OpenFHE 1.6.1 path, **not** completed GPU base conversion, Hybrid key
switching or CKKS inference. See the [roadmap](openfhe-integration-plan.md).

## 2. Public latency versus resident execution

All following numbers are **medians in milliseconds**, not minima or means.
Public calls include host residue validation, guards, pageable transfers and
blocking output copies. Public multiply also includes its by-value host copy/
output allocation. Setup/table construction is excluded.

Resident CUDA-event experiments exclude preparation/upload and result checking/
download. Multi-launch intervals still include submission gaps. The two scopes
are separate experiments: their ratio is **not a measured production speedup**,
and their difference cannot be labelled pure PCIe overhead.

### Largest workload: N=65536, K=40, 60-bit primes

| Operation | GPU public | GPU resident event | M1 Pro OpenFHE CPU |
|---|---:|---:|---:|
| Forward | 8.079 | 0.552 | 21.071 |
| Inverse | 8.045 | 0.598 | 21.827 |
| Pointwise | 11.858 | 0.0478 | 3.115 |
| Complete fused ring product | 12.431 | 1.723 | 65.687 |
| Complete unfused ring product | 33.412 | Not separately measured | Not separately measured |

The public/resident forward ratio is 14.63; the product ratio is 7.22.
Pointwise's approximately 248× scope gap is especially clear evidence that
public latency must not be interpreted as arithmetic-kernel time.
The source validates **both** pointwise input arrays and transfers two arrays
in and one out. Compare [wrapper calls](../src/cuda/ntt_kernels.cu) and
[host residue validation](../include/ntt/cuda_validation.hpp).

Across the 140 equally weighted workload cells, the median public/resident
ratio is **9.20× for forward** and **5.67× for the ring product**. These are
medians of per-cell ratios, not application-weighted aggregate accelerations.

### Representative complete-product latencies

| N | K | Prime bits | Public fused | Resident fused | Public unfused | M1 CPU product |
|---:|---:|---:|---:|---:|---:|---:|
| 4096 | 1 | 30 | 0.0869 | 0.0451 | 0.1628 | 0.0664 |
| 4096 | 1 | 60 | 0.0847 | 0.0410 | 0.1537 | 0.0676 |
| 4096 | 40 | 30 | 1.280 | 0.335 | 2.834 | 2.783 |
| 4096 | 40 | 60 | 1.288 | 0.295 | 3.118 | 2.823 |
| 65536 | 3 | 30 | 1.149 | 0.192 | 2.924 | 4.409 |
| 65536 | 3 | 60 | 1.134 | 0.163 | 2.958 | 5.050 |
| 65536 | 40 | 30 | 14.659 | 1.986 | 27.772 | 56.943 |
| 65536 | 40 | 60 | 12.431 | 1.723 | 33.412 | 65.687 |

## 3. Fusion is already a measured improvement

The complete unfused and fused chains are both directly timed under the same
public-call protocol. Fused is faster in **all 140 workloads**. The per-cell
unfused/fused median-latency ratio ranges from **1.77× to 3.03×**, with a
median of **2.36×**. At N=65536/K=40/60 bits it is **2.69×**.

This is transfer/API fusion, not butterfly-kernel fusion. Both chains launch
eight kernels: two per forward, one pointwise, three for inverse including
normalization. For one polynomial buffer of size S, the wrapper reduces operand
transfer payload from **9S to 3S** and reduces repeated validations/guards.
Source accounting is explained in [CUDA dataflow](cuda-buffer-dataflow.md).

Consequently, preserve residency when designing base conversion and Hybrid
key switching. Returning to the host after every NTT would discard an already
demonstrated benefit. This does not predict a 2.36× Hybrid or inference gain.

## 4. Transfer scale, kernel trace and remaining bottleneck uncertainty

At N=65536/K=40 one u64 array is **20 MiB**. The separately measured pinned
copy events at 60 bits are:

| Direction | Median | Effective payload rate |
|---|---:|---:|
| H2D | 1.614 ms | 12.99 GB/s |
| D2H | 1.558 ms | 13.46 GB/s |

Rates use payload bytes / event time, decimal GB/s, not measured DRAM bandwidth
or a claim about the PCIe link generation. The smaller 4096/K=1 copies achieve
only about 5.4 GB/s because per-copy overhead matters. Public calls use pageable
host vectors; these pinned-copy times are not an additive decomposition of
their timings. In particular, 1.723 ms resident product time is **not** a
promised end-to-end time for an input that still lives on the host.

The selected Nsight Systems trace uses N=65536/K=3/60 bits, one warmup and two
samples. It is a different workload and profiler-perturbed observation, not a
replacement timing series. Of its approximately 0.845 ms total kernel time:

- Forward strided/contiguous kernels account for 29.5% / 26.7%.
- Inverse strided/contiguous kernels account for 20.2% / 18.0%.
- Pointwise and inverse scaling account for about 2.8% each.

These totals aggregate different operations and invocation counts. They are
not percentages of one ring product. Total copy time in this trace is about
5.349 ms, but includes event reset/upload and validation/download that are
excluded from resident timing. It must not be used as a production bottleneck
percentage.

Compile-time resources are 40 registers for forward-strided, 38 for the other
three transform kernels, and 24 for pointwise/scale, with **zero spill loads/
stores**. This does not establish achieved occupancy or utilization.

Nsight Compute fails with `ERR_NVGPUCTRPERM`; counters are unavailable. We
therefore cannot yet rank modular reduction against strided memory transactions,
barriers or occupancy as the dominant **resident-kernel** bottleneck. The
current [`mul_mod`](../include/ntt/mod_arith.hpp) uses u64 operands and a u128
remainder; specialized exact reduction is a candidate experiment, not a
measured winner. Host validation/copy/staging costs also need separate timing.

## 5. Thirty-bit results and scaling

Thirty-bit primes currently use exactly the same **u64 storage and arithmetic
implementation** as wider primes. They neither halve transfer size nor select
a dedicated u32 modular kernel.

For each of 35 matching N/K pairs, divide the 60-bit resident median by the
30-bit median. The median ratio is **0.882 for forward** and **0.881 for fused
product**; all 35 ratios are below one. In this run, 60-bit resident execution
is about 12% faster by that paired-ratio summary, not slower.

Do not conclude that wider primes are intrinsically faster. Prime values and
seeded residues differ; runtime remainder lowering may be input-dependent;
widths ran in fixed order; clocks and active-load telemetry were not captured
throughout the batch. The explanation remains unverified. Test a proven u32
implementation separately and repeat with interleaved/randomized order before
claiming a 30-bit advantage or diagnosing the cause of this inversion.

At 60 bits, increasing N from 4096 to 65536 at K=40 grows resident product
latency from 0.295 to 1.723 ms (**5.83×**) while N grows 16×. Increasing K
from one to 40 at N=65536 grows it from 0.0852 to 1.723 ms (**20.22×**).
This is consistent with fixed launch costs and improved use of parallel work
at larger shapes, but does not prove the occupancy mechanism or predict scaling
beyond K=40. Small shapes remain important: at N=4096/K=1 the M1 CPU product
is faster than the public GPU product for both displayed widths.

## 6. CPU comparison and measurement reliability

The CPU path is OpenFHE **1.6.1 NativeVector/CRT-FTT**, sequential tower
execution, OpenMP OFF and native optimization OFF, built with Apple LLVM 21.
It is not a best-tuned parallel DCRT/HEXL or CKKS application baseline.

At N=65536/K=40/60 bits, CPU/public-GPU ratios are **2.61× for forward** and
**5.28× for product**. Across the matrix, the M1 CPU takes longer than the
public GPU in 114/140 forward cells and 136/140 product cells. These are
**cross-platform latency comparisons**, not same-host acceleration factors,
architecture rankings or proof that one GPU is the most suitable purchase.

Only one ordered run per cell was collected. The 100 samples are repetitions
within that run, not 100 independent sessions; p5/p95 describe sample spread,
not confidence intervals. Do not overinterpret close differences:

- Largest 60-bit public fused product: median **12.431 ms**, mean **13.817 ms**,
  p5 **11.963 ms**, p95 **20.732 ms**.
- Its resident product: median **1.723 ms**, p95 **1.728 ms**.
- Across operation/cell pairs, median p95/median is 1.068 for public GPU,
  1.013 for events and 1.107 for CPU. Some cells have much larger tails.

There are non-monotonic public results, e.g. 60-bit forward N=16384/K=16
is 1.786 ms while K=24 is 1.348 ms. This is a reason to repeat/control the
experiment, not choose a dispatch boundary from this single noisy series.
Resident curves are smoother, but kernel counters and repeated sessions are
still needed to explain them.

## 7. Next work, without keeping a rental running

1. **Make split collection routine.** Add explicit GPU-only/CPU-only scope
   selection and platform provenance to the normal resumable collector. The
   existing legacy `--benchmark` still schedules CPU timings on the GPU host.
2. **Prepare resident-pipeline experiments.** Keep strict validation at the
   external input boundary, represent validated device buffers explicitly, and
   measure chains without repeated host roundtrips. Do not remove validation
   from untrusted public inputs to manufacture a speedup.
3. **Prepare one-factor kernel candidates offline.** Prove u32/30-bit and
   u64 reduction bounds against bigint/reference/OpenFHE tests; separately
   prepare the coalesced-layout candidate. Do not change both together and
   attribute the result to either one.
4. **Freeze a bounded next rental batch.** Include original baseline versus
   each candidate, exactness and sanitizers, repeated/interleaved timing order,
   active clocks/power metadata, and the same 30/50/55/60-bit/K≤40 matrix.
   Check counter permissions early; use the actual RTX 5090 for primary-device
   acceptance. Counter access may require a different rental environment.

No mathematical implementation changes or new GPU executions were made for
this analysis. Remaining OpenFHE base-conversion, Hybrid and integration gates
are unchanged. The [daily journal](journal/2026-11-14.md) tracks the evidence
and the [backlog](implementation-backlog.md) tracks subsequent implementation.
