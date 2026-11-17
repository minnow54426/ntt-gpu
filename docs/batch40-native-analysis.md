# GPU results: what we learned

**Measured on November 14, 2026 · temporary sm_89 GPU · M1 Pro CPU baseline**

The tested NTT baseline passes its executed correctness checks. Keeping
intermediate arrays on the GPU reduces the measured multiplication time.
Next, validate the newer resident API and measure the remaining public-call
costs before adding base conversion or key switching.

The rental reports **RTX 4080 SUPER, compute capability 8.9**. These are
**not RTX 5090 results**.

## Correctness and coverage

- All **29 native tests** and **eight sanitizer runs** passed.
- All **280 GPU measurement jobs** and **140 M1 CPU jobs** passed.
- The matrix covers N=4096–65536, 1–40 RNS prime towers, and 30/50/55/60-bit primes.

A tower stores the polynomial modulo one prime. K counts those towers, not
independent ciphertexts. Exact
comparisons use the pinned OpenFHE 1.6.1 path. GPU base conversion, Hybrid key
switching and encrypted inference are **not implemented or validated by this run**.

## Largest workload

N=65536, K=40, 60-bit primes. All times below are **medians in milliseconds**.

| Operation | GPU public call | GPU resident experiment | M1 OpenFHE CPU |
|---|---:|---:|---:|
| Forward NTT | 8.079 | 0.552 | 21.071 |
| Inverse NTT | 8.045 | 0.598 | 21.827 |
| Pointwise multiply | 11.858 | 0.048 | 3.115 |
| Complete polynomial multiply | 12.431 | 1.723 | 65.687 |

Public calls include host validation and transfers. The resident experiment
excludes input upload and output validation/download; it is **not end-to-end
latency**. Its difference from public time is not pure PCIe overhead.

M1-versus-GPU numbers compare different platforms, not same-host acceleration
or the best possible parallel CPU implementation.

## Three findings

**1. Keep data on the GPU.** Directly measured fused multiplication beats the
unfused chain in all 140 workloads: **2.36× median benefit**, ranging from
1.77× to 3.03×. At the largest 60-bit workload, it reduces 33.412 ms to
12.431 ms. It removes repeated host roundtrips, not NTT kernel launches.
Preserve this residency when adding base conversion and key switching.

**2. The public-call cost matters.** At the largest workload, resident
multiplication takes 1.723 ms versus 12.431 ms through the public interface.
Transfers, host array scans, staging, allocations and launch overhead need
separate measurements. Small jobs can favor CPU: at N=4096/K=1, the M1 CPU
product beats the public GPU call for both 30-bit and 60-bit primes.

**3. Thirty-bit arithmetic still needs its own implementation.** All widths
currently use u64 storage and the same modular arithmetic. In this run the
60-bit resident cases are about 12% faster than their paired 30-bit cases.
The reason is unproven; prime/input values and fixed run order differ. This
does not establish an intrinsic advantage for wider primes.

## What to do next

1. Use the [split collector](offline-test-tooling.md#collector-and-validation):
   GPU-only measurement jobs on the rental and the OpenFHE CPU baseline locally.
2. Validate the [prepared resident operation chain](resident-operation-chain.md)
   on named GPU hardware, including its lifetime and failure behavior.
3. Test arithmetic and memory-layout candidates separately.
4. Repeat a frozen batch on the actual RTX 5090, with controlled ordering and
   profiling permissions checked early.

All preparation and analysis can proceed locally; no rental needs to stay
running. Follow the [roadmap](openfhe-integration-plan.md) and
[daily journal](journal/2026-11-14.md) for the evidence and remaining gates.

## Measurement notes

<details markdown="1">
<summary>Protocol, uncertainty and evidence</summary>

Each operation uses five warmups and 100 samples with matched N/q/roots/seed.
The reported 32 GiB GPU capacity is unusual for that model; its physical
configuration is unverified.
The GPU batch contains 154,000 timed samples; the separate CPU batch contains
56,000. Raw JSONL summaries and manifest hashes were revalidated locally.

The original mixed batch was cancelled after 366 completed jobs. The fresh
280-job GPU batch is separate; the cancelled run is not labelled a full pass.
The archived benchmark-only patch changes timing selection, not device kernels.

The M1 baseline uses sequential OpenFHE NativeVector/CRT-FTT, OpenMP OFF and
native optimization OFF. It is not a tuned parallel DCRT or CKKS baseline.

Only one ordered run per workload was collected. At the largest 60-bit workload,
public product p95 is 20.732 ms versus a 12.431 ms median; close differences and
non-monotonic public timings need repeated runs. Sample percentiles are not
confidence intervals.

Large pinned-copy experiments reach approximately 13.0 GB/s H2D and 13.5 GB/s
D2H payload rates. They do not decompose the pageable public-call measurements.
A selected Nsight Systems trace was captured. Nsight Compute hardware counters
were blocked by provider permissions, so the dominant resident-kernel
bottleneck is still unproven.

Sanitizers cover bounded smoke inputs, not every input or concurrent lifecycle.
See the [November 14 record](journal/2026-11-14.md) for archive identity,
exactness scope, source provenance and local evidence locations.

</details>
