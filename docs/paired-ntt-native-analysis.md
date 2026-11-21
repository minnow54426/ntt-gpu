# Analyze the paired Ada NTT measurements

The November 19 Ada sample favors group twiddles for device-event NTT
timings. The median per-workload legacy/group ratios were 1.52 for forward,
1.50 for inverse and 1.49 for fused multiplication. Host-call gains were
smaller and sensitive to variability. Keep the group-twiddle default while
measuring the resident interface's allocation, parameter and transfer costs.

## What was measured

The server reported RTX 4080 SUPER, compute 8.9, driver 595.71.05 and CUDA
12.8. The clean source snapshot was 0f2df64, based on the local work after
587598a. Both schedules passed native exactness and the selected sanitizers.
The [run record](journal/2026-11-19.md#native-gpu-batch-completed) describes the
export and checks.

Each schedule ran eight workloads, combining N=4,096/65,536, K=3/40 and
30/60-bit primes, with two warmups and ten samples per operation. Both used
the same coefficients, primes, roots and fixed seed. The default schedule
ran before the legacy schedule.

The ratio below is the legacy median time divided by the group-twiddle
median time for one matched workload. A ratio above one favors group
twiddles. Each table entry summarizes those eight ratios, with equal weight
per workload.

| Timing scope and operation | Median ratio | Range | Group faster |
|---|---:|---:|---:|
| Device event, forward | 1.52 | 1.20–1.61 | 8/8 |
| Device event, inverse | 1.50 | 1.18–1.59 | 8/8 |
| Device event, fused multiplication | 1.49 | 1.26–1.53 | 8/8 |
| Device event, pointwise control | 1.01 | 0.68–1.21 | 4/8 |
| Host public call, forward | 1.01 | 0.89–1.38 | 4/8 |
| Host public call, inverse | 1.12 | 1.00–1.23 | 8/8 |
| Host public call, fused multiplication | 1.05 | 0.93–1.21 | 5/8 |
| Resident interface, device chain | 1.16 | 0.85–1.47 | 7/8 |
| Resident interface, host boundary | 1.03 | 0.66–1.28 | 4/8 |
| Resident interface, host roundtrip | 1.03 | 0.87–1.33 | 5/8 |

Device-event intervals exclude input/output transfers and session setup.
Host public calls include their validation and transfers. The resident
device chain includes allocation, copies, events and host orchestration,
with inputs already uploaded. Its output check downloads after the timer.
The two resident host controls include input uploads and the final download;
the roundtrip control additionally copies intermediates through the host.

The pointwise and transfer controls vary despite having equivalent arithmetic
or copy work between schedules. Treat the ten-sample sweep as an initial
comparison. Clock or host scheduling drift can affect the fixed run order.

## The order-balanced resident repeat

A follow-up used N=65,536, K=3, 60-bit primes, five warmups and fifty samples
per operation. The order was legacy, group, group, legacy. These are median
host-wall milliseconds from each independent run.

| Order | Schedule | Device chain | Host boundary | Host roundtrip |
|---:|---|---:|---:|---:|
| 1 | Legacy | 0.7401 | 3.2347 | 9.1446 |
| 2 | Group | 0.7172 | 3.3319 | 7.6012 |
| 3 | Group | 0.7264 | 3.2802 | 7.5232 |
| 4 | Legacy | 0.7298 | 3.0951 | 7.6587 |

Pairing runs 1/2 and 4/3 gives device-chain ratios of 1.032 and 1.005.
The corresponding host-boundary ratios are 0.971 and 0.944, and host-roundtrip
ratios are 1.203 and 1.018. The larger device-event advantage does not carry
through consistently to these interface measurements. Pooling the four runs
would hide this variation.

## Read the timeline with its setup boundary

The captured resident experiment contains three sessions, warmups and checks.
It records 71 cudaMalloc calls, 71 cudaFree calls and three stream creations.
The stream-creation calls consumed 240.246 ms in the whole trace, including a
240.218 ms first call. Session creation occurs outside the benchmark timers.

Consequently, the trace's 84.8% stream-creation share describes the whole
profiled process. It cannot identify the dominant cost inside the measured
resident calls. Inspect allocation and host orchestration within explicitly
marked operation ranges in the next profile. Hardware counters were blocked
by ERR_NVGPUCTRPERM, leaving occupancy and memory-traffic conclusions open.

## Reproduce the analysis on the M1 Pro

Verify the saved archive checksum and analyze the extracted record:

```sh
cd build/gpu-session-20261119
shasum -a 256 -c session-20261119.tar.gz.sha256
cd ../..
python3 scripts/analyze_gpu_batch.py \
  build/gpu-session-20261119/session-20261119 \
  --output-dir build/paired-analysis-review
```

Use a fresh output directory. The analyzer checks native/sanitizer status,
the complete selected matrix, log hashes, raw summaries, binary identities,
paired inputs and environment settings. It also validates the four balanced
repeat records. It writes 112 matched operation rows to CSV, twelve repeat
rows to a separate CSV, and a JSON summary. Hash validation checks record
consistency; the native tests supply the arithmetic evidence.

The next work is to measure the
[resident ModUp/ModDown chain](basis-conversion-preparation.md#keep-modup-and-moddown-in-the-resident-chain),
then evaluate conversion-plan caching and allocation reuse. The November 19
source predates that composition. RTX 5090 acceptance and complete CKKS
application timing remain future measurements.
