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
per workload. Tables scroll horizontally on a phone.

<div class="site-table-scroll" role="region" aria-label="Matched NTT timing ratios" tabindex="0">
<table style="min-width:34rem">
<colgroup><col style="width:12rem"><col style="width:5rem"><col><col></colgroup>
<thead><tr><th>Timing scope and operation</th><th>Median ratio</th><th>Range</th><th>Group faster</th></tr></thead>
<tbody>
<tr><td>Device event, forward</td><td>1.52</td><td>1.20–1.61</td><td>8/8</td></tr>
<tr><td>Device event, inverse</td><td>1.50</td><td>1.18–1.59</td><td>8/8</td></tr>
<tr><td>Device event, fused multiplication</td><td>1.49</td><td>1.26–1.53</td><td>8/8</td></tr>
<tr><td>Device event, pointwise control</td><td>1.01</td><td>0.68–1.21</td><td>4/8</td></tr>
<tr><td>Host public call, forward</td><td>1.01</td><td>0.89–1.38</td><td>4/8</td></tr>
<tr><td>Host public call, inverse</td><td>1.12</td><td>1.00–1.23</td><td>8/8</td></tr>
<tr><td>Host public call, fused multiplication</td><td>1.05</td><td>0.93–1.21</td><td>5/8</td></tr>
<tr><td>Resident interface, device chain</td><td>1.16</td><td>0.85–1.47</td><td>7/8</td></tr>
<tr><td>Resident interface, host boundary</td><td>1.03</td><td>0.66–1.28</td><td>4/8</td></tr>
<tr><td>Resident interface, host roundtrip</td><td>1.03</td><td>0.87–1.33</td><td>5/8</td></tr>
</tbody></table>
</div>

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

<div class="site-table-scroll" role="region" aria-label="Order balanced resident repeat" tabindex="0">
<table style="min-width:42rem">
<colgroup><col style="width:4rem"><col style="width:7rem"><col><col><col></colgroup>
<thead><tr><th>Order</th><th>Schedule</th><th>Device chain</th><th>Host boundary</th><th>Host roundtrip</th></tr></thead>
<tbody>
<tr><td>1</td><td>Legacy</td><td>0.7401</td><td>3.2347</td><td>9.1446</td></tr>
<tr><td>2</td><td>Group</td><td>0.7172</td><td>3.3319</td><td>7.6012</td></tr>
<tr><td>3</td><td>Group</td><td>0.7264</td><td>3.2802</td><td>7.5232</td></tr>
<tr><td>4</td><td>Legacy</td><td>0.7298</td><td>3.0951</td><td>7.6587</td></tr>
</tbody></table>
</div>

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


## Resident conversion on November 22

Both schedules passed resident ModUp and ModDown comparisons with pinned
OpenFHE on the reported RTX 4080 SUPER. The collector matched 32 jobs per
schedule, including eight conversion jobs per schedule. Each operation used
two warmups and ten samples. The two schedules ran sequentially.

At N=65,536 with 40 60-bit Q primes and three P primes, median times in
milliseconds were:

| ModUp followed by ModDown | Group twiddle | Legacy |
|---|---:|---:|
| Inputs already on device | 3.776 | 3.687 |
| Upload input and download result | 25.460 | 21.307 |
| Also download and re-upload intermediate QP | 47.831 | 44.778 |

For the resident pair, the median legacy/group ratio across the eight
workloads was 1.004. The host controls include copies and host work such as
residue validation. Their difference from the resident timings cannot be
assigned entirely to PCIe transfers. The largest intermediate contains 43
towers, or 21.5 MiB, and the explicit roundtrip copies it in each direction.

The conversion timeline has no per-operation markers. It spans six
sessions and includes verification downloads. Across the trace, 184
allocation calls took 10.617 ms in total, 184 frees took 11.058 ms, and 181
asynchronous copy calls took 19.475 ms. These are aggregate CPU CUDA API
durations. The first stream creation took 138.498 ms and occurred outside
the benchmark timers. Hardware counters were unavailable.

The first optimization comparison should test reusable device workspace
allocations against the current allocation path. Keep table generation,
polynomial transfer policy, exactness checks and schedule fixed. Measure the
resident pair and the host-facing pair separately. Add operation ranges to
the profile to identify which calls occur inside each timer. This experiment
follows the measured host overhead without assuming allocations account for
most of it.

## What changed in the November 24 Ada results

The RTX 4080 SUPER batch compared both schedules on the same eight workload
shapes, with two warmups and ten samples per job. Both schedules passed
native exactness and all sanitizer stages. Conversion runs replayed the
pinned OpenFHE references generated on the Mac. The
[rental record](journal/2026-11-24.md#complete-the-native-ada-batch) lists the
tested revision and the profiler limits.

Ratios are the legacy median divided by the group-twiddle median for each
workload. Values above one favor group twiddles.

| Timing boundary | Forward median ratio | Inverse median ratio | Group faster |
|---|---:|---:|---|
| Resident CUDA events | 1.463× | 1.471× | 8 of 8 for both |
| Public-call wall time | 0.914× | 0.891× | 3 of 8 for both |

Forward public calls also varied by shape. At N=65,536 with 40 towers, the
30-bit case took 7.410 ms with group twiddles and 6.747 ms with legacy.
The 60-bit case took 6.672 ms and 7.301 ms respectively. The two builds ran
sequentially. Their resident NTT repeat at N=65,536 with three 60-bit primes
also varied. Group-twiddle medians changed from 1.081 ms to 0.708 ms while
the two legacy medians stayed near 0.73 ms. A controlled repeat of the
public-call workload is needed to interpret the reversal.

For the group-twiddle ModUp/ModDown pair at N=65,536 with 40 60-bit Q primes,
the median was 4.244 ms with device-resident inputs, 20.989 ms at the host
boundary, and 39.374 ms with an explicit intermediate round trip. The
host-facing pair also includes upload validation, synchronization and
staging. The difference between the timings measures those boundaries as a
whole.

The Nsight Systems profiles used three 60-bit Q primes at N=65,536. They
show CUDA kernels and nested operation ranges. The conversion report
contains 12 measured `ntt.sample` ranges. Its named operation totals also
include warmups, and its session setup ranges sit outside the timers. The
whole-trace summary cannot attribute allocations, copies or CPU waits to
a specific measured operation. The measured 40-tower case still needs its
own trace. Nsight Compute did not return hardware counters.

For the RTX 5090 follow-up, repeat the two large public-call cases in
legacy/group/group/legacy order. Then capture the large conversion case and
inspect the CUDA API calls and GPU copies inside each measured NVTX sample.
That evidence can guide a workspace reuse comparison while keeping plan
generation, transfers and exactness checks fixed.
