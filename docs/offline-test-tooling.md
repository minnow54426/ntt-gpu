<div id="prepare-the-next-gpu-session-offline" aria-hidden="true"></div>

# Collect GPU and CPU measurements separately

Use the rental for GPU checks and timings; collect the OpenFHE CPU baseline
locally. Both runs use the same polynomial parameters and seeded inputs, while
retaining their own machine and build records.

The [November 14 batch](batch40-native-analysis.md) exercised an earlier CUDA
baseline. The November 16 collector and [resident runtime](resident-operation-chain.md)
pass host tests and CUDA compilation; their remaining native checks are still
open. The [preparation guide](offline-gpu-preparation.md) covers dependencies
and hardware setup.

<div id="rental-gpu-local-cpu-split" aria-hidden="true"></div>

## Collector and validation

From a reviewed clean checkout with pinned OpenFHE installed, run on the GPU:

```sh
python3 scripts/gpu_session.py --mode gpu \
  --cuda-root /path/to/cuda --openfhe-prefix /path/to/openfhe \
  --architectures native --benchmark --benchmark-scope gpu \
  --output-dir bench-results-gpu \
  --benchmark-warmups 5 --benchmark-samples 100
```

On the local Linux or macOS CPU machine, use the same source revision:

```sh
python3 scripts/gpu_session.py --mode cpu \
  --openfhe-prefix /path/to/local/openfhe \
  --benchmark --output-dir bench-results-cpu
```

CPU mode configures CUDA off and never calls nvcc, a CUDA executable or a GPU
profiler. Its successful report says `CPU_CHECKS_PASSED` and
`gpu_execution: NOT_RUN`. This path is verified on Linux; the newer macOS
collector path has control-flow tests but still needs a native M1 run.

### Choose the measurement scopes

A *workload* fixes N, the number of prime towers K, and the prime width.
K counts primes in one polynomial, not ciphertexts. Defaults cover five ring
sizes (4096–65536), four widths (30/50/55/60 bits), and
K=1/3/8/16/24/32/40: **140 workloads**.

<div class="site-table-scroll" role="region" aria-label="Benchmark collection modes and timing scopes" tabindex="0" markdown="1">
<div style="min-width:42rem" markdown="1">

| Session / selection | Jobs | Timing scopes |
|---|---|---|
| `--mode gpu --benchmark-scope gpu` | 280 | Five-operation GPU public calls, resident events/pinned copies |
| `--mode cpu` | 140 | Four-operation OpenFHE CPU baseline |
| `--mode gpu --benchmark-scope all` | 420 | Legacy public calls including scalar CPU forward, events, OpenFHE CPU baseline |

</div>
</div>

Add `--benchmark` to collect timings. Direct GPU collector commands default to
`all`; the rental helper `scripts/autodl_setup.sh` defaults to `gpu`.
Set `--benchmark-scope all` or `NTT_BENCHMARK_SCOPE=all` for a combined rental run.
GPU-only selection runs the native CTest group with its OpenFHE references
and fixtures, followed by the requested sanitizers. Run standalone CPU
checks and CPU timing on the local host.

Use `--benchmark-dimensions 65536 --benchmark-towers 3 40` for a bounded
batch. Any K from 1 through 40 is selectable; the default samples seven counts.
Match dimensions, tower counts, widths, warmups and samples across machines.
Different platforms are not a same-host acceleration comparison.

Use `--benchmark-prime-bits 30 60` to limit widths. For the matched default
versus legacy GPU experiment, use `--compare-ntt` and the fresh-directory
commands in the [rental queue](next-gpu-rental.md).

At 100 samples per operation, the default GPU selection records 154,000 timed
samples; CPU records 56,000; combined records 224,000. These repeat fixed
operands, not that many independent random inputs. Thirty-bit primes still use
`uint64_t` buffers and arithmetic. Parameter files contain zero placeholders;
the executables generate the seeded dense operands used for timings.

### Validate and retain records

[The collector](../scripts/gpu_session.py) keeps parameter files, raw JSONL,
logs, hashes and `report.json`. [The report validator](../scripts/benchmark_report.py)
rejects missing/truncated samples, duplicate indices, wrong parameters/scopes,
nonfinite or negative times, and summaries inconsistent with the raw samples.
A malformed record fails even if the executable exits zero.

To recompute a saved summary:

```sh
python3 scripts/benchmark_report.py /path/to/benchmark-n65536-k3-b30-public.log
```

Record validation checks consistency; exactness tests establish arithmetic
correctness. Hashes detect changes or corruption, not fabricated measurements.
The source and hardware record are still required.

### Restart and preserve partial results

Rerun the same command with `--resume --output-dir <record-directory>`.
Environment probes, build, exactness and requested sanitizers rerun. Successful
measurement jobs can be reused only when clean source, selected scope, binary
and OpenFHE hashes, hardware identity, parameters and repetitions match.
Their raw logs must pass hash and content validation again.

Changing CPU/GPU mode or scope requires a fresh directory. CPU resume ignores
unused CUDA binaries, GPU visibility and architecture settings. Clock readings
are not CPU identity keys. Old reports are archived; new stage logs have unique
attempt names. Reused jobs retain their original attempt identity.

Independent measurement failures preserve completed jobs, but the overall run
fails. Setup, exactness or sanitizer failure stops before timing. The default
per-command timeout is 1,800 seconds. Do not run concurrent collectors in one
directory. Export the **entire record directory** and verify its archive hash
before ending the rental.

### Optional selected profiling

Add `--profile` only when both `nsys` and `ncu` work and the host permits GPU
counter access. The bounded capture uses N=65536/K=3/60-bit, one warmup and two
samples. Nsight Systems records a public-call timeline; Nsight Compute's basic
set records the first eight launches of the event experiment. Attempt-specific
prefixes preserve earlier artifacts. Requested profiling fails if tools,
permissions or capture fail; omit the flag to leave profiling pending.

Profiler timings are separate from benchmark samples. Native build logs retain
`ptxas -v` resources and line mappings. Full per-pass counters, active telemetry,
production CKKS batches and an evidence-backed bottleneck decision remain
later work.

## Raw measurement protocol

The CLI protocols use a parameter file's exact N/q/root list and the same
fixed seed. Reset and packing occur **before** every warmup/sample timer,
including inverse. Output includes individual samples, summaries and an
explicit completion marker. Percentiles use linear interpolation at
`(count−1)*fraction`.

| Command | Operations and timing boundary |
|---|---|
| `ntt_cuda_cli gpu-bench-jsonl parameters.txt 5 100` | GPU forward, inverse, pointwise, unfused product and fused product; host-wall public-call time |
| `ntt_cuda_cli bench-jsonl parameters.txt 5 100` | The same GPU operations plus scalar reference forward; legacy combined public protocol |
| `ntt_cuda_cli events-jsonl parameters.txt 5 100` | Resident forward, inverse, pointwise and product; separate pinned H2D/D2H copies |
| `ntt_resident_bench parameters.txt 2 10` | Host-wall time for the device-resident chain, public host boundary, and explicit host-roundtrip control |
| `ntt_openfhe_bench parameters.txt 5 100` | Four OpenFHE NativeVector/CRT-FTT CPU operations |

Select `--benchmark-scope gpu-chain` to add the resident prototype timings
to the GPU public and CUDA-event records. For the device chain, inputs are
uploaded once and the timer ends when the coefficient output is ready on the
device. The exact coefficient check downloads it after the timer. The host
boundary times both input uploads and the final download. The roundtrip
control also downloads and re-uploads each intermediate. All three scopes
include prototype allocations, copies and synchronization inside their measured
interval. Each checks every result and excludes one-time plan setup. Keep
warmups and samples at 200 combined or fewer.

Public GPU calls include validation, guards, transfers and the multiply
interface's by-value input allocation/copy. Result retention/destruction is
outside the repeated timer. Last results are checked against the reference
outside timing; the full exactness suite remains a separate check.
`gpu-bench-jsonl` uses `host_wall_gpu_public_call`; legacy `bench-jsonl`
retains its distinct protocol. The old `bench` command is a 46-bit legacy path
and is no longer used by the collector.

The [measurement helper](../include/ntt/measurement.hpp) has fake-clock tests
for reset order, warmups, summaries and JSON escaping. They test control logic,
not performance.

## Resident events and CPU baseline

The event experiment excludes setup, allocation, input upload/reset and output
download/validation from resident timing. Each iteration drains the default
stream after reset, records start/stop events and waits for the stop event.
Results are checked afterward. Multi-launch intervals can include submission
gaps; they are not isolated per-kernel instruction times.

Pinned H2D and D2H copies use preallocated host memory and `cudaMemcpyAsync`.
Their times are separate experiments, not an additive breakdown of the
pageable public call. The public wrapper remains synchronous. There is no
measured copy/compute overlap or multi-stream throughput here. See NVIDIA's
[event API](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__EVENT.html)
and [timing guidance](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html).

The OpenFHE comparator uses sequential per-tower preconditioned CRT-FTT and
NativeVector arithmetic. Tables, packing, reset and reference checks are
outside timing; internal library allocations remain included. It records
compiler/OpenFHE flags and OpenMP/native-optimization settings. This is not a
best-tuned DCRT/OpenMP/HEXL or complete CKKS baseline, and its multiplication
allocation boundary differs from the GPU public API.

## Exactness and sanitizer scopes

`cuda_benchmark_matrix_exact` checks the 140 default parameter cells against
OpenFHE with seven patterns, independent inverse, roundtrip and product.
Custom K values also receive the benchmark's reference checks, but do not
extend that fixed direct-OpenFHE gate.

Each of memcheck, racecheck, initcheck and synccheck runs three scopes:

- Legacy `--smoke`: N=65536, three widths, zero/dense patterns.
- `--matrix-smoke`: N=4096/65536, 30/60-bit, K=3/40, zero/dense.
- `ntt_resident_cuda_tests --smoke`: dense N=65536, unsorted 55/50/60-bit basis.

That is twelve sanitizer runs in the current collector. The older Ada batch
passed the first two scopes; it predates the resident scope. Host gates now
include `resident_runtime_host` and `resident_openfhe_host`; CUDA builds add
`resident_cuda_preflight` and GPU-labelled `resident_cuda_exact`.
Host/compiler success does not close these native gates.

## Shared allocation ownership and injected failures

[SetupResources](../include/ntt/setup_resources.hpp) registers each of the
wrapper constructor's six allocations immediately and releases them in reverse
order. Its four constant uploads use the same adapter boundary. Host tests
exercise this actual ownership code with a host-memory adapter.

`setup_ownership_host` injects before and during each of ten adapter operations:
20 failure cases, plus normal and cleanup-reporting paths. The GPU-labelled
`cuda_setup_fault_injection` uses the real constructor at N=65536 with three
50-bit towers and verifies successful CUDA frees after pre-call injection.
That probe passed in the older Ada batch.

Synthetic exceptions do not reproduce driver OOM, a partial DMA failure,
wrong-device restoration or a destroyed context. Those tests remain open.
Ordinary destructor cleanup is nonthrowing and cannot report errors to its
caller; a diagnostic trace is not a production recovery interface.

## Verify without renting

```sh
cmake -S . -B build -DNTT_ENABLE_CUDA=OFF
cmake --build build -j2
ctest --test-dir build --output-on-failure
.venv/bin/python -m unittest discover -s tests/tooling -v
```

These run host arithmetic, ownership, timing and collector/report checks.
Use the [preparation guide](offline-gpu-preparation.md) for compiler-only CUDA
checks. Native validation must identify the exact source revision and machine;
older results cannot establish correctness of later runtime changes.
