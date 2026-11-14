# Prepare the next GPU session offline

**2026-11-14 · host checks complete; new CUDA code uncompiled/unrun locally**

Freeze update: [compiler-only CI for f1ecff1](https://github.com/minnow54426/ntt-gpu/actions/runs/43599616970)
now passes for **sm_89 and sm_120**, including driverless host checks. Local
regression passes 18 standalone and 22 OpenFHE CPU CTests (140 workload cells).
The batch is ready for the next rental's **native validation**, not accepted
as a GPU result. CUDA/OpenFHE combined native gates, event measurements,
sanitizers and profiler captures still need execution on a real device.

The [first-run analysis](first-native-analysis.md) identified measurement and
failure-path gaps. This increment implements host-testable preparation for the
next session. It does not reconnect to the stopped rental or close B02/B04.

## Raw measurement protocol

The new `ntt_cuda_cli bench-jsonl <parameter-vector> [warmups samples]` path
uses the vector file's exact N/q/root list; coefficients are generated from a
fixed documented seed, not copied from that file. The batch covers N=4096,
8192, 16384, 32768 and 65536. Every
warm-up and timed call resets fixed inputs **before** the timer, including the
inverse. Defaults are five warm-ups and 100 samples per operation.

Six operations produce separate sample and summary records:

- Scalar reference forward, with tower packing/reset outside timing.
- Public GPU forward and independent inverse.
- Public GPU pointwise multiply.
- The complete four-public-call unfused product.
- The complete fused public product.

All GPU numbers are **host-wall public-call** time, including transfers,
validation and guards, not CUDA-event/kernel time. Fused multiplication still
includes its by-value left copy/output allocation; it retains the result and
releases the previous result outside timing. This is explicitly different
from the original benchmark's discarded result. Last results for each operation
are checked against the reference outside timing; full exactness is a separate
suite. This command's scalar CPU column is not OpenFHE; the new separate
OpenFHE comparator below avoids conflating them.

Workload metadata includes exact q/roots, GPU identity, driver/runtime API,
input seed/policy, warm-ups, sample count and timing/setup scope. Raw samples
survive in JSONL, with mean, median and p5/p95 using linear interpolation at
`(count−1)*fraction`. Output ends with an explicit completion marker.

The host-only [measurement helper](../include/ntt/measurement.hpp) is tested
with a fake clock: reset occurs outside timing, each operation sees a reset
input, warm-up calls are distinct, and percentile/JSON escaping checks pass.
These tests are not performance observations.

## Resident events and CPU baseline

`ntt_cuda_cli events-jsonl <parameter-vector> [warmups samples]` uses the same
q/roots and operands. Its six operations are resident forward, inverse,
pointwise and fused ring product, plus one-buffer H2D and D2H pinned transfers.
Setup, allocations, input reset/upload and validation/download are outside
resident timing. Both copy directions use a preallocated pinned buffer and
`cudaMemcpyAsync`; the public wrapper remains synchronous/pageable.

Every warm-up/sample drains the default stream after reset, records a start
event, enqueues the operation, records/synchronizes the stop event and reads
elapsed milliseconds. Results are checked after every iteration outside events.
Multi-launch event intervals can include submission gaps; they are not a sum
of isolated per-kernel counters or pure device arithmetic. Event resources and
pinned allocations have scoped cleanup. None of this CUDA path has compiled
or run locally. The host external-clock protocol tests only ordering/error logic.

The policy follows NVIDIA's [event API](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__EVENT.html)
and [timing/pinned-memory guidance](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html).
Event results and pinned-copy results come from **separate experiments**, not
an additive decomposition of pageable public-call time. There is no overlap,
multi-stream throughput, per-pass event instrumentation or pinned public API yet.

`ntt_openfhe_bench <parameter-vector> [warmups samples]` is built with the
pinned OpenFHE compatibility targets. It times OpenFHE's preconditioned native
CRT-FTT forward/inverse, NativeVector pointwise product and complete ring
product, using sequential per-tower calls and the exact same seeded operands.
Tables, packing, reset and reference checks are outside host-wall timing.
Metadata records OpenFHE/compiler flags and OpenMP/native-optimization settings;
the collector retains CPU identity and build flags. All four outputs are
checked against our reference; the CPU protocol gate follows the default
140-workload matrix described below.
This is an optimized-arithmetic OpenFHE comparator, not a best-tuned DCRT/OpenMP,
HEXL or end-to-end CKKS baseline. Library-internal allocations remain included;
unlike GPU public multiplication it does not include a by-value return copy.

## Collector and validation

With `--benchmark`, [gpu_session.py](../scripts/gpu_session.py) now generates
parameter vectors at **30, 50, 55 and 60 bits**, all five dimensions above,
and K=1/3/8/16/24/32/40 towers: **140 workloads and 420 measurement jobs**, three timing
scopes per workload. Every scope must match the requested N/K/q/roots/seed
and repetition counts. Thirty-bit parameters still use `uint64_t` storage;
no `uint32_t` specialization or reduced transfer size is claimed.
It retains `benchmark-n65536-k3-b30-public.log` and similarly named event/CPU
JSONL logs, and adds regenerated summaries to `report.json`. The old `bench` command
remains a legacy 46-bit path; the collector no longer uses it.

[benchmark_report.py](../scripts/benchmark_report.py) rejects truncated output,
missing operations/samples, wrong scope/width, duplicate indices, nonfinite or
negative samples, and summaries inconsistent with raw observations. Collector
failure remains nonzero even if the executable exited zero with malformed
output. This validates the record, not GPU authenticity or mathematical
correctness; those require the native gates and source provenance.

On the next compatible GPU host, from a reviewed clean checkout:

```sh
python3 scripts/gpu_session.py --mode gpu \
  --cuda-root /path/to/cuda --openfhe-prefix /path/to/openfhe \
  --architectures native --benchmark --profile \
  --output-dir bench-results-next \
  --benchmark-warmups 5 --benchmark-samples 100
```

Offline analysis after that future session:

```sh
python3 scripts/benchmark_report.py /path/to/benchmark-n65536-k3-b30-public.log
```

`--benchmark-dimensions 65536 --benchmark-towers 3 40` narrows a diagnostic batch
while retaining all four prime widths. Defaults use five warm-ups and 100
samples per operation. Any individual prime count from 1 through 40 can be
selected with `--benchmark-towers`; the default is a representative sweep,
not every integer count. The complete default batch contains 224,000 timed
operation samples, not 224,000 independent random inputs; rental runtime remains
unmeasured. Commands have a configurable timeout (default 1,800 seconds each).

Here K means **RNS primes/towers in one polynomial**, not the number of
ciphertexts or independent polynomials. Benchmark parameter files contain
zero coefficient placeholders to avoid generating unused, huge CRT integers
at K=40. The executables still generate the same seeded dense operands; real
correctness fixtures retain their original nonzero input generation.

The full suite now requires `cuda_benchmark_matrix_exact`: all 140 default
parameter cells, seven patterns, independent inverse, roundtrip and product
comparisons directly against OpenFHE. This is prepared, not native evidence.
Each sanitizer runs the historical `--smoke` and new `--matrix-smoke`: the
latter selects N=4096/65536, 30/60-bit K=3/40, zero/dense inputs, not the full matrix.
Non-default integer tower counts get the benchmark's reference checks, but are
not additional cells of this fixed direct-OpenFHE/sanitizer gate.

### Restart and preserve partial results

Rerun the same command with `--resume --output-dir bench-results-next` after
an interrupted/failed attempt. Environment probes, build, exactness and
sanitizers always rerun. Prior reports are archived and stage logs get unique
retry names. Benchmark jobs resume only with the same clean source revision,
binary and OpenFHE-library hashes, device/driver identity, selection policy,
matrix and repetition counts. Completed raw logs must pass SHA-256 and full
record/parameter validation again. Different contexts require a fresh directory.

Independent measurement jobs continue after a measurement failure, preserving
successful logs/summary checkpoints; the overall batch still exits nonzero.
Setup/exactness/sanitizer failures stop before benchmarks. An output hash
detects corruption, not fabricated measurements. Attempt IDs identify reused
results; they do not normalize thermal/clock differences between attempts.
Do not run concurrent collectors in the same directory. Download the **entire
record directory**, including manifest, raw logs, report histories, parameter
files and profiler artifacts, and verify its archive hash before stopping rental.

### Optional selected profiling

`--profile` requires both `nsys` and `ncu`, with usable GPU counter permissions.
It runs a separate N=65536/K=3/60-bit workload at one warm-up/two samples:
Nsight Systems captures a public-call timeline; Nsight Compute's basic set
captures the **first eight kernel launches** of the event experiment, not all
operations. Unique per-attempt artifact prefixes avoid overwriting prior data.
Profiler-perturbed timings are not accepted as benchmark samples. Missing tools,
permission failures or capture errors fail the requested profiling stage;
otherwise omit the flag and keep profiling pending. Compiler `ptxas -v` resource
output and line mappings are enabled in native builds and retained in build logs.

Measured bottleneck decisions, per-pass events, full memory-sector/bank/occupancy
counter analysis, active telemetry, deep CKKS levels and ciphertext batch sweeps
remain future work. Prepared collection commands do not close D14 or P2.

## Shared allocation ownership and injected failures

[SetupResources](../include/ntt/setup_resources.hpp) owns the actual constructor's
six allocations, registers ownership immediately after allocation succeeds,
and releases acquired pointers in reverse order. Four constant uploads cross
the same runtime adapter. Host tests use a host-memory adapter through this
same owner; the CUDA adapter uses real allocation/upload/free calls. This keeps
cleanup bookkeeping at one testable seam rather than copying the constructor
algorithm into a fake test.

`setup_ownership_host` injects at all ten operation positions in two ways:
before the adapter call and as an adapter exception. It verifies acquired
allocations, completed uploads, reverse release order, no surviving host
allocations, and nonthrowing cleanup-failure accounting. These are 20 failure
cases plus normal/cleanup-reporting paths, not actual CUDA failures.

The prepared `cuda_setup_fault_injection` GPU-labelled CTest uses the real
constructor at N=65536 and three 50-bit towers. It runs one successful setup,
then injects before each of six allocations and four uploads. It requires each
acquired allocation to reach a successful real CUDA free. The collector now
requires this gate, the new matrix gate and the host gates to be registered.

This new CUDA probe has **not** been compiled or executed on this Mac. Native
compilation/execution is mandatory before accepting the refactor. Even after
it passes, a synthetic pre-call exception does not simulate a partially failed
driver transfer, OOM state, wrong-device restore failure or destroyed context.
Those remain separate failure tests. Ordinary destructor cleanup still cannot
surface errors to the caller; the diagnostic trace is not a production lifecycle
interface.

## Verify without renting

```sh
cmake -S . -B build -DNTT_ENABLE_CUDA=OFF
cmake --build build -j 8
ctest --test-dir build --output-on-failure
.venv/bin/python -m unittest discover -s tests/tooling -v
```

These run real host ownership, timing/statistics and collector/report logic,
not an emulated GPU arithmetic backend. Historical native results in the
[session record](journal/2026-11-14.md) apply to source 19cc285, **not** the new
unverified CUDA changes. Keep the primary-device and native failure gates open.
