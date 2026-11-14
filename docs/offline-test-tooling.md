# Prepare the next GPU session offline

**2026-11-14 · host checks complete; new CUDA code uncompiled/unrun locally**

The [first-run analysis](first-native-analysis.md) identified measurement and
failure-path gaps. This increment implements host-testable preparation for the
next session. It does not reconnect to the stopped rental or close B02/B04.

## Raw measurement protocol

The new `ntt_cuda_cli bench-jsonl <parameter-vector> [warmups samples]` path
uses the vector file's exact N/q/root list; coefficients are generated from a
fixed documented seed, not copied from that file. N must be 65536. Every
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
suite. The CPU comparator is not optimized OpenFHE.

Workload metadata includes exact q/roots, GPU identity, driver/runtime API,
input seed/policy, warm-ups, sample count and timing/setup scope. Raw samples
survive in JSONL, with mean, median and p5/p95 using linear interpolation at
`(count−1)*fraction`. Output ends with an explicit completion marker.

The host-only [measurement helper](../include/ntt/measurement.hpp) is tested
with a fake clock: reset occurs outside timing, each operation sees a reset
input, warm-up calls are distinct, and percentile/JSON escaping checks pass.
These tests are not performance observations.

## Collector and validation

With `--benchmark`, [gpu_session.py](../scripts/gpu_session.py) now generates
parameter vectors at 50, 55 and 60 bits, then runs the sample protocol on each.
It retains raw `benchmark-b50.log`, `benchmark-b55.log`, `benchmark-b60.log`
JSONL and adds regenerated summaries to `report.json`. The old `bench` command
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
  --architectures native --benchmark \
  --benchmark-warmups 5 --benchmark-samples 100
```

Offline analysis after that future session:

```sh
python3 scripts/benchmark_report.py /path/to/benchmark-b50.log
```

The sample recorder does not yet provide resident CUDA events, per-pass/copy
timing, Nsight counters, active telemetry, batch sweeps or an optimized OpenFHE
CPU benchmark. Those are explicit next increments, not completed by raw samples.

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
requires this fifth GPU gate and both new host gates to be registered.

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
