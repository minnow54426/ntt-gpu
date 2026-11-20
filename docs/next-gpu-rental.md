# Bundle the next RTX 5090 session

**Status: the expanded batch passed on November 19 on a server reporting
RTX 4080 SUPER / compute 8.9.** Both schedules also compiled for sm_120.
The results and CUDA timeline were copied to the M1 Pro and verified.
RTX 5090 hardware acceptance remains open. The queue below can be reused for
that device after the next group of changes is ready.

The November 20 resident ModUp/ModDown composition is a later source change.
Its extended `resident_cuda_exact` fixture and `--smoke` cases must pass on
the next GPU run. The earlier native result covers standalone conversion
and the earlier resident NTT chain. See the
[conversion guide](basis-conversion-preparation.md#keep-modup-and-moddown-in-the-resident-chain).

- The collector runs two versions: the new group-twiddle schedule and the
  older, or legacy, schedule. It checks that both use the same inputs and GPU,
  and saves each version’s results separately.
- The resident benchmark checks each result against an exact CPU product.
  The deferred host driver exercises the protocol and labels its output as
  simulation. The native collector rejects that output.
- An independent CPU calculation matches the selected OpenFHE version’s
  ApproxModUp and ApproxModDown operations. These change the set of primes
  used to represent a polynomial. Tests use N=65,536, one to five active Q
  primes and three auxiliary P primes. The reusable converter now also has
  a broader all-dimension CPU comparison and prepared CUDA correctness probes.
  See [conversion preparation](basis-conversion-preparation.md).

## One rental, ordered GPU work

| Order | GPU-only item | Evidence to bring home |
|---|---|---|
| 1 | Preflight the actual RTX 5090, toolkit, driver, and profiler tools | Device identity, commit, build options and tool versions |
| 2 | Verify the new default CUDA path and resident chain | Native exactness against pinned OpenFHE, lifecycle/failure checks, and the NTT/resident smoke scopes under all four sanitizers |
| 3 | Verify the legacy control build before timing | Native exactness and appropriate sanitizer smoke, with separate binary hashes and logs |
| 4 | Check Shoup and approximate basis conversion | Native integer and pinned OpenFHE comparisons, plus all four sanitizer tools |
| 5 | Compare legacy versus group twiddles on matched inputs | Raw paired resident forward/inverse samples first; public-call timings reported separately |
| 6 | Measure the resident operation chain and capture a selected profile | End-to-end chain versus an intentionally host-roundtripped chain; device timeline and hardware counters if permitted |

Use a small first timing block: N=4,096 and 65,536; K=3 and 40 towers;
30- and 60-bit primes. These choices give eight workloads per build. Use the
same thread layout, modular arithmetic, random-input seed, warmup count and
measurement count for both versions. Record GPU clock speeds. If temperature
changes affect the results, repeat with the versions run in the opposite order. Expand to 50/55-bit primes and
the remaining dimensions only after the required block is saved. The small
timing block does not replace the full native exactness matrix.

### Commands to prepare before provisioning

The compiler-only CI matrix builds pinned OpenFHE and both schedules for
sm_89 and sm_120, including the new native test executables, with no driver. For a local dry run that also links pinned OpenFHE, use a Linux
host with the toolkit installed:

```sh
python3 scripts/gpu_session.py --mode offline --compare-ntt \
  --architectures 120 --cuda-root /path/to/cuda \
  --openfhe-prefix /path/to/openfhe \
  --build-dir build/ntt-ab-offline --output-dir bench-results-ntt-ab-offline
```

On the rented GPU, confirm that the selected device is the RTX 5090
(compute capability 12.0) and that the driver and toolkit work before timing:

```sh
nvidia-smi --query-gpu=index,name,compute_cap,driver_version --format=csv
nvcc --version
compute-sanitizer --version
nsys --version
ncu --version
```

In a separate terminal, record clock and temperature readings while the
collector runs. Stop the clock log when the collector finishes.

```sh
nvidia-smi --query-gpu=timestamp,index,temperature.gpu,clocks.sm,clocks.mem,power.draw \
  --format=csv -l 2 > bench-results-ntt-ab-5090-clocks.csv
```

From a clean checkout of the verified revision, use fresh directories for
the native eight-shape sweep:

```sh
python3 scripts/gpu_session.py --mode gpu --compare-ntt --benchmark \
  --architectures 120 --cuda-root /path/to/cuda \
  --openfhe-prefix /path/to/openfhe \
  --build-dir build/ntt-ab-5090 --output-dir bench-results-ntt-ab-5090 \
  --benchmark-dimensions 4096 65536 --benchmark-towers 3 40 \
  --benchmark-prime-bits 30 60 --benchmark-scope gpu-chain \
  --benchmark-warmups 2 --benchmark-samples 10
```

Both child runs include the new arithmetic and conversion probes. A conversion
failure is retained with its test output and prevents timing that build.
Both child runs must pass their exactness and sanitizer gates before their
own timings. `comparison.json` only reports matched runs when the two
manifests agree on the environment and the actual workload records match
(including the roots and seed). The two sets of raw records stay separate.
The collector finishes the default schedule before starting the legacy one.
Temperature or clock changes between runs could therefore affect the
comparison, even with matching inputs.
Use `--ntt-schedule` to collect separately controlled repeats or a separate
profiler capture. The combined command takes fresh directories and does not
resume or profile. One failed child does not erase the other child's results.

### Capture the resident timeline and export evidence

After the default branch of the paired run passes, reuse its build and
N=65,536/K=3/60-bit vector. Run the timeline and hardware-counter captures
independently so a counter permission failure cannot cost the timeline.
Use fresh output names if you repeat a capture.

```sh
records=bench-results-ntt-ab-5090
binary=build/ntt-ab-5090/group_twiddle_ct/ntt_resident_bench
vector="$records/group_twiddle_ct/benchmark-vector-n65536-k3-b60.txt"
timeout 180 nsys profile --trace=cuda --sample=none --cpuctxsw=none \
  --output "$records/resident-timeline" "$binary" "$vector" 1 2 \
  > "$records/resident-nsys.log" 2>&1 || echo "$?" > "$records/resident-nsys.exit"
timeout 180 ncu --set basic --launch-count 8 \
  --export "$records/resident-counters" "$binary" "$vector" 1 2 \
  > "$records/resident-ncu.log" 2>&1 || echo "$?" > "$records/resident-ncu.exit"
```

Profile timings stay separate from benchmark samples. On the Linux host,
move the clock log into the record directory, then archive it:

```sh
mv "$records-clocks.csv" "$records/clocks.csv"
tar -czf "$records.tar.gz" "$records"
sha256sum "$records.tar.gz" > "$records.tar.gz.sha256"
```

Copy both files to the M1 Pro and verify before ending the rental:

```sh
shasum -a 256 -c bench-results-ntt-ab-5090.tar.gz.sha256
tar -tzf bench-results-ntt-ab-5090.tar.gz > /dev/null
```

## When to rent, and when to stop

Before renting, confirm that the code compiles for sm_120, the RTX 5090
architecture, without requiring a GPU driver. Set a rental time budget and
check that the provider allows the profiling tools and counters you need.

Run correctness before timing. If a required gate fails, save the failure
evidence and skip timing that build. If counters are unavailable,
record that limitation separately and continue independent checks or timings.
End the session after the planned work (or budget cutoff) once raw logs,
measurements and profiles are copied locally and their hashes verified.
Review results offline, then choose which NTT path to carry into GPU base
conversion. See [collector details](offline-test-tooling.md) and the
[integration roadmap](openfhe-integration-plan.md).
