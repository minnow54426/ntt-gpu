# Bundle the next GPU session

**Status: the November 24 RTX 4080 SUPER diagnostic passed native checks
and timings.** Both schedules and the conversion replay passed. Nsight
Systems captured CUDA/NVTX activity, and the instance blocked Nsight Compute
counters. RTX 5090 hardware acceptance remains open. See the
[paired results](paired-ntt-native-analysis.md#what-changed-in-the-november-24-ada-results).

Use the RTX 4080 SUPER as the working baseline if the RTX 5090 is hard to
rent. A standard RTX 4080 can also run the same checks. Record each card
separately and compare speedups on the same card. Save the RTX 5090 run for
when one is available.

The [paired Ada analysis](paired-ntt-native-analysis.md) is complete on the
M1 Pro. The November 21 conversion benchmark adds both ModUp input formats,
independent ModDown, their resident composition and two host-boundary controls.

The November 24 native run covered resident ModUp/ModDown composition. The
newer Hybrid key-switching, relinearization and rotation adapters pass host
comparisons with OpenFHE; their CUDA path still needs the next native run.
Keep those source versions separate when interpreting the earlier results.

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

## TODO: verify AutoDL profiling access at the next rental

- [x] Checked Nsight access on the November 24 AutoDL RTX 4080 SUPER rental.

Nsight Systems captured a real CUDA kernel and its NVTX range. Nsight Compute
returned `ERR_NVGPUCTRPERM` when requesting SM, L2 and DRAM metrics. The
profiling gate remains open. Repeat the probe before the long batch on a new
rental, including the RTX 5090. See the
[November 24 rental notes](journal/2026-11-24.md#autodl-profiling-preflight-on-ada).

To complete this verification:

1. Record the provider, instance identity, exact GPU model, driver, CUDA toolkit,
   `nsys --version` and `ncu --version`.
2. Run a small correct CUDA workload under Nsight Systems with CUDA/NVTX
   tracing and CPU sampling disabled. Confirm that its report contains actual
   kernel activity and the expected NVTX labels.
3. Run a small kernel under Nsight Compute and confirm numeric SM, L2-cache
   and DRAM metrics. A version listing or an empty report does not establish
   counter access; record missing or unsupported metrics separately.
4. Save the commands, logs and report files in the rental evidence archive.
   Record tracing and counters separately as confirmed, blocked or unverified.
   Keep any `ERR_NVGPUCTRPERM` error and the provider's stated permissions.

Check off this TODO after attempting both captures and recording their results,
including a blocked result. Leave it pending if the probes were not attempted.
Blocked counters leave the hardware-profiling gate open; continue independent
correctness checks, benchmarks and any working timeline capture. Container
root alone does not grant host performance-counter access; see
[NVIDIA's permission guidance](https://developer.nvidia.com/nvidia-development-tools-solutions-err_nvgpuctrperm-permission-issue-performance-counters).

Run this preflight before the long batch. The later operation profiles do not
replace this early access check.

## One rental, ordered GPU work

| Order | GPU-only item | Evidence to bring home |
|---|---|---|
| 1 | Verify the selected GPU and repeat the AutoDL profiling-access probe | Device identity, commit, versions, a working CUDA/NVTX trace and confirmed or blocked SM/L2/DRAM counter access |
| 2 | Verify the new default CUDA path and resident chain | Native exactness against pinned OpenFHE, lifecycle/failure checks, and the NTT/resident smoke scopes under all four sanitizers |
| 3 | Verify the legacy control build before timing | Native exactness and appropriate sanitizer smoke, with separate binary hashes and logs |
| 4 | Check Shoup, the Barrett candidate and approximate basis conversion | Native integer and pinned OpenFHE comparisons, plus all four sanitizer tools |
| 5 | Compare legacy versus group twiddles on matched inputs | Raw paired resident forward/inverse samples first; public-call timings reported separately |
| 6 | Measure the resident operation chain and capture a selected profile | End-to-end chain versus an intentionally host-roundtripped chain; device timeline and hardware counters if permitted |

The current source also requires `resident_cuda_hybrid_exact` in the native
selection. It compares precomputed digits and both key-switch contributions
before and after ModDown against pinned OpenFHE. It also checks ciphertext
residues and decoded CKKS slots for key switching, relinearization, and
left and right rotations at full and reduced levels. The same fixture also
checks a hoisted batch containing a left shift, a zero shift, and a right
shift. Build a package from the updated source to include these checks.

### Compare individual and hoisted CKKS rotations

Prepare a fixture on the Mac using the host build with pinned OpenFHE.
Use a new output path.

```sh
build/pull-cpu/ntt_openfhe_hybrid --prepare-rotation build/rotation-benchmark-input.bin
```

Copy that fixture to the rental and pass
`--rotation-fixture /path/to/rotation-benchmark-input.bin` to the batch
command below. The fixture contains fresh test ciphertexts, rotation
evaluation keys, and expected OpenFHE outputs. Key generation and reference
evaluation happen during local preparation.

The runner compares four individual adapter calls with one hoisted call
using the same input and shifts `[1, -1, 2, -2]`. It measures both paths
for N=65,536 with five and three active Q towers. Each level has two warmup
pairs and ten measured pairs, alternating the order of the two paths. The
wall time includes session setup, input and key transfers, computation, and
output restoration. Correctness comparisons run outside the timed interval.
The archive retains the raw pairs and records the fixture hash and median
paired speedup. The native collector requires CUDA records.

## Run the queue as one batch

### Freeze the source and prepared inputs

Before the next rental, use the current CPU/OpenFHE build to prepare the eight
vector/reference sets below and the rotation fixture above. Commit the source
changes before packaging. The handoff pins the exact commit, includes its Git
bundle and a portable verifier, and checks every copied input against its hash.
It contains source and data; Linux binaries and dependencies still need to be
built for the receiving machine. Nothing in these commands rents or runs a GPU.

```sh
python3 scripts/prepare_rental_bundle.py create \
  --prepared-vectors build/prepared-vectors \
  --rotation-fixture build/rotation-benchmark-input.bin \
  --output-dir build/rental-handoff --target rtx-4080-super
```

Use fresh paths. Keep the resulting `.tar.gz`, `.tar.gz.sha256` and
`.tar.gz.verify.json` together. Verify the external checksum before extraction
on the receiving machine:

```sh
sha256sum -c rental-handoff.tar.gz.sha256
tar -xzf rental-handoff.tar.gz
python3 rental-handoff/handoff.py verify rental-handoff
python3 rental-handoff/handoff.py restore rental-handoff /absolute/path/to/ntt-gpu
python3 rental-handoff/handoff.py plan rental-handoff /absolute/path/to/ntt-gpu
```

The portable helper can verify the files before the source checkout exists.
Its `plan` command requires that checkout to match the frozen commit and be
clean. It validates all eight vector/reference sets and the rotation fixture,
then prints the full `gpu_command_for_later`, including the rotation benchmark
and extended probes. It does not execute that command. Supply `--cuda-root`,
`--openfhe-prefix`, `--build-dir` and `--output-dir` to `plan` as needed for the
receiving machine; build and result directories must be new and disjoint.

Optionally pass `--verification-dir` to retain offline reports. Those reports
keep their original revision and scope; packaging does not promote host
simulation or compilation into native evidence. Prebuilt binaries are not
included or made reusable under the bundle's revision.

The handoff status `OFFLINE_HANDOFF_READY` establishes source/data consistency,
not GPU correctness or timings. A successful compile is also separate from
native acceptance. Probe Nsight permissions before the long batch and preserve
blocked-counter errors in the evidence archive.

### Preview and execute the prepared queue

[`gpu_rental_batch.py`](../scripts/gpu_rental_batch.py) combines the queue
into one command. Pass the
[Mac-prepared directory](#commands-to-prepare-before-provisioning) to the plan
preview to verify every vector and OpenFHE reference in the matrix before
renting. This prints JSON without building, contacting a provider or running
GPU work:

```sh
python3 scripts/gpu_rental_batch.py --mode plan --target rtx-4080-super \
  --prepared-vectors build/prepared-vectors \
  --conversion-references build/prepared-vectors \
  --extended-probes --defer-analysis
```

Before renting, compile both schedules for the selected GPU and run their host tests:

```sh
python3 scripts/gpu_rental_batch.py --mode offline --target rtx-4080-super \
  --cuda-root /path/to/cuda --openfhe-prefix /path/to/openfhe \
  --build-dir build/rental-offline --output-dir bench-results-rental-offline \
  --budget-minutes 120
```

On the rental, use a clean checkout with pinned OpenFHE already installed.
Copy the complete prepared directory before starting the batch.
Select the physical index and exact model shown by `nvidia-smi`. The batch
verifies both the model and compute capability, then selects that GPU by UUID:

```sh
python3 scripts/gpu_rental_batch.py --mode gpu --target rtx-4080-super --device-index 0 \
  --cuda-root /usr/local/cuda \
  --openfhe-prefix build/openfhe-compat/install \
  --prepared-vectors /path/to/prepared-vectors \
  --conversion-references /path/to/prepared-vectors \
  --extended-probes --defer-analysis \
  --build-dir build/rental-4080-super --output-dir bench-results-rental-4080-super \
  --budget-minutes 120
```

Use `--target rtx-4080` for the standard RTX 4080 or `--target rtx-5090` for
an RTX 5090. The Ada targets build for `sm_89`; the RTX 5090 builds for
`sm_120` and remains the default for existing invocations without a target.
Use fresh build and output directories. This command reuses the collector
below, with eight workloads and four measurement scopes per schedule:
**64 measurement jobs and 56 sanitizer commands across both schedules**,
in addition to their native exactness tests. It then runs a representative
resident NTT case in legacy/group/group/legacy order, using five warmups and
50 samples per run. The small repeat controls run order; it does not replace
the workload sweep.

The runner checks prepared files before selecting a device. Paired benchmark
manifests record the input hashes with the source revision. Both conversion
profilers use the same reference as the measured batch. A conversion
reference error during profiling is recorded while the NTT captures remain
available.

The added Barrett scope checks its CUDA intrinsic implementation before any
timings. It runs under memcheck, racecheck, initcheck and synccheck in both
builds. The candidate is not enabled in the timed NTT dispatch; this batch
first validates the primitive. Current NTT timings therefore remain reference
arithmetic measurements. Older archives retain their original scope counts;
the analyzer requires the Barrett sanitizer records when that probe is
declared in a new run.

Clock and temperature logging spans the session. Four independent profiler
commands use separate NVTX profiling binaries to capture a timeline and
limited counters for resident NTT and ModUp/ModDown. Their fingerprints are
saved separately from benchmark binaries. Missing tools or blocked counters leave separate failure logs
and produce `GPU_BATCH_PASSED_WITH_LIMITATIONS` if the required checks and
measurement analysis pass. A profiler returning without a capture file is
also a recorded limitation. The timelines label setup, warmups, samples,
verification and cleanup; counter captures select `ntt-gpu@ntt.sample/` and
at most eight kernels. Inspect the operation nested inside a sample rather
than whole-process totals. Instrumented output is rejected by the benchmark
validator. [Compile and check these profiles offline](offline-gpu-preparation.md#prepare-profiles-without-renting).

The optional `--extended-probes` stage adds four public-call runs per prime
width at N=65,536 and K=40. Within each width, the schedule order is
legacy/group/group/legacy, with five warmups and 50 samples per run. The batch
keeps raw records for 30 and 60 bits separately. The analyzer checks each
completed repeat against its paired inputs and executable.

After the baseline profiles, the same option captures a CUDA/NVTX timeline
for the 60-bit K=40 conversion case and exports SQLite beside the report.
Use the database on the Mac to inspect CUDA API calls and copies within
measured `ntt.sample` ranges. [Nsight export guide](https://docs.nvidia.com/nsight-systems/AnalysisGuide/index.html#sqlite-schema-reference)

The budget covers preflight, compilation, checks, timings and profiles; it is
a cutoff, not a runtime estimate. Each collector command shares the remaining
budget. At expiry or a required failure, subsequent dependent GPU work stops.
Cleanup and evidence packaging continue. The script does not stop provider
billing: end the rental after exporting and verifying its evidence.

### Read and export the batch record

The batch preserves partial records on failure or interruption. Its files are:

| File | What it establishes |
|---|---|
| `batch.json` | Source/device preflight, overall status and separate profiler results |
| `paired/` | Both schedules' exactness, sanitizer logs, raw samples and matched workload index |
| `controlled-repeat.json` | Ordered repeats, binary fingerprints and raw-log hashes |
| `analysis/` | Revalidated operation comparisons and separate repeat CSV |
| `clocks.csv` | Device telemetry, when available |
| `SHA256SUMS` | Fingerprints of the saved evidence files |

After stopping telemetry, the script creates a sibling `.tar.gz` archive,
checks every archived file against its source bytes, and writes
`.tar.gz.sha256` and `.tar.gz.verify.json`. It performs no extraction and
refuses to overwrite previous records. `OFFLINE_READY` explicitly means no
GPU execution. A verified archive may contain a failed run: read `batch.json`
and both child reports before accepting its results.

Copy the archive and checksum to the local machine, then verify:

```sh
shasum -a 256 -c bench-results-rental-4080-super.tar.gz.sha256
```

With `--defer-analysis`, run the analysis on the Mac after verifying and
extracting the archive. Pass the extracted directory that contains `paired/`.

```sh
python3 scripts/analyze_gpu_batch.py /path/to/bench-results-rental-4080-super \
  --output-dir build/ada-rental-analysis
```

Completed extended repeats appear in `public-repeats.csv`. Partial repeats
stay in the raw archive with their failure stage and are excluded from the CSV.

The commands below remain available for a manually controlled session.

Use a small first timing block: N=4,096 and 65,536; K=3 and 40 towers;
30- and 60-bit primes. These choices give eight workloads per build. Use the
same thread layout, modular arithmetic, random-input seed, warmup count and
measurement count for both versions. Record GPU clock speeds. If temperature
changes affect the results, repeat with the versions run in the opposite order. Expand to 50/55-bit primes and
the remaining dimensions only after the required block is saved. The small
timing block does not replace the full native exactness matrix.

### Commands to prepare before provisioning

Generate the benchmark inputs and sanitizer sample on the M1 Pro before
starting the billing window. Copy the whole directory, including its checksums,
to the GPU host.

```sh
python3 scripts/benchmark_batch.py build/prepared-vectors \
  --dimensions 4096 65536 --towers 3 40 --prime-bits 30 60 \
  --conversion-bench build/openfhe-compat/ntt-build/ntt_resident_conversion_bench_host
```

Run standalone CPU tests, OpenFHE CPU benchmarks and results analysis locally.
Build Linux CUDA binaries and dependencies on a compatible CPU-only Linux host
before provisioning when that host is available. The native tests currently
compute some OpenFHE reference arrays in the same process as CUDA, and the
benchmark includes CPU orchestration where it is part of the measured operation.
The optional conversion reference export moves the six conversion benchmark
operations' inputs and OpenFHE reference results to the Mac. The benchmark
and its sanitizer smoke can load the exported arrays and compare all
results exactly. Separate native correctness tests still calculate their
OpenFHE references during the tests.

The compiler-only CI matrix builds pinned OpenFHE and both schedules for
sm_89 and sm_120, including the new native test executables, with no driver. For a local dry run that also links pinned OpenFHE, use a Linux
host with the toolkit installed:

```sh
python3 scripts/gpu_session.py --mode offline --compare-ntt \
  --architectures 120 --cuda-root /path/to/cuda \
  --openfhe-prefix /path/to/openfhe \
  --build-dir build/ntt-ab-5090 --output-dir bench-results-ntt-ab-offline
```

To reuse these builds, stage the clean checkout, build directories, OpenFHE
installation, and offline records on the rental at the same absolute paths.
Use a compatible Linux host and CUDA runtime. Add
`--offline-records bench-results-ntt-ab-offline` to the native command. The
collector verifies the commit, paths, architecture, schedule, executables,
CTest definition, generated CTest vectors, and OpenFHE libraries. It then
runs the native tests without configuring, compiling, or regenerating those
CTest vectors. Keep a copy of the offline reports with the result archive.

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
  --prepared-vectors /path/to/prepared-vectors \
  --conversion-references /path/to/prepared-vectors \
  --benchmark-prime-bits 30 60 --benchmark-scope gpu-conversion \
  --benchmark-warmups 2 --benchmark-samples 10
```

This selection records 32 measurement jobs per build, including eight
conversion jobs. Every conversion sample is checked against pinned OpenFHE output. The collector
checks the reference bundle hashes and records them in its manifest.
Fixed NTT session setup is excluded. Per-call conversion parameter construction
and upload, allocation and host orchestration are included. The final checks
for resident operations download outside their timers. The two host controls
include their boundary transfers.

The conversion selection also runs its benchmark at N=4,096/K=40/60-bit,
one warmup and two samples, under all four sanitizer tools. Those instrumented
times belong to correctness logs. The paired timing sweep starts afterward.

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
binary=build/ntt-ab-5090/group_twiddle_ct/ntt_resident_profile
vector="$records/group_twiddle_ct/benchmark-vector-n65536-k3-b60.txt"
timeout 180 nsys profile --trace=cuda,nvtx --sample=none --cpuctxsw=none \
  --output "$records/resident-timeline" "$binary" "$vector" 1 2 \
  > "$records/resident-nsys.log" 2>&1 || echo "$?" > "$records/resident-nsys.exit"
timeout 180 ncu --nvtx --nvtx-include 'ntt-gpu@ntt.sample/' --set basic --launch-count 8 \
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

Before renting, confirm that the code compiles for the selected card without
requiring a GPU driver. Set a rental time budget and
check that the provider allows the profiling tools and counters you need.

Run correctness before timing. If a required gate fails, save the failure
evidence and skip timing that build. If counters are unavailable,
record that limitation separately and continue independent checks or timings.
End the session after the planned work (or budget cutoff) once raw logs,
measurements and profiles are copied locally and their hashes verified.
Review results offline, then choose which NTT path to carry into GPU base
conversion. See [collector details](offline-test-tooling.md) and the
[integration roadmap](openfhe-integration-plan.md).
