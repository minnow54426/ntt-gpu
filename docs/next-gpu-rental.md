# Bundle the next GPU session

Use one prepared session to check answers, collect timings and export evidence.
The working card is **RTX 4080 SUPER**. A standard RTX 4080 is also supported;
record it separately. RTX 5090 validation remains a later requirement.

| Evidence | Current scope |
|---|---|
| November 24 Ada run | NTT, resident conversion and reference replay passed native checks; CUDA/NVTX tracing worked |
| New Hybrid adapters | Key switching, relinearization and rotations pass CPU/OpenFHE comparisons and compile for CUDA; native checks remain queued |
| Hardware counters | The November 24 AutoDL instance returned `ERR_NVGPUCTRPERM`; probe each new instance |
| D11/P2 | Still open; offline preparation does not complete native acceptance |

Keep results tied to their source revision and exact card. See the
[paired Ada analysis](paired-ntt-native-analysis.md#what-changed-in-the-november-24-ada-results)
for measured results. This page is the current procedure; older journals are
dated records.

The [November 28 handoff](journal/2026-11-28.md) freezes refactored source
`4fb89d7` with matching offline builds, checked inputs and a verified restore
and batch plan. Use that frozen checkout for this batch.

## Commands to prepare before provisioning

On a CPU machine, build the pinned OpenFHE tests, generate all eight
vector/reference sets, and prepare the rotation fixture:

```sh
bash scripts/openfhe_setup.sh
python3 scripts/benchmark_batch.py build/prepared-vectors \
  --dimensions 4096 65536 --towers 3 40 --prime-bits 30 60 \
  --conversion-bench build/openfhe-compat/ntt-build/ntt_resident_conversion_bench_host
build/openfhe-compat/ntt-build/ntt_openfhe_hybrid \
  --prepare-rotation build/rotation-benchmark-input.bin
```

Use fresh output paths. The vector and reference manifests record hashes.
Conversion benchmarks and their sanitizer smoke reuse those exact OpenFHE
arrays; standalone correctness tests also calculate references during their
run. The rotation fixture contains fixed test ciphertexts, evaluation keys
and expected outputs. Keep all of these files together.

Build CUDA and pinned OpenFHE on a compatible Linux CPU host before the paid
session when possible:

```sh
python3 scripts/gpu_rental_batch.py --mode offline --target rtx-4080-super \
  --cuda-root /path/to/cuda --openfhe-prefix /path/to/openfhe \
  --build-dir build/rental-offline --output-dir bench-results-rental-offline \
  --budget-minutes 120
```

`OFFLINE_READY` means compilation and host checks passed, with no GPU
execution. Use `--target rtx-4080` for that model or `--target rtx-5090` for
sm_120. Ada targets compile for sm_89. Existing invocations without a target
still default to RTX 5090, so specify the intended card explicitly.
The normal batch uses fresh builds. Advanced reuse of verified binaries belongs
to the [direct collector](offline-test-tooling.md#reuse-verified-offline-builds).

### Compare individual and hoisted CKKS rotations

The fixture covers N=65,536 with five and three active Q towers and shifts
`[1, -1, 2, -2]`. The runner compares four independent adapter calls with one
batch that reuses digit preparation. Each level uses two warmup pairs and
ten measured pairs, alternating which path runs first.

Timings include session setup, input/key transfers, computation and output
restoration. Correctness comparisons run outside the timer. The archive
retains raw pairs, the fixture hash and median paired speedup. Only CUDA
records count as native performance evidence.

## Run the queue as one batch

### Freeze the source and prepared inputs

Commit source changes, then create a handoff from that clean revision:

```sh
python3 scripts/prepare_rental_bundle.py create \
  --prepared-vectors build/prepared-vectors \
  --rotation-fixture build/rotation-benchmark-input.bin \
  --output-dir build/rental-handoff --target rtx-4080-super
```

The package includes a Git bundle, checked inputs and a portable verifier.
It contains source/data; build Linux binaries for the receiving image.
Optional `--verification-dir` records retain their original revision and
scope. They do not make binaries reusable under another commit or path.
Use a new handoff for later source changes; an earlier archive stays frozen.

Copy the archive and both sidecars to the receiving machine. Verify the
external checksum before extraction, then restore into a **new** checkout:

```sh
sha256sum -c rental-handoff.tar.gz.sha256
tar -xzf rental-handoff.tar.gz
python3 rental-handoff/handoff.py verify rental-handoff
python3 rental-handoff/handoff.py restore rental-handoff /path/to/ntt-gpu
```

The portable helper needs Python and Git. It verifies the complete inventory
and hashes before checkout. Restore refuses an existing destination, and
planning requires a clean checkout at the frozen source revision.

### TODO: verify AutoDL profiling access at the next rental

- [x] Checked the November 24 AutoDL RTX 4080 SUPER instance.

That instance allowed a real CUDA/NVTX trace and blocked SM/L2/DRAM counters.
Repeat these probes **before the long batch on each new instance**:

1. Save the instance identity, GPU, driver, CUDA, `nsys --version` and
   `ncu --version`.
2. Trace a small verified CUDA workload with CUDA/NVTX and CPU sampling off.
   Confirm actual kernel activity and its NVTX range in the report.
3. Attempt a small Nsight Compute capture with numeric SM, L2 and DRAM
   metrics. An installed tool or empty report does not establish access.
4. Save commands, logs, reports and permission errors. Record tracing and
   counters separately as confirmed, blocked or unverified.

The checkbox records the November 24 attempts, including the blocked result.
It does not confirm access on another machine or close the profiling gate.
Container root alone does not grant host counters; see
[NVIDIA's permission guidance](https://developer.nvidia.com/nvidia-development-tools-solutions-err_nvgpuctrperm-permission-issue-performance-counters).
Blocked counters do not prevent independent correctness, timing or working
timeline capture. The batch's later profiles do not replace this early probe.

### Preview and execute the prepared queue

After restoring, review the complete plan without running GPU work:

```sh
python3 rental-handoff/handoff.py plan rental-handoff /path/to/ntt-gpu \
  --cuda-root /usr/local/cuda \
  --openfhe-prefix /path/to/ntt-gpu/build/openfhe-compat/install
```

The plan validates all inputs and prints `gpu_command_for_later`. It includes
rotation timing, extended repeats and deferred local analysis. Set fresh,
disjoint `--build-dir` and `--output-dir` paths as needed.

After the profiling-access check below, run the planned command. The equivalent
full batch invocation from the clean checkout is:

```sh
python3 scripts/gpu_rental_batch.py --mode gpu \
  --target rtx-4080-super --device-index 0 \
  --cuda-root /usr/local/cuda \
  --openfhe-prefix build/openfhe-compat/install \
  --prepared-vectors /path/to/rental-handoff/prepared \
  --conversion-references /path/to/rental-handoff/prepared \
  --rotation-fixture /path/to/rental-handoff/rotation-benchmark.bin \
  --extended-probes --defer-analysis \
  --build-dir build/rental-next --output-dir bench-results-rental-next \
  --budget-minutes 120
```

The runner checks the exact GPU model and capability, then selects it by UUID.
Use the physical index from `nvidia-smi`. A required failure stops dependent
work and retains the evidence. The budget covers builds, checks, measurements
and profiles; it is a cutoff rather than a runtime estimate. Packaging continues
after cutoff. **The script does not stop provider billing.**

## One rental, ordered GPU work

| Order | Work | Retained evidence |
|---|---|---|
| 1 | Verify the card and profiling access | Device/tool identity, trace, counter results or permission errors |
| 2 | Check both schedules before timing | Native OpenFHE comparisons, resident/Hybrid checks and selected sanitizer scopes |
| 3 | Run the matched eight-workload matrix | 64 measurement jobs and 56 sanitizer commands across both schedules |
| 4 | Run controlled comparisons | Resident NTT repeat; 30/60-bit, 40-tower public-call repeats; individual/hoisted rotations |
| 5 | Capture operation profiles | Separate NTT/conversion captures and the 40-tower conversion timeline/SQLite export |
| 6 | Export and verify | Raw records, hashes, analysis inputs and archive |

`resident_cuda_hybrid_exact` checks digits, both key-switch contributions,
ModDown, ciphertext residues and decoded slots. It covers key switching,
relinearization, left/right rotation and a hoisted `[1, 0, -1]` batch at full
and reduced levels. The Barrett primitive has its own exactness and sanitizer
checks; it is not selected in timed NTT dispatch.

Each schedule has 32 matrix measurement jobs. Complete the same inputs,
primes, roots, seed and timing scopes before comparing versions. The default
matrix runs schedules sequentially; controlled repeats use
legacy/group/group/legacy order to expose drift.

### Capture the resident timeline and export evidence

The batch uses separate NVTX binaries for profiling. It labels setup,
warmups, samples, verification and cleanup, and attempts Nsight Systems and
Nsight Compute independently. With `--extended-probes`, it captures the
60-bit, 40-tower conversion case and exports SQLite for local inspection.

Attribute costs inside measured `ntt.sample` ranges. Whole-process totals can
include setup or verification outside the timers. Keep instrumented timings
separate from benchmark samples. Missing tools, blocked counters and empty
captures remain explicit limitations. See
[offline profiling preparation](offline-gpu-preparation.md#prepare-profiles-without-renting)
and [collector options](offline-test-tooling.md#optional-selected-profiling)
for individual experiments.

### Read and export the batch record

| File | Meaning |
|---|---|
| `batch.json` | Overall status, source/device identity and stage outcomes |
| `paired/` | Both schedules' exactness, sanitizer logs and raw matrix records |
| `controlled-repeat.json` | Ordered resident repeats and binary/log fingerprints |
| `ckks-rotation-pairs.log` | Paired rotation records; summaries are in `batch.json` |
| `clocks.csv` | Device telemetry when available |
| `SHA256SUMS` | Saved-file fingerprints |

The runner creates a sibling `.tar.gz`, checks every archived file, and writes
`.tar.gz.sha256` and `.tar.gz.verify.json`. A verified archive can contain a
failed run: read `batch.json` and both child reports before accepting results.
With `--defer-analysis`, copy and verify the evidence locally before releasing
the rental, then analyze the directory containing `paired/`:

```sh
shasum -a 256 -c bench-results-rental-next.tar.gz.sha256
tar -xzf bench-results-rental-next.tar.gz
python3 scripts/analyze_gpu_batch.py bench-results-rental-next \
  --output-dir build/rental-analysis
```

Completed extended repeats appear in `public-repeats.csv`; partial failures
stay in the raw archive and are excluded from comparisons. Preserve raw
records and each machine's identity rather than pooling unlike devices.

## When to rent, and when to stop

Rent after source, inputs and offline builds are ready. Confirm tool and
counter permissions before committing the long session. Run correctness
before timing, retain failures, and stop dependent work when a required gate
fails. Export and verify the archive before ending billing.

Choose the next optimization from the recorded operation costs. The
[roadmap](openfhe-integration-plan.md) owns phase status; this procedure does
not advance D11/P2.
