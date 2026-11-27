# ntt-gpu

This project builds exact GPU polynomial operations for CKKS, an encryption
scheme that supports computation on encrypted data. Start with the Number
Theoretic Transform (NTT), which makes polynomial multiplication easier by
evaluating at special modular points. Next add base conversion, which changes
the set of primes representing each coefficient, and Hybrid key switching,
which uses these operations to change a ciphertext's key representation.
Finally connect the GPU implementation to the OpenFHE cryptography library.

**What works:** the CPU reference and recorded Ada GPU runs agree with pinned
OpenFHE. The [November 24 run](docs/paired-ntt-native-analysis.md#what-changed-in-the-november-24-ada-results)
also checked resident ModUp/ModDown. The newer Hybrid adapters pass CPU checks
and compile for CUDA; their native validation remains queued.
**Next:** use RTX 4080 SUPER for the prepared batch and retain RTX 5090 for
target validation. The main workload has 65,536 coefficients.

## Build & test

Requires CMake, a C++20 compiler, and Python 3. CUDA is optional and detected
automatically.

```sh
cmake -S . -B build -DNTT_ENABLE_CUDA=OFF
cmake --build build -j
ctest --test-dir build --output-on-failure
```

These commands build the program and run its CPU tests. The tests check the
arithmetic, transform, assignment of work, buffer lifetimes and saved reports.
A separate, slower calculation checks small examples independently. Other
tests use the full N=65536 size. To follow one butterfly, use the
[N=8 debugging example](docs/vscode-debugging.md).

CUDA builds register additional tests labelled `gpu`. Run those on actual
hardware; a passing CPU simulation or CUDA compilation cannot validate device
barriers, driver failures or GPU speed. The [preparation guide](docs/offline-gpu-preparation.md)
lists the current native checks.

### Exact OpenFHE compatibility

The optional compatibility test uses this exact OpenFHE 1.6.1 revision:
[`b59e1f291965f14fbee5172cd83a20f5b856204f`](https://github.com/openfheorg/openfhe-development/tree/b59e1f291965f14fbee5172cd83a20f5b856204f):

```sh
bash scripts/openfhe_setup.sh
```

The optional CPU suite compares complete arrays with real OpenFHE operations:

| Check | Read more |
|---|---|
| Forward and independent inverse through [`NativePoly::SwitchFormat`](https://github.com/openfheorg/openfhe-development/blob/b59e1f291965f14fbee5172cd83a20f5b856204f/src/core/include/lattice/hal/default/poly-impl.h#L454-L474) | [Representation, roots and tower packing](docs/openfhe-representation-contract.md) |
| Hybrid digit extension, key multiplication and reduction | [Exact operation graph](docs/openfhe-hybrid-operation-graph.md) |
| Buffer metadata and resident-chain host behavior | [Backend contract](docs/device-backend-contract.md) and [prototype](docs/resident-operation-chain.md) |

For the Hybrid tests, enable OpenFHE's public-key encryption module (PKE),
select 64-bit native integers, and turn reduced-noise mode OFF.
CKKS decryption returns approximate numbers, so a plausible decrypted answer
could hide an array mismatch. These tests require the same values, positions
and roots. Design decisions and remaining implementation work appear in the
[architecture decision record (ADR)](docs/architecture-decision.md) and
[backlog](docs/implementation-backlog.md).

To reuse an installation, configure with `-DNTT_ENABLE_OPENFHE_COMPAT=ON`
and set `CMAKE_PREFIX_PATH` to its install prefix. OpenFHE is not required
for the standalone build.

### First session on a rented GPU

Follow the [rental runbook](docs/next-gpu-rental.md): prepare inputs locally,
freeze the clean source, check profiler permissions on the selected instance,
then run correctness, sanitizers and measurements as one bounded batch.
The runbook owns the current matrix and commands. The
[collector guide](docs/offline-test-tooling.md) describes individual experiments,
resume rules and timing scopes.

For a separate OpenFHE CPU baseline, without CUDA:

```sh
python3 scripts/gpu_session.py --mode cpu \
  --openfhe-prefix build/openfhe-compat/install \
  --benchmark --output-dir bench-results-local-cpu
```

Match inputs and repetitions before comparing CPU/GPU results. Each report
keeps its own machine, build and timing boundary. Use a new record directory
when changing the mode or scope.

The default build targets `sm_89` and, with CUDA 12.8+, `sm_120`. For one known
GPU, use `-DCMAKE_CUDA_ARCHITECTURES=native` (CMake 3.24+).

Default CUDA uses one OpenFHE-style group twiddle per butterfly, with the same
two-pass thread mapping. Configure a separate build with
`-DNTT_CUDA_LEGACY_BUTTERFLY=ON` for the old two-multiply baseline.
GPU JSONL records identify the choice as `group_twiddle_ct` or `legacy_dif`.
The [November 19 paired run](docs/journal/2026-11-19.md#native-gpu-batch-completed)
measured both schedules on a server reporting RTX 4080 SUPER / compute 8.9.

## Design

The [resident operation-chain prototype](docs/resident-operation-chain.md) adds
polynomial handles whose inputs cannot be changed, one ordered GPU work queue
(a stream), and temporary CPU buffers owned by the session. A limited operation
log retains resources needed by unfinished work. Its host transport passes exact
OpenFHE comparisons. Recorded Ada runs checked the earlier CUDA paths; updated
runtime and Hybrid code needs another native run. Production integration remains
planned.

The [resident basis-conversion path](docs/basis-conversion-preparation.md#keep-modup-and-moddown-in-the-resident-chain)
composes ModUp and ModDown without intermediate downloads and has recorded
native Ada evidence. The [opt-in Hybrid adapters](include/ntt/openfhe_hybrid_adapter.hpp)
add transactional key switching, relinearization and rotations. Production
integration, native validation of the newer adapters and tuning remain open.

Both backends use the same negacyclic transform convention:

```text
psi is a primitive 2N-th root modulo q; psi^N = -1
forward: A_j = sum_i a_i * psi^((2j+1)*i) mod q
input: natural coefficient order; output: bit-reversed evaluation order
```

The CUDA schedule views each tower as a 256×256 array. Pass 1 gathers columns,
whose elements are 256 positions apart; pass 2 loads rows of adjacent elements.
Each pass handles eight stages using memory shared by a thread block.
One forward transform launches two kernels and reads/writes the full array
twice. Column accesses do not combine efficiently into memory transactions,
so counting array bytes alone does not measure traffic to GPU memory.

Stage bodies in [`transform_stage.hpp`](include/ntt/transform_stage.hpp) are compiled for
both CPU simulation and CUDA. The current `__uint128_t` remainder arithmetic
is a correctness baseline. Shoup/Barrett reduction, coalesced layouts, and
register staging are planned optimizations, subject to exact checks and profiling.

Use the [work-distribution explorer](docs/ntt-65536-work-distribution.html)
to inspect blocks, threads, pairs, and twiddle exponents.

## Roadmap

| Deliverable | State |
|---|---|
| CPU reference, oracle, schedule simulation | Complete |
| CUDA kernels and automatic GPU test registration | Native Ada diagnostic passed; RTX 5090 validation pending |
| Named-device baseline and tuning | Initial Ada sample collected; full profiling/tuning pending |
| Base conversion and Hybrid key switching | Planned |
| OpenFHE integration with CPU fallback | Planned |

The v0.1 target is 2027-01-31; v0.2 key-switch/automorphism work targets
2027-04-30. See the [tracked integration plan](docs/openfhe-integration-plan.md)
for working-day gates and the [benchmark plan](docs/benchmark-tuning-plan.md)
for measurements. Performance claims require named hardware and matching timing
boundaries.

## Docs

Start with [NTT and butterflies](docs/butterfly-explained.html), then
[CPU to CUDA](docs/gpu-ntt-explained.html). The [implementation survey](docs/gpu-ntt-survey.md)
and [GPU comparison](docs/gpu-selection-survey.md) explain the design choices.

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r docs/requirements.txt
.venv/bin/python scripts/build_docs.py --check
python3 -m http.server 8000 --bind 127.0.0.1 --directory build/docs-site
```

Open `http://127.0.0.1:8000/`. See [writing and publishing](docs/README.md)
for editing, journals, and GitHub Pages export.

## Provenance

Independent clean-room implementation. No code, data, or artifacts from past
or present employers. Development history is recorded in git.

Separate profiling executables use pinned [NVIDIA NVTX C API headers](third_party/nvtx/README.md),
distributed under their upstream [Apache 2.0 license with LLVM exceptions](third_party/nvtx/LICENSE.txt).
The ordinary benchmark executables do not enable this instrumentation.

## License

TBD before accepting outside contributions.
