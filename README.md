# ntt-gpu

This project builds exact GPU polynomial operations for CKKS, an encryption
scheme that supports computation on encrypted data. Start with the Number
Theoretic Transform (NTT), then add base conversion, Hybrid key switching and
integration with the OpenFHE cryptography library.

**What works:** the CPU reference and tested CUDA baseline agree with OpenFHE.
A [temporary Ada GPU run](docs/batch40-native-analysis.md) passed its selected
native checks. The newer [resident-chain prototype](docs/resident-operation-chain.md)
passes host tests and compiles for CUDA; it has not run on a GPU.
**Next:** validate the current source on RTX 5090. RTX 4090 remains the planned
control. The main workload has 65,536 coefficients.

## Build & test

Requires CMake, a C++20 compiler, and Python 3. CUDA is optional and detected
automatically.

```sh
cmake -S . -B build -DNTT_ENABLE_CUDA=OFF
cmake --build build -j
ctest --test-dir build --output-on-failure
```

This runs arithmetic, transform, schedule, ownership and reporting checks on
CPU. Small fixtures keep the independent quadratic oracle practical; full-size
fixtures exercise N=65536. To follow one butterfly, use the
[N=8 debugging example](docs/vscode-debugging.md).

CUDA builds register additional tests labelled `gpu`. Run those on actual
hardware; a passing CPU simulation or CUDA compilation cannot validate device
barriers, driver failures or GPU speed. The [preparation guide](docs/offline-gpu-preparation.md)
lists the current native checks.

### Exact OpenFHE compatibility

The optional gate pins OpenFHE 1.6.1 at
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

Hybrid tests require OpenFHE PKE, NativeInteger 64 and reduced-noise OFF.
Agreement with approximate CKKS decryption alone is insufficient: the arrays,
ordering and roots must match exactly. Architecture decisions and remaining
runtime work are tracked in the [ADR](docs/architecture-decision.md) and
[backlog](docs/implementation-backlog.md).

To reuse an installation, configure with `-DNTT_ENABLE_OPENFHE_COMPAT=ON`
and set `CMAKE_PREFIX_PATH` to its install prefix. OpenFHE is not required
for the standalone build.

### First session on a rented GPU

Finish the [offline preparation](docs/offline-gpu-preparation.md) first. On the
GPU image, use the selected clean checkout and an existing pinned OpenFHE prefix:

```sh
bash scripts/autodl_setup.sh
```

The helper records the environment, builds, runs exactness and sanitizer checks,
then collects 280 GPU measurement jobs. The matrix spans N=4096–65536,
30/50/55/60-bit primes and 1–40 RNS towers. A tower is one copy of the polynomial
modulo one prime. Public-call and device-event timings stay separate.

Use `NTT_REPO_DIR`, `NTT_CUDA_ROOT` and `NTT_OPENFHE_PREFIX` for machine-specific
paths. `--resume --output-dir <record>` reruns checks and reuses only matching,
hash-verified measurements. `--profile` requests Nsight captures. The helper
does not pull source, install dependencies or change drivers. Failed stages
return nonzero; retain the complete record directory. See the
[collector guide](docs/offline-test-tooling.md) for timing scopes and options.

Collect the matching 140-job CPU baseline locally without a CUDA installation:

```sh
python3 scripts/gpu_session.py --mode cpu \
  --openfhe-prefix build/openfhe-compat/install \
  --benchmark --output-dir bench-results-local-cpu
```

Match workload and repetition options across CPU and GPU sessions. Their
machine/build identities remain separate. Direct GPU collector commands default
to the combined 420-job selection; choose `--benchmark-scope gpu` for GPU-only
work. Changing modes or scopes requires a new record directory.

The default build targets `sm_89` and, with CUDA 12.8+, `sm_120`. For one known
GPU, use `-DCMAKE_CUDA_ARCHITECTURES=native` (CMake 3.24+).

Default CUDA uses one OpenFHE-style group twiddle per butterfly, with the same
two-pass thread mapping. Configure a separate build with
`-DNTT_CUDA_LEGACY_BUTTERFLY=ON` for the old two-multiply baseline.
GPU JSONL records identify the choice as `group_twiddle_ct` or `legacy_dif`.
The previous native benchmark measured the legacy path, not the new default.

## Design

The [resident operation-chain prototype](docs/resident-operation-chain.md) adds
immutable polynomial handles, one explicit stream, owned host staging and a
bounded failure-aware journal. Its host transport passes exact OpenFHE tests;
the CUDA adapter is compiled but not executed here. This experimental API does
not replace the synchronous baseline or enable the production OpenFHE backend.

Both backends use the same negacyclic transform convention:

```text
psi is a primitive 2N-th root modulo q; psi^N = -1
forward: A_j = sum_i a_i * psi^((2j+1)*i) mod q
input: natural coefficient order; output: bit-reversed evaluation order
```

The CUDA schedule splits at 256. Pass 1 gathers 256 strided columns; pass 2
loads 256 contiguous rows. At N=65536, each pass handles eight stages in shared
memory. One forward uses two launches and four logical array movements.
The strided pass is uncoalesced, so logical bytes are not measured DRAM traffic.

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

## License

TBD before accepting outside contributions.
