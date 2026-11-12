# ntt-gpu

Exact GPU kernels for CKKS: NTT first, then base conversion, Hybrid key
switching, and OpenFHE integration. The target workload is N=65,536 on RTX 5090,
with RTX 4090 as the compatibility control.

**Current state:** the CPU reference, Python oracle, portable GPU-schedule
simulator, and OpenFHE transform/representation/Hybrid graph gates are implemented.
CUDA binaries compile for `sm_89` and `sm_120` with a compiler-only toolkit;
execution on a real GPU remains the next hardware gate. CPU simulation does
not validate GPU barriers or launch behavior.

## Build & test

Requires CMake, a C++20 compiler, and Python 3. CUDA is optional and detected
automatically.

```sh
cmake -S . -B build -DNTT_ENABLE_CUDA=OFF
cmake --build build -j
ctest --test-dir build --output-on-failure
```

The fifteen portable checks cover vector generation at N=64/1024/65536, unit
tests, the independent Python oracle, the shared device schedule, column and
tile index/twiddle proofs in both directions, arithmetic exactness and
full-size adversarial transforms, backend metadata, host CUDA preflight, session
failure reporting, and the N=8 debug example. Small fixtures keep the
quadratic oracle and schoolbook multiplication practical.

With nvcc, three GPU gates compare CUDA with the same oracle and CPU
reference. Enabling OpenFHE adds a fourth direct CUDA/OpenFHE gate. A separate
compiled-wrapper constructor check requires no GPU. Array ordering,
forward/inverse results, round trips, and polynomial
multiplication must agree exactly. See [run and debug](docs/vscode-debugging.md)
to step through the N=8 example.

### Exact OpenFHE compatibility

The optional gate pins OpenFHE 1.6.1 at
`b59e1f291965f14fbee5172cd83a20f5b856204f`:

```sh
bash scripts/openfhe_setup.sh
```

It compares N=65536 `NativePoly` coefficient/evaluation conversions with our
reference, including bit-reversed evaluation order. Its `FirstPrime(50, ...)`
and `LastPrime(59, ...)` fixtures are 51- and 59-bit primes. Cases include zero,
maximum, ramp, random coefficients, and independent random inverse inputs.

A second gate, `openfhe_representation_contract`, checks exact evaluation
indices, format-changing APIs, root selection, and three-tower DCRT packing at
50/55/60-bit primes. Read the [D7 contract](docs/openfhe-representation-contract.md)
for the integration boundary and remaining scope.

A third gate, `openfhe_hybrid_graph`, traces the exact Hybrid digit extension,
key MAC, ModDown, and ciphertext wrapper graph at N=65536. It checks full and
reduced levels with deterministic diagnostic keys and actual generated keys,
using independent unreduced-CRT formulas. It requires the PKE library and the
64-bit, non-reduced-noise build; see the [D8 graph](docs/openfhe-hybrid-operation-graph.md).

The normal CPU build also runs `backend_contract_check`, which checks D9's
buffer/layout metadata, stream dependencies, cache identities and preflight
fallback policy. The [D9 backend proposal](docs/device-backend-contract.md)
includes compile-checked abstract interfaces; it does not implement a CUDA
runtime backend or OpenFHE adapter.

D10 accepts the behavioral design in [ADR 0001](docs/architecture-decision.md)
and tracks the remaining native GPU/runtime work in the
[implementation backlog](docs/implementation-backlog.md). P1 is complete;
the next task is D11's GPU environment record, not a claimed CUDA result.

To reuse an installation, configure with `-DNTT_ENABLE_OPENFHE_COMPAT=ON`
and set `CMAKE_PREFIX_PATH` to its install prefix. OpenFHE is not required
for the standalone build.

### First session on a rented GPU

Finish the [offline preparation](docs/offline-gpu-preparation.md) first. On the
GPU image, use the selected clean checkout and an existing pinned OpenFHE prefix:

```sh
bash scripts/autodl_setup.sh
```

The script records hardware/toolchain details, requires CUDA and OpenFHE test
registration, builds, runs exact tests and sanitizer smoke checks, and collects
the 46-bit transfer-inclusive baseline benchmark. Each session writes logs and
`report.json` under an ignored `bench-results-*` directory. Any failed stage
returns nonzero. It does not update source, install packages or change drivers.
Select paths with `NTT_REPO_DIR`, `NTT_CUDA_ROOT`, and `NTT_OPENFHE_PREFIX`;
`NTT_CUDA_ARCH` defaults to `native`. See the preparation guide for full commands
and the remaining P2 gates.

The default build targets `sm_89` and, with CUDA 12.8+, `sm_120`. For one known
GPU, use `-DCMAKE_CUDA_ARCHITECTURES=native` (CMake 3.24+).

## Design

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

Stage bodies in [`dev_core.hpp`](include/ntt/dev_core.hpp) are compiled for
both CPU simulation and CUDA. The current `__uint128_t` remainder arithmetic
is a correctness baseline. Shoup/Barrett reduction, coalesced layouts, and
register staging are planned optimizations, subject to exact checks and profiling.

Use the [work-distribution explorer](docs/ntt-65536-work-distribution.html)
to inspect blocks, threads, pairs, and twiddle exponents.

## Roadmap

| Deliverable | State |
|---|---|
| CPU reference, oracle, schedule simulation | Complete |
| CUDA kernels and automatic GPU test registration | Implemented; hardware validation pending |
| Named-device baseline and tuning | Planned |
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
