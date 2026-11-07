# ntt-gpu

GPU-accelerated RNS NTT, key-switching, and automorphism kernels for FHE (and, by shared
core operation, ZK proving). Target: best-in-class, honestly-benchmarked CUDA kernels.

**Status:** Layer 1 (CPU `ref` backend) complete and cross-verified — 2026-10-29.
Layer 2 written and Mac-proven (schedule simulator green); GPU gates light up
automatically wherever nvcc exists.

## Roadmap

- **v0.1 (2027-01-31):** correct, tested RNS NTT CUDA kernel + benchmark table
  (RTX 4090 + RTX 5090), e.g. "2^16 coefficients, 40-bit limbs: X ms, N× vs OpenFHE CPU".
  - [x] Layer 1 — CPU reference negacyclic NTT + Python big-int oracle (2026-10-29)
  - [ ] Layer 2 — CUDA kernel, validated on the same vectors
    - [x] 2a — the GPU stage schedule (`include/ntt/dev_core.hpp`), proven on CPU
      by a portable simulator gate (`device_schedule_check`) at n = 64 / 1024 / 4096
    - [x] 2b — the kernels themselves (`src/cuda/ntt_kernels.cu`), thin wrappers
      around the proven schedule; GPU gates register automatically under nvcc
    - [ ] 2c — first run on a rented GPU (see below), bench table recorded
    - [ ] 2d — systematic benchmark and autotuning matrix across GPU launch
      parameters and NTT implementations; select winners per ring size, RNS
      tower count, and latency/throughput objective
- **v0.2 (2027-04-30):** key-switching / automorphism kernels; first upstream PRs.

Benchmarking and tuning are part of the implementation, not a final reporting
step. The primary target is RTX 5090 (`sm_120`), with RTX 4090 (`sm_89`) as the
cross-architecture control. The matrix includes `N = 4096..65536`, realistic
CKKS modulus widths and tower counts, threads/block, values/thread, radix,
register and shared-memory use, split/tile shape, batch size, streams, twiddle
representation, and transfer policy. It compares the correctness baseline,
optimized two-pass variants, radix-4/8 register staging, and a coalesced
Four-Step/transpose path. Kernel-only and end-to-end results are reported
separately. See [`docs/benchmark-tuning-plan.md`](docs/benchmark-tuning-plan.md).

## Build & test

No dependencies beyond CMake, a C++20 compiler, and Python 3 (stdlib only).
Without a CUDA toolkit (e.g. the dev Mac):

```sh
cmake -S . -B build && cmake --build build -j && ctest --test-dir build --output-on-failure
```

Six gates: vector generation ×3 sizes (RNS primes + 2N-th roots of unity at
n = 64/1024/4096), unit tests (modular arithmetic, bit-reversal, forward/inverse
roundtrip, NTT-multiply vs O(N²) schoolbook mod x^N+1), the cross-language
oracle check — the C++ fast transform must equal a definitional naive O(N²)
big-integer transform in Python, limb by limb, with position-wise CRT
reconstruction over all limbs — and the device-schedule check: the simulator
(`src/device_sim.cpp`) runs the exact stage bodies the CUDA kernels run and
must match the CPU reference array-for-array, bit-reversed order included.

For a ready-to-run macOS VS Code setup, including an `N=8` example and LLDB
single-stepping instructions, see
[`docs/vscode-debugging.md`](docs/vscode-debugging.md).

With nvcc present, CMake auto-detects it and adds `ntt_cuda_cli` plus three
GPU gates, unchanged commands otherwise: `cuda_oracle_crosscheck` (the same
Python oracle against the CUDA binary — its output is byte-identical to
`ntt_cli`), `cuda_matches_ref` (byte-compare of both binaries' transform
output at n = 1024 and 4096; valid at any size, use a 2^18 fixture freely),
and `cuda_verify` (in-process GPU-vs-ref forward/roundtrip/multiply on the
fixtures plus a generated n = 65536 case).

### Exact OpenFHE compatibility

The OpenFHE gate pins **OpenFHE 1.6.1** at commit
`b59e1f291965f14fbee5172cd83a20f5b856204f`, builds it under the ignored
`build/` tree, and compares the public `NativePoly` coefficient/evaluation
conversion against `NegacyclicNtt` element for element:

```sh
bash scripts/openfhe_setup.sh
```

`openfhe_exact_compat` covers `N = 4096, 8192, 16384, 32768, 65536`, two
OpenFHE-selected CKKS-sized modulus widths (50 and 59 bits), zero, maximum,
ramp, and deterministic-random coefficient inputs, plus independent random
evaluation-domain inputs for inverse transforms. Equality includes OpenFHE's
exact bit-reversed evaluation ordering; this is not a decrypt-tolerance test.
The CUDA gates already require GPU output to equal the same CPU reference, so
the two exact gates compose to OpenFHE == reference == CUDA on GPU hardware.

The ordinary dependency-free build remains unchanged. To reuse an existing
OpenFHE installation instead of the setup script, configure with
`-DNTT_ENABLE_OPENFHE_COMPAT=ON` and point `CMAKE_PREFIX_PATH` at its install
prefix; CMake requires version 1.6.1 exactly.

Architecture default: sm_89 + sm_120 when the toolkit is ≥ 12.8, else sm_89
(the build embeds compute_89 PTX, which a 5090 driver JITs) — one build runs
on both rentable cards. Override with `-DCMAKE_CUDA_ARCHITECTURES=native`
(CMake ≥ 3.24) on a single known box.

### First session on a rented GPU

```sh
bash scripts/autodl_setup.sh          # or: NTT_REPO_DIR=... NTT_CUDA_ARCH=...
```

Idempotent one-shot: finds nvcc (PATH or `/usr/local/cuda/bin`), clone-or-pull,
configure + build, run every gate, run the bench sweep, and write `VERDICT.txt`
(GPU name, driver, toolkit, arch sanity check, per-gate results, bench table).
Exit code reflects the gates. Bench numbers are attached to that hardware
header or they don't get quoted.

## Design

Two backends behind one interface:

- `ref` — CPU reference (correctness is defined here; big-int-checked test vectors)
- `cuda` — GPU kernel (validated against `ref` on identical vectors)

The CUDA port's index math lives once in `include/ntt/dev_core.hpp`, compiled
by the Mac simulator and by the kernels alike, so schedule and device cannot
drift; the reference arithmetic (`mod_arith.hpp`) is `__host__ __device__`, so
the GPU runs literally the reference `mul_mod`. The schedule is a two-pass
split at 256: pass 1 over 256 strided groups (the far stages, whose pair
distance is a multiple of 256, pair values sharing their offset mod 256),
pass 2 over contiguous 256-tiles (the near stages). All 16 stages at
N = 65,536 run on shared-memory counters; the "mix" between passes is pure
data movement (scatter strided, gather contiguous), so VRAM traffic is 4
round-trips per value instead of the 14 a global-memory phase would cost —
and one forward is 2 kernel launches. v1 uses the `__uint128_t`
division lowering on device — emulated there, and honestly slow; the bench
table quantifies exactly that pre-Barrett gap, and Barrett/Montgomery lands
later behind the same signature with a dedicated cross-check. The simulator
is serial by nature: it proves the math, not barriers or launch configs —
the GPU gates exist for that residue.

Benchmarks are only ever reported from the `cuda` backend on named hardware.

Transform convention (see `include/ntt/ntt.hpp`): ψ a primitive 2N-th root of
unity mod q, `forward: A_j = Σ_i a_i·ψ^((2j+1)·i) mod q` — evaluation at the
odd powers of ψ, where x^N+1 splits into linear factors.

## Docs

- [`docs/butterfly-explained.html`](docs/butterfly-explained.html) — an interactive,
  dependency-free beginner's explainer: why the NTT is built from butterfly units,
  from the one-line definition through the negacyclic fold rule to the full N=8
  network (open in any browser).
- [`docs/gpu-ntt-explained.html`](docs/gpu-ntt-explained.html) — the GPU port
  explained from zero: the machine's concepts and real parameters (threads,
  blocks, SMs, the 4090/5090 numbers), then how this repo splits the NTT into
  strided groups and contiguous tiles so every stage runs on a shared-memory
  counter.
- [`docs/gpu-ntt-survey.md`](docs/gpu-ntt-survey.md) — primary-source survey of
  GPU NTT and CKKS implementations, with the techniques this project can absorb.
- [`docs/gpu-selection-survey.md`](docs/gpu-selection-survey.md) — primary-source
  comparison of RTX 5090, RTX 4090, RTX PRO 6000, L40S, A100, H100/H200, and
  B200 for this project's exact `u32`/`u64` modular-arithmetic workload.
- [`docs/benchmark-tuning-plan.md`](docs/benchmark-tuning-plan.md) — reproducible
  benchmark matrix and tuning/selection rules for GPU parameters and kernel
  variants.
- [`docs/openfhe-integration-plan.md`](docs/openfhe-integration-plan.md) — the
  tracked 95-day implementation plan (plus risk reserve) from the current NTT
  baseline through GPU base conversion, Hybrid key switching, and validated
  OpenFHE integration.
- [`docs/vscode-debugging.md`](docs/vscode-debugging.md) — shared VS Code build,
  run, test, and LLDB step-debugging setup for the small CPU NTT example.

## Provenance

Independent clean-room implementation. Contains no code, data, or artifacts from any
past or present employer. Development history is fully documented in this repository's
git log from the first commit.

## License

TBD (decide before accepting outside contributions).
