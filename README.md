# ntt-gpu

GPU-accelerated RNS NTT, key-switching, and automorphism kernels for FHE (and, by shared
core operation, ZK proving). Target: best-in-class, honestly-benchmarked CUDA kernels.

**Status:** Layer 1 (CPU `ref` backend) complete and cross-verified — 2026-10-29. CUDA backend next.

## Roadmap

- **v0.1 (2027-01-31):** correct, tested RNS NTT CUDA kernel + benchmark table
  (RTX 4090 + RTX 5090), e.g. "2^16 coefficients, 40-bit limbs: X ms, N× vs OpenFHE CPU".
  - [x] Layer 1 — CPU reference negacyclic NTT + Python big-int oracle (2026-10-29)
  - [ ] Layer 2 — CUDA kernel, validated on the same vectors
- **v0.2 (2027-04-30):** key-switching / automorphism kernels; first upstream PRs.

## Build & test

No dependencies beyond CMake, a C++20 compiler, and Python 3 (stdlib only):

```sh
cmake -S . -B build && cmake --build build -j && ctest --test-dir build --output-on-failure
```

Three gates: vector generation (RNS primes + 2N-th roots of unity), unit tests
(modular arithmetic, bit-reversal, forward/inverse roundtrip, NTT-multiply vs
O(N²) schoolbook mod x^N+1), and the cross-language oracle check — the C++
fast transform must equal a definitional naive O(N²) big-integer transform in
Python, limb by limb, with position-wise CRT reconstruction over all limbs.

## Design

Two backends behind one interface:

- `ref` — CPU reference (correctness is defined here; big-int-checked test vectors)
- `cuda` — GPU kernel (validated against `ref` on identical vectors)

Benchmarks are only ever reported from the `cuda` backend on named hardware.

Transform convention (see `include/ntt/ntt.hpp`): ψ a primitive 2N-th root of
unity mod q, `forward: A_j = Σ_i a_i·ψ^((2j+1)·i) mod q` — evaluation at the
odd powers of ψ, where x^N+1 splits into linear factors.

## Docs

- [`docs/butterfly-explained.html`](docs/butterfly-explained.html) — an interactive,
  dependency-free beginner's explainer: why the NTT is built from butterfly units,
  from the one-line definition through the negacyclic fold rule to the full N=8
  network (open in any browser).

## Provenance

Independent clean-room implementation. Contains no code, data, or artifacts from any
past or present employer. Development history is fully documented in this repository's
git log from the first commit.

## License

TBD (decide before accepting outside contributions).
