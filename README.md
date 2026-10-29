# ntt-gpu

GPU-accelerated RNS NTT, key-switching, and automorphism kernels for FHE (and, by shared
core operation, ZK proving). Target: best-in-class, honestly-benchmarked CUDA kernels.

**Status:** pre-v0.1 (scaffolded 2026-10-29)

## Roadmap

- **v0.1 (2027-01-31):** correct, tested RNS NTT CUDA kernel + benchmark table
  (RTX 4090 + RTX 5090), e.g. "2^16 coefficients, 40-bit limbs: X ms, N× vs OpenFHE CPU".
- **v0.2 (2027-04-30):** key-switching / automorphism kernels; first upstream PRs.

## Design

Two backends behind one interface:

- `ref` — CPU reference (correctness is defined here; big-int-checked test vectors)
- `cuda` — GPU kernel (validated against `ref` on identical vectors)

Benchmarks are only ever reported from the `cuda` backend on named hardware.

## Provenance

Independent clean-room implementation. Contains no code, data, or artifacts from any
past or present employer. Development history is fully documented in this repository's
git log from the first commit.

## License

TBD (decide before accepting outside contributions).
