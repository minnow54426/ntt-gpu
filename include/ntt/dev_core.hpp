#pragma once
// The GPU stage schedule, written once.
//
// Everything index-heavy about the CUDA port lives in this header and is
// compiled three ways:
//   1. by the host compiler inside src/device_sim.cpp — the portable
//      simulator that proves the schedule against NegacyclicNtt on the Mac
//      (ctest gate device_schedule_check), before any GPU is rented;
//   2. by nvcc as device code inside src/cuda/ntt_kernels.cu;
//   3. implicitly by anyone reading ntt.hpp — the stage bodies below are
//      the CPU loops unrolled one "thread slice" at a time.
// Because both sides share these bodies, the only code that can drift
// between simulator and kernel is launch scaffolding — and that residual is
// exactly what the GPU gates (cuda_matches_ref, cuda_verify) check on the
// rented box.
//
// Schedule contract (mirrors ntt.hpp exactly):
//   forward (DIF): stages d = n/2, n/4, ..., 1; pair p of a stage maps to
//     group g = p / d, offset j = p % d, indices (g*2d + j, +d);
//     butterfly (u, v) -> (u + pp[d]*v, (u - pp[d]*v) * pp[(n/d)*j])
//     where pp holds powers of psi.
//   inverse (DIT): the same stages from d = 1 up; pp holds powers of
//     psi^-1; butterfly (x, y) -> (x + y*pp[(n/d)*j], (x - y*pp[(n/d)*j]) * pp[d]).
//   The two-phase GPU split (global stages first at tile_len = n, then
//   shared-memory stages at tile_len = B = min(1024, n)) walks the SAME
//   d-sequence — B is a power of two <= n, so no stage is skipped, added,
//   or reordered.
//
// Two traps this file exists to pin:
//   - the shared phase runs stages of the GLOBAL transform inside a B-wide
//     tile: the twiddle scale (n/d)*j and the ladder constant pp[d] use the
//     global n and global d, never the tile length (a tile-local NTT of
//     size B would be silently wrong);
//   - within one stage the n/2 pairs are disjoint (each index belongs to
//     exactly one pair), so any thread interleaving is race-free AND
//     produces the identical array — which is why the simulator may run
//     the same body serially while the kernel runs it in any order.

#include <cstdint>

#include "ntt/bitrev.hpp"
#include "ntt/mod_arith.hpp"

// nvcc compiles this header as device code (inlined from a __global__
// function); everything else compiles it as plain host code.
#if defined(__CUDACC__)
#define NTT_DEV __device__ __forceinline__
#else
#define NTT_DEV inline
#endif

namespace ntt {

// One DIF (forward) stage, one thread's slice of the pairs.
//   tile     — the array segment this stage operates on: the whole limb
//              (global phase, tile_len = n) or one B-wide shared-memory
//              tile (shared phase, tile_len = B). The caller guarantees
//              tile_len is a multiple of 2*d.
//   d        — pair distance for this stage, a power of two <= tile_len/2.
//   n        — the GLOBAL transform length (twiddle exponents use n).
//   pp       — powers of psi: pp[e] = psi^e for e in [0, n].
//   tid/nthr — this thread's index and count (simulator: 0..T; kernel:
//              global phase passes the grid-stride pair index/step, shared
//              phase passes threadIdx.x/blockDim.x — both cover [0, pairs)
//              and pairs are disjoint, so either slicing is exact).
NTT_DEV void dev_dif_stage(uint64_t* tile, uint64_t tile_len, uint64_t d,
                           uint64_t n, const uint64_t* pp, uint64_t q,
                           uint32_t tid, uint32_t nthr) {
  const uint64_t lg = log2_of_pow2(d);  // d is a power of two
  for (uint64_t p = tid; p < tile_len / 2; p += nthr) {
    const uint64_t g = p >> lg;              // which 2d-wide group
    const uint64_t j = p & (d - 1);          // offset inside the group
    const uint64_t u = (g << (lg + 1)) + j;  // = i + j in ntt.hpp's loop
    const uint64_t v = u + d;
    const uint64_t x = tile[u];
    const uint64_t cv = mul_mod(pp[d], tile[v], q);  // psi^d on the v leg
    tile[u] = add_mod(x, cv, q);
    tile[v] = mul_mod(sub_mod(x, cv, q), pp[(n >> lg) * j], q);
  }
}

// One DIT (inverse) stage, one thread's slice. Identical index mapping;
// pp holds powers of psi^-1 and the multiply lands on the other legs —
// the exact inverse butterfly of dev_dif_stage (see ntt.hpp::inverse).
NTT_DEV void dev_dit_stage(uint64_t* tile, uint64_t tile_len, uint64_t d,
                           uint64_t n, const uint64_t* pp, uint64_t q,
                           uint32_t tid, uint32_t nthr) {
  const uint64_t lg = log2_of_pow2(d);
  for (uint64_t p = tid; p < tile_len / 2; p += nthr) {
    const uint64_t g = p >> lg;
    const uint64_t j = p & (d - 1);
    const uint64_t u = (g << (lg + 1)) + j;
    const uint64_t v = u + d;
    const uint64_t x = tile[u];
    const uint64_t t = mul_mod(tile[v], pp[(n >> lg) * j], q);  // y*w^-1
    tile[u] = add_mod(x, t, q);
    tile[v] = mul_mod(sub_mod(x, t, q), pp[d], q);  // * psi_inv^d
  }
}

// --- pass 1 of the two-pass split: strided-group ("column") stages ---------
//
// The whole schedule is the user-shaped two-pass split: N values = 256
// strided groups x 256 contiguous groups (docs/gpu-ntt-explained.html,
// chapter 4). Every stage whose distance d is a multiple of 256 pairs
// values with the SAME offset r = u mod 256 — pair (u, u+d) is
// (256q + r, 256(q + d/256) + r) — so the far stages live inside the
// strided group r = {r, r+256, ...}. Gathered into a tile indexed by q,
// the pair is (q, q + delta) with delta = d/256, and it runs with the
// ORIGINAL global tables: ladder pp[delta << 8] (= psi^d) and twiddle
// index (n >> (8 + lg)) * ((j << 8) | r) — which is exactly the global
// u mod 2d folded into local coordinates. No new twiddle math: the
// butterflies are the reference butterflies, regrouped.
//
// The near stages (d < 256) pair within contiguous 256-tiles and use
// dev_dif_stage/dev_dit_stage unchanged with tile_len = 256 (or n when
// n < 256): the whole forward is pass 1 (strided groups, delta =
// sub_len/2 .. 1) then pass 2 (contiguous tiles, d = min(n/2, 128) .. 1),
// and the inverse mirrors. The "mix" between passes is pure data
// movement: pass 1 scatters strided, pass 2 gathers contiguous.

NTT_DEV void dev_dif_col_stage(uint64_t* tile, uint64_t sub_len,
                               uint64_t delta, uint64_t n,
                               const uint64_t* pp, uint64_t mod,
                               uint32_t tid, uint32_t nthr, uint64_t r) {
  const uint64_t lg = log2_of_pow2(delta);  // delta is a power of two
  for (uint64_t p = tid; p < sub_len / 2; p += nthr) {
    const uint64_t g = p >> lg;              // which 2*delta-wide group
    const uint64_t j = p & (delta - 1);      // offset inside the group
    const uint64_t u = (g << (lg + 1)) + j;  // pair, in q coordinates
    const uint64_t v = u + delta;
    const uint64_t x = tile[u];
    const uint64_t cv = mul_mod(pp[delta << 8], tile[v], mod);  // psi^d
    tile[u] = add_mod(x, cv, mod);
    tile[v] = mul_mod(sub_mod(x, cv, mod),
                      pp[(n >> (8 + lg)) * ((j << 8) | r)], mod);
  }
}

// DIT mirror: the exact inverse butterfly, same q-coordinates and tables.
NTT_DEV void dev_dit_col_stage(uint64_t* tile, uint64_t sub_len,
                               uint64_t delta, uint64_t n,
                               const uint64_t* pp, uint64_t mod,
                               uint32_t tid, uint32_t nthr, uint64_t r) {
  const uint64_t lg = log2_of_pow2(delta);
  for (uint64_t p = tid; p < sub_len / 2; p += nthr) {
    const uint64_t g = p >> lg;
    const uint64_t j = p & (delta - 1);
    const uint64_t u = (g << (lg + 1)) + j;
    const uint64_t v = u + delta;
    const uint64_t x = tile[u];
    const uint64_t t = mul_mod(tile[v], pp[(n >> (8 + lg)) * ((j << 8) | r)],
                               mod);  // y * w^-1
    tile[u] = add_mod(x, t, mod);
    tile[v] = mul_mod(sub_mod(x, t, mod), pp[delta << 8], mod);  // * psi^-d
  }
}

}  // namespace ntt
