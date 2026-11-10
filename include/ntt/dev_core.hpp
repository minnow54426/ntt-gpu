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
//   The GPU runs the SAME d-sequence as a two-pass split at 256:
//   pass 1 gathers each strided group {r, r+256, ...} onto a counter and
//   runs the far stages (d a multiple of 256) via dev_*_col_stage;
//   pass 2 gathers each contiguous 256-tile and runs the near stages via
//   dev_*_stage with tile_len = min(256, n). B is a power of two <= n,
//   so no stage is skipped, added, or reordered.
//
// Two traps this file exists to pin:
//   - every stage is a stage of the GLOBAL transform, whatever it is
//     grouped over: the twiddle scale (n/d)*j and the ladder constant
//     pp[d] use the global n and global d, never the group or tile length
//     (a group-local NTT would be silently wrong — the col stages fold
//     the global exponents into q-coordinates explicitly);
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
//   tile_values          — the contiguous values operated on by this stage.
//   tile_length          — number of values in tile_values.
//   global_pair_distance — distance between the two butterfly inputs.
//   transform_length     — length of the complete transform, not the tile.
//   twiddle_powers       — psi^e for every exponent e in [0, n].
//   modulus              — prime modulus for the current RNS limb.
//   thread_index/count   — grid-stride ownership of disjoint butterflies.
NTT_DEV void dev_dif_stage(uint64_t *tile_values, uint64_t tile_length,
                           uint64_t global_pair_distance, uint64_t transform_length,
                           const uint64_t *twiddle_powers, uint64_t modulus,
                           uint32_t thread_index, uint32_t thread_count) {
  const uint64_t pair_distance_log2 =
      log2_of_pow2(global_pair_distance); // distance is a power of two
  for (uint64_t pair_index = thread_index; pair_index < tile_length / 2;
       pair_index += thread_count) {
    const uint64_t butterfly_group_index = pair_index >> pair_distance_log2;
    const uint64_t offset_within_group = pair_index & (global_pair_distance - 1);
    const uint64_t upper_index =
        (butterfly_group_index << (pair_distance_log2 + 1)) + offset_within_group;
    const uint64_t lower_index = upper_index + global_pair_distance;
    const uint64_t upper_value = tile_values[upper_index];
    const uint64_t scaled_lower_value = mul_mod(twiddle_powers[global_pair_distance],
                                                tile_values[lower_index], modulus);
    const uint64_t twiddle_index =
        (transform_length >> pair_distance_log2) * offset_within_group;
    tile_values[upper_index] = add_mod(upper_value, scaled_lower_value, modulus);
    tile_values[lower_index] =
        mul_mod(sub_mod(upper_value, scaled_lower_value, modulus),
                twiddle_powers[twiddle_index], modulus);
  }
}

// One DIT (inverse) stage, one thread's slice. Identical index mapping;
// inverse_twiddle_powers holds powers of psi^-1 and the multiply lands on
// the other legs—the exact inverse of dev_dif_stage (see ntt.hpp::inverse).
NTT_DEV void dev_dit_stage(uint64_t *tile_values, uint64_t tile_length,
                           uint64_t global_pair_distance, uint64_t transform_length,
                           const uint64_t *inverse_twiddle_powers, uint64_t modulus,
                           uint32_t thread_index, uint32_t thread_count) {
  const uint64_t pair_distance_log2 = log2_of_pow2(global_pair_distance);
  for (uint64_t pair_index = thread_index; pair_index < tile_length / 2;
       pair_index += thread_count) {
    const uint64_t butterfly_group_index = pair_index >> pair_distance_log2;
    const uint64_t offset_within_group = pair_index & (global_pair_distance - 1);
    const uint64_t upper_index =
        (butterfly_group_index << (pair_distance_log2 + 1)) + offset_within_group;
    const uint64_t lower_index = upper_index + global_pair_distance;
    const uint64_t upper_value = tile_values[upper_index];
    const uint64_t twiddle_index =
        (transform_length >> pair_distance_log2) * offset_within_group;
    const uint64_t twiddled_lower_value = mul_mod(
        tile_values[lower_index], inverse_twiddle_powers[twiddle_index], modulus);
    tile_values[upper_index] = add_mod(upper_value, twiddled_lower_value, modulus);
    tile_values[lower_index] =
        mul_mod(sub_mod(upper_value, twiddled_lower_value, modulus),
                inverse_twiddle_powers[global_pair_distance], modulus);
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

NTT_DEV void dev_dif_col_stage(uint64_t *gathered_group_values,
                               uint64_t strided_group_length,
                               uint64_t local_pair_distance, uint64_t transform_length,
                               const uint64_t *twiddle_powers, uint64_t modulus,
                               uint32_t thread_index, uint32_t thread_count,
                               uint64_t strided_group_offset) {
  const uint64_t local_pair_distance_log2 = log2_of_pow2(local_pair_distance);
  for (uint64_t pair_index = thread_index; pair_index < strided_group_length / 2;
       pair_index += thread_count) {
    const uint64_t butterfly_group_index = pair_index >> local_pair_distance_log2;
    const uint64_t offset_within_local_group = pair_index & (local_pair_distance - 1);
    const uint64_t upper_local_index =
        (butterfly_group_index << (local_pair_distance_log2 + 1)) +
        offset_within_local_group;
    const uint64_t lower_local_index = upper_local_index + local_pair_distance;
    const uint64_t upper_value = gathered_group_values[upper_local_index];
    const uint64_t global_pair_distance = local_pair_distance << 8;
    const uint64_t scaled_lower_value =
        mul_mod(twiddle_powers[global_pair_distance],
                gathered_group_values[lower_local_index], modulus);
    const uint64_t global_offset_within_group =
        (offset_within_local_group << 8) | strided_group_offset;
    const uint64_t twiddle_index =
        (transform_length >> (8 + local_pair_distance_log2)) *
        global_offset_within_group;
    gathered_group_values[upper_local_index] =
        add_mod(upper_value, scaled_lower_value, modulus);
    gathered_group_values[lower_local_index] =
        mul_mod(sub_mod(upper_value, scaled_lower_value, modulus),
                twiddle_powers[twiddle_index], modulus);
  }
}

// DIT mirror: same q-coordinates, inverse-power table. Composing this
// unscaled stage with its DIF counterpart yields 2*input; the complete
// transform pays all per-stage factors with a single final N^-1 scale.
NTT_DEV void dev_dit_col_stage(uint64_t *gathered_group_values,
                               uint64_t strided_group_length,
                               uint64_t local_pair_distance, uint64_t transform_length,
                               const uint64_t *inverse_twiddle_powers, uint64_t modulus,
                               uint32_t thread_index, uint32_t thread_count,
                               uint64_t strided_group_offset) {
  const uint64_t local_pair_distance_log2 = log2_of_pow2(local_pair_distance);
  for (uint64_t pair_index = thread_index; pair_index < strided_group_length / 2;
       pair_index += thread_count) {
    const uint64_t butterfly_group_index = pair_index >> local_pair_distance_log2;
    const uint64_t offset_within_local_group = pair_index & (local_pair_distance - 1);
    const uint64_t upper_local_index =
        (butterfly_group_index << (local_pair_distance_log2 + 1)) +
        offset_within_local_group;
    const uint64_t lower_local_index = upper_local_index + local_pair_distance;
    const uint64_t upper_value = gathered_group_values[upper_local_index];
    const uint64_t global_offset_within_group =
        (offset_within_local_group << 8) | strided_group_offset;
    const uint64_t twiddle_index =
        (transform_length >> (8 + local_pair_distance_log2)) *
        global_offset_within_group;
    const uint64_t twiddled_lower_value =
        mul_mod(gathered_group_values[lower_local_index],
                inverse_twiddle_powers[twiddle_index], modulus);
    const uint64_t global_pair_distance = local_pair_distance << 8;
    gathered_group_values[upper_local_index] =
        add_mod(upper_value, twiddled_lower_value, modulus);
    gathered_group_values[lower_local_index] =
        mul_mod(sub_mod(upper_value, twiddled_lower_value, modulus),
                inverse_twiddle_powers[global_pair_distance], modulus);
  }
}

} // namespace ntt
