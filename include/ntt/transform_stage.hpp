#pragma once

#include <vector>

#include "ntt/dev_core.hpp"

namespace ntt {

#ifdef NTT_CUDA_LEGACY_BUTTERFLY
inline constexpr bool kUseLegacyButterfly = true;
#else
inline constexpr bool kUseLegacyButterfly = false;
#endif
inline constexpr const char *kTransformSchedule =
    kUseLegacyButterfly ? "legacy_dif" : "group_twiddle_ct";

// Retain the N+1 stride for both builds. The group-twiddle schedule uses
// T[m+g] = psi^bit_reverse(m+g, log2(N)); the inverse table uses psi^-1.
inline std::vector<uint64_t> transform_twiddles(uint64_t n, uint64_t root,
                                                uint64_t modulus) {
  std::vector<uint64_t> table(n + 1);
  const auto width = log2_of_pow2(n);
  uint64_t power = 1;
  for (uint64_t exponent = 0; exponent < n; ++exponent) {
    table[kUseLegacyButterfly ? exponent : bit_reverse(exponent, width)] = power;
    power = mul_mod(power, root, modulus);
  }
  table[n] = power;
  return table;
}

// Each shared-memory window maps local index i to global origin + Stride*i.
// Columns have stride 256; contiguous tiles have stride 1. The group
// twiddle depends on the global group, not the pair's offset within it.
#if defined(__CUDACC__)
#define NTT_STAGE_INLINE __device__ __forceinline__
#else
#define NTT_STAGE_INLINE inline
#endif
template <bool Inverse, uint64_t Stride>
NTT_STAGE_INLINE void transform_stage(uint64_t *values, uint64_t window_length,
                                     uint64_t local_distance, uint64_t n,
                                     const uint64_t *twiddles, uint64_t modulus,
                                     uint32_t thread, uint32_t threads,
                                     uint64_t global_origin) {
  static_assert(Stride == 1 || Stride == 256);
  if constexpr (kUseLegacyButterfly) {
    if constexpr (Stride == 1) {
      if constexpr (Inverse)
        dev_dit_stage(values, window_length, local_distance, n, twiddles, modulus, thread, threads);
      else
        dev_dif_stage(values, window_length, local_distance, n, twiddles, modulus, thread, threads);
    } else {
      if constexpr (Inverse)
        dev_dit_col_stage(values, window_length, local_distance, n, twiddles, modulus, thread, threads, global_origin);
      else
        dev_dif_col_stage(values, window_length, local_distance, n, twiddles, modulus, thread, threads, global_origin);
    }
  } else {
    const auto distance_bits = log2_of_pow2(local_distance);
    const auto group_width_bits = distance_bits + (Stride == 1 ? 1 : 9);
    const auto first_twiddle = (n >> group_width_bits) + (global_origin >> group_width_bits);
    for (uint64_t pair = thread; pair < window_length / 2; pair += threads) {
      const auto local_group = pair >> distance_bits;
      const auto upper = pair + (local_group << distance_bits);
      const auto lower = upper + local_distance;
      const auto twiddle = twiddles[first_twiddle + local_group];
      const auto a = values[upper];
      const auto b = values[lower];
      if constexpr (Inverse) {
        values[upper] = add_mod(a, b, modulus);
        values[lower] = mul_mod(sub_mod(a, b, modulus), twiddle, modulus);
      } else {
        const auto product = mul_mod(b, twiddle, modulus);
        values[upper] = add_mod(a, product, modulus);
        values[lower] = sub_mod(a, product, modulus);
      }
    }
  }
}
#undef NTT_STAGE_INLINE

} // namespace ntt
