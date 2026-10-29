#pragma once
// Bit-reversal permutation, the memory-movement half of every NTT.

#include <algorithm>
#include <cstdint>
#include <vector>

namespace ntt {

// log2 of a power of two n.
inline uint64_t log2_of_pow2(uint64_t n) {
  uint64_t lg = 0;
  while ((uint64_t{1} << lg) < n) ++lg;
  return lg;
}

// Reverses the low lg bits of x.
inline uint64_t bit_reverse(uint64_t x, uint64_t lg) {
  uint64_t r = 0;
  for (uint64_t i = 0; i < lg; ++i) {
    r = (r << 1) | (x & 1);
    x >>= 1;
  }
  return r;
}

// In-place bit-reversal permutation of v (v.size() must be a power of two).
inline void bit_reverse_permute(std::vector<uint64_t>& v) {
  const uint64_t n = v.size();
  if (n < 2) return;
  const uint64_t lg = log2_of_pow2(n);
  for (uint64_t i = 0; i < n; ++i) {
    const uint64_t j = bit_reverse(i, lg);
    if (i < j) std::swap(v[i], v[j]);
  }
}

}  // namespace ntt
