#pragma once
// Bit-reversal permutation, the memory-movement half of every NTT.

#include <algorithm>
#include <cstdint>
#include <vector>

// Host+device qualifier (mirrors mod_arith.hpp): the CUDA kernels call
// log2_of_pow2/bit_reverse from device code; bit_reverse_permute stays
// host-only (std::vector, and no bit-reversal pass exists in the shipped
// schedule — it is kept as documented, tested machinery).
#if !defined(NTT_HD)
#if defined(__CUDACC__)
#define NTT_HD __host__ __device__ inline
#else
#define NTT_HD inline
#endif
#endif

namespace ntt {

// log2 of a power of two n.
NTT_HD uint64_t log2_of_pow2(uint64_t n) {
  uint64_t lg = 0;
  while ((uint64_t{1} << lg) < n)
    ++lg;
  return lg;
}

// Reverses the low lg bits of x.
NTT_HD uint64_t bit_reverse(uint64_t x, uint64_t lg) {
  uint64_t r = 0;
  for (uint64_t i = 0; i < lg; ++i) {
    r = (r << 1) | (x & 1);
    x >>= 1;
  }
  return r;
}

// In-place bit-reversal permutation of v (v.size() must be a power of two).
//
// One forward pass suffices because reversing bits twice returns the original
// index: bitrev(bitrev(i)) == i. The permutation is therefore a set of
// disjoint pairs (i, j) that point at each other, plus self-mapped slots
// (bitrev(i) == i) that stay put — visiting i already reveals both members
// of its pair.
//
// The `i < j` guard swaps each pair exactly once, from its smaller index.
// Without it every pair would be swapped twice (once from i, once from j),
// and the second swap would undo the first, leaving v unchanged. Self-mapped
// slots (i == j) are skipped by the same guard.
inline void bit_reverse_permute(std::vector<uint64_t> &v) {
  const uint64_t n = v.size();
  if (n < 2)
    return;
  const uint64_t lg = log2_of_pow2(n);
  for (uint64_t i = 0; i < n; ++i) {
    const uint64_t j = bit_reverse(i, lg);
    if (i < j)
      std::swap(v[i], v[j]);
  }
}

} // namespace ntt
