#pragma once
// Packed per-operation parameters shared by CPU test transport and CUDA.
#include "ntt/basis_conversion.hpp"

namespace ntt::resident::detail {
inline std::vector<uint64_t> pack_basis(const ApproximateBasisPlan &p) {
  std::vector<uint64_t> out;
  for (const auto *part : {&p.source_moduli(), &p.target_moduli(), &p.inverses(),
                          &p.inverse_quotients(), &p.source_hats_mod_target()})
    out.insert(out.end(), part->begin(), part->end());
  return out;
}
NTT_HD uint64_t resident_basis_value(const uint64_t *input, uint64_t n,
                                           uint64_t index, uint64_t source_count,
                                           uint64_t target_count, const uint64_t *table) {
  return approximate_basis_value(input, n, index % n, index / n, source_count,
      table, table + source_count, target_count, table + source_count + target_count,
      table + 2 * source_count + target_count, table + 3 * source_count + target_count);
}
NTT_HD uint64_t resident_correction_value(uint64_t left, uint64_t right,
                                                 uint64_t tower, uint64_t count,
                                                 const uint64_t *table) {
  return mul_mod(sub_mod(left, right, table[tower]), table[count + tower], table[tower]);
}
} // namespace ntt::resident::detail
