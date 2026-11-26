#pragma once
// Packed per-operation parameters shared by CPU test transport and CUDA.
#include "ntt/backend_contract.hpp"
#include "ntt/basis_conversion.hpp"
#include "ntt/bitrev.hpp"

namespace ntt::resident::detail {
inline uint64_t special_prime_product(const proposal::Parameters &parameters, uint64_t modulus) {
  uint64_t product = 1;
  for (const auto &prime : parameters.p())
    product = mul_mod(product, prime.q % modulus, modulus);
  return product;
}

NTT_HD uint64_t automorphism_source_index(uint64_t destination, uint64_t dimension,
                                          uint64_t log_dimension, uint32_t automorphism) {
  const uint64_t odd_exponent = 2 * bit_reverse(destination, log_dimension) + 1;
  const uint64_t source_exponent = (odd_exponent * automorphism) & (2 * dimension - 1);
  return bit_reverse(source_exponent / 2, log_dimension);
}

NTT_HD uint64_t rotated_b_value(uint64_t b, const uint64_t *c0, uint64_t source,
                                uint64_t tower, uint64_t active_q, const uint64_t *scales,
                                uint64_t modulus) {
  // P*c0 has a zero residue in every special-prime tower.
  if (tower < active_q)
    b = add_mod(b, mul_mod(c0[source], scales[tower], modulus), modulus);
  return b;
}

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
NTT_HD void hybrid_mac_value(uint64_t &accumulator, uint64_t digit, uint64_t key,
                             uint64_t modulus, bool first_digit) {
  const uint64_t product = mul_mod(digit, key, modulus);
  accumulator = first_digit ? product : add_mod(accumulator, product, modulus);
}
} // namespace ntt::resident::detail
