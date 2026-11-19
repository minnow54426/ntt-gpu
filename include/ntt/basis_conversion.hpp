#pragma once
// Coefficient-domain approximate CRT switch, with the unreduced CRT lift
// used by pinned OpenFHE. This does not perform NTT or ModDown scaling.
#include "ntt/param_gen.hpp"
#include "ntt/shoup.hpp"
#include <algorithm>
#include <span>

namespace ntt {
// Source-major coefficients and source-major hat table. One independent
// output coefficient can be assigned to one GPU thread. Reducing each term
// bounds the accumulator even for 40 source towers and 63-bit moduli.
NTT_HD uint64_t approximate_basis_value(
    const uint64_t *input, uint64_t dimension, uint64_t coefficient,
    uint64_t target_index, uint64_t source_count, const uint64_t *source_moduli,
    const uint64_t *target_moduli, uint64_t target_count, const uint64_t *inverses,
    const uint64_t *inverse_quotients, const uint64_t *source_hats_mod_target) {
  const uint64_t target = target_moduli[target_index];
  uint64_t sum = 0;
  for (uint64_t source = 0; source < source_count; ++source) {
    const uint64_t scaled =
        shoup_multiply(input[source * dimension + coefficient], inverses[source],
                       inverse_quotients[source], source_moduli[source]);
    const uint64_t term =
        mul_mod(scaled % target,
                source_hats_mod_target[source * target_count + target_index], target);
    sum = add_mod(sum, term, target);
  }
  return sum;
}

class ApproximateBasisPlan {
  std::vector<uint64_t> source_moduli_, target_moduli_, source_inverses_,
      source_inverse_quotients_, source_hats_;

public:
  // Primes and order are immutable snapshots. Rebuild for a shortened digit
  // or active prefix because the product of source primes changes.
  ApproximateBasisPlan(std::vector<uint64_t> source, std::vector<uint64_t> target)
      : source_moduli_(std::move(source)), target_moduli_(std::move(target)) {
    if (source_moduli_.empty() || source_moduli_.size() > 40 ||
        target_moduli_.empty() || target_moduli_.size() > 40)
      throw std::invalid_argument(
          "basis switch requires 1..40 source and target primes");
    auto all = source_moduli_;
    all.insert(all.end(), target_moduli_.begin(), target_moduli_.end());
    std::sort(all.begin(), all.end());
    if (std::adjacent_find(all.begin(), all.end()) != all.end())
      throw std::invalid_argument("basis switch requires distinct disjoint primes");
    for (auto modulus : all)
      if (modulus < 3 || modulus >= (uint64_t{1} << 63) ||
          !detail::is_prime_mr(modulus))
        throw std::invalid_argument("basis switch requires odd primes below 2^63");

    for (size_t source_index = 0; source_index < source_moduli_.size();
         ++source_index) {
      const auto modulus = source_moduli_[source_index];
      uint64_t hat = 1;
      for (size_t other = 0; other < source_moduli_.size(); ++other)
        if (other != source_index)
          hat = mul_mod(hat, source_moduli_[other] % modulus, modulus);
      source_inverses_.push_back(inv_mod(hat, modulus));
      source_inverse_quotients_.push_back(
          shoup_precompute(source_inverses_.back(), modulus));
      for (auto target_modulus : target_moduli_) {
        uint64_t reduced_hat = 1;
        for (size_t other = 0; other < source_moduli_.size(); ++other)
          if (other != source_index)
            reduced_hat = mul_mod(reduced_hat, source_moduli_[other] % target_modulus,
                                  target_modulus);
        source_hats_.push_back(reduced_hat);
      }
    }
  }
  const auto &source_moduli() const {
    return source_moduli_;
  }
  const auto &target_moduli() const {
    return target_moduli_;
  }
  const auto &inverses() const {
    return source_inverses_;
  }
  const auto &inverse_quotients() const {
    return source_inverse_quotients_;
  }
  const auto &source_hats_mod_target() const {
    return source_hats_;
  }

  void validate_input(uint64_t dimension, std::span<const uint64_t> input) const {
    if (dimension < 2 || dimension > 65536 || (dimension & (dimension - 1)) ||
        input.size() != dimension * source_moduli_.size())
      throw std::invalid_argument(
          "basis switch requires power-of-two N in [2,65536] and source*N values");
    for (size_t source = 0; source < source_moduli_.size(); ++source)
      for (size_t index = 0; index < dimension; ++index)
        if (input[source * dimension + index] >= source_moduli_[source])
          throw std::invalid_argument("basis switch requires canonical input residues");
  }
  std::vector<uint64_t> convert(uint64_t dimension,
                                std::span<const uint64_t> input) const {
    validate_input(dimension, input);
    std::vector<uint64_t> result(dimension * target_moduli_.size());
    for (size_t target = 0; target < target_moduli_.size(); ++target)
      for (size_t index = 0; index < dimension; ++index)
        result[target * dimension + index] = approximate_basis_value(
            input.data(), dimension, index, target, source_moduli_.size(),
            source_moduli_.data(), target_moduli_.data(), target_moduli_.size(),
            source_inverses_.data(), source_inverse_quotients_.data(),
            source_hats_.data());
    return result;
  }
};
} // namespace ntt
