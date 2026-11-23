#pragma once
// Test-only independent twist + cyclic DIT algorithm. Never production dispatch.
#include "ntt/barrett.hpp"
#include <algorithm>
#include <vector>

namespace barrett_test {
inline void bit_reverse(std::vector<uint64_t> &values) {
  for (size_t i = 1, j = 0; i < values.size(); ++i) {
    size_t bit = values.size() >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) std::swap(values[i], values[j]);
  }
}
inline uint64_t power(uint64_t base, uint64_t exponent, ntt::BarrettModulus plan) {
  uint64_t result = 1;
  while (exponent) {
    if (exponent & 1) result = ntt::barrett_multiply(result, base, plan);
    base = ntt::barrett_multiply(base, base, plan);
    exponent >>= 1;
  }
  return result;
}
inline void cyclic(std::vector<uint64_t> &values, uint64_t omega,
                   ntt::BarrettModulus plan) {
  bit_reverse(values);
  for (size_t length = 2; length <= values.size(); length <<= 1) {
    const auto step = power(omega, values.size() / length, plan);
    for (size_t group = 0; group < values.size(); group += length) {
      uint64_t twiddle = 1;
      for (size_t index = 0; index < length / 2; ++index) {
        const auto upper = values[group + index];
        const auto lower = ntt::barrett_multiply(values[group + index + length / 2], twiddle, plan);
        values[group + index] = ntt::add_mod(upper, lower, plan.modulus);
        values[group + index + length / 2] = ntt::sub_mod(upper, lower, plan.modulus);
        twiddle = ntt::barrett_multiply(twiddle, step, plan);
      }
    }
  }
}
inline void transform(std::vector<uint64_t> &values, uint64_t q, uint64_t psi,
                      bool inverse) {
  const auto plan = ntt::barrett_precompute(q);
  if (values.size() < 2 || (values.size() & (values.size() - 1)) != 0)
    throw std::invalid_argument("test transform requires a power-of-two size");
  for (auto value : values)
    if (value >= q) throw std::invalid_argument("test transform requires canonical values");
  if (inverse) {
    bit_reverse(values); // bit-reversed evaluations to natural evaluations
    psi = ntt::inv_mod(psi, q);
  } else {
    uint64_t twist = 1;
    for (auto &value : values) {
      value = ntt::barrett_multiply(value, twist, plan);
      twist = ntt::barrett_multiply(twist, psi, plan);
    }
  }
  cyclic(values, ntt::barrett_multiply(psi, psi, plan), plan);
  if (inverse) {
    const auto scale = ntt::inv_mod(values.size(), q);
    uint64_t twist = 1;
    for (auto &value : values) {
      value = ntt::barrett_multiply(ntt::barrett_multiply(value, scale, plan), twist, plan);
      twist = ntt::barrett_multiply(twist, psi, plan);
    }
  } else {
    bit_reverse(values); // natural evaluations to the project's storage order
  }
}
} // namespace barrett_test
