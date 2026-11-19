#pragma once
#include "ntt/mod_arith.hpp"

namespace ntt {
// Prepare once on the host. A Shoup multiplier pairs a constant w with
// floor(w * 2^64 / q). Supports canonical operands and 1 < q < 2^63.
inline uint64_t shoup_precompute(uint64_t constant, uint64_t modulus) {
  if (modulus <= 1 || modulus >= (uint64_t{1} << 63) || constant >= modulus)
    throw std::invalid_argument("Shoup requires canonical constant and 1 < q < 2^63");
  return static_cast<uint64_t>((static_cast<__uint128_t>(constant) << 64) / modulus);
}

// Requires value, constant < modulus and the matching precomputed quotient.
// The quotient estimate is at most one below floor(value*constant/q).
// Thus the exact residual is in [0,2q), and its low 64 bits retain it.
// One subtraction produces a canonical residue without integer division.
NTT_HD uint64_t shoup_multiply(uint64_t value, uint64_t constant, uint64_t quotient,
                               uint64_t modulus) {
#ifdef __CUDA_ARCH__
  const uint64_t estimate = __umul64hi(value, quotient);
#else
  const uint64_t estimate =
      static_cast<uint64_t>((static_cast<__uint128_t>(value) * quotient) >> 64);
#endif
  const uint64_t residual = value * constant - estimate * modulus;
  return residual >= modulus ? residual - modulus : residual;
}
} // namespace ntt
