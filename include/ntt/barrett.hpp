#pragma once
// B05 candidate only. Reference NTT dispatch still uses mul_mod.
#include "ntt/mod_arith.hpp"

namespace ntt {
struct BarrettModulus {
  uint64_t modulus, reciprocal_low, reciprocal_high;
};
// Host setup: mu=floor(2^128/q), including power-of-two/even moduli.
inline BarrettModulus barrett_precompute(uint64_t modulus) {
  if (modulus <= 1 || modulus >= (uint64_t{1} << 63))
    throw std::invalid_argument("Barrett requires 1 < q < 2^63");
  const auto maximum = ~static_cast<__uint128_t>(0);
  const auto reciprocal = maximum / modulus + (maximum % modulus == modulus - 1);
  return {modulus, static_cast<uint64_t>(reciprocal),
          static_cast<uint64_t>(reciprocal >> 64)};
}
namespace barrett_detail {
NTT_HD uint64_t high_product(uint64_t left, uint64_t right) {
#ifdef __CUDA_ARCH__
  return __umul64hi(left, right);
#else
  return static_cast<uint64_t>((static_cast<__uint128_t>(left) * right) >> 64);
#endif
}
// Low limb of high128(x*mu). For x=a*b<q^2 and q<2^63, the full
// quotient estimate is <q and therefore fits in this one 64-bit limb.
NTT_HD uint64_t quotient(uint64_t low, uint64_t high, BarrettModulus plan) {
  const uint64_t first = low * plan.reciprocal_high;
  const uint64_t second = high * plan.reciprocal_low;
  const uint64_t sum = first + second;
  const uint64_t middle = sum + high_product(low, plan.reciprocal_low);
  const uint64_t carry = static_cast<uint64_t>(sum < first) +
                         static_cast<uint64_t>(middle < sum);
  return high * plan.reciprocal_high + high_product(low, plan.reciprocal_high) +
         high_product(high, plan.reciprocal_low) + carry;
}
} // namespace barrett_detail

// Preconditions: a,b<q and plan is barrett_precompute(q), 1<q<2^63.
// The estimate is floor(a*b/q) or one below it. The exact residual is
// in [0,2q), hence <2^64. Its low limb is sufficient; subtract q at most
// once. No integer division/remainder occurs in this data path.
NTT_HD uint64_t barrett_multiply(uint64_t a, uint64_t b, BarrettModulus plan) {
  const uint64_t low = a * b;
  const uint64_t estimate = barrett_detail::quotient(
      low, barrett_detail::high_product(a, b), plan);
  const uint64_t residual = low - estimate * plan.modulus;
  return residual >= plan.modulus ? residual - plan.modulus : residual;
}
} // namespace ntt
