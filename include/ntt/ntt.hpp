#pragma once
// Negacyclic NTT over Z_q[x]/(x^N + 1), the transform FHE ciphertext
// arithmetic runs on (and ZK provers reuse for polynomial multiplication).
//
// Convention (pinned here; the Python oracle and the future CUDA backend must
// match it exactly):
//
//   psi  = primitive 2N-th root of unity mod q  (validated: psi^N == -1)
//   omega = psi^2                                (primitive N-th root)
//
//   forward:  A_j = sum_i a_i * psi^((2j+1)*i)  mod q    (natural mathematical j)
//   storage:  output[k] = A_bit_reverse(k, log2(N)); natural coefficients in
//   inverse:  bit-reversed evaluations in; natural coefficients out
//
// Evaluation at the odd powers psi^(2j+1) is where x^N + 1 splits into
// linear factors, so pointwise products in this domain are products mod
// (x^N + 1) — the property FHE key-switching and MSM/poly arguments rely on.
//
// Reference algorithm (correctness-first, bit-reversal-free): the classic
// production pairing of a DIF forward with a DIT inverse, both with the
// negacyclic twist folded in as ladder constants:
//   forward (DIF): stages with pair distance d = n/2, n/4, ..., 1;
//     butterfly  (u, v) -> (u + psi^d * v,  (u - psi^d * v) * w)
//     with w = (psi^(n/d))^j at pair position j. Natural-order
//     coefficients in; values out in BIT-REVERSED evaluation order.
//   inverse (DIT): the same stages from d = 1 up to n/2, every butterfly
//     undone, then one n^-1 scale. Bit-reversed values in, natural
//     coefficients out.
// The pairing kills both bit-reversal passes: pointwise products are
// indifferent to a fixed permutation of the value order, so the DIF's
// output scramble is exactly the order the DIT inverse wants — the chain
// forward -> multiply -> inverse contains no permutation anywhere. Every
// evaluation point is still an odd power of psi; the individual ladder
// and stride twiddles need not be odd powers. The twist and cyclic
// machinery are merged.
// The CUDA port runs this same stage sequence, regrouped as a two-pass
// split at 256 (see include/ntt/dev_core.hpp) — identical values out,
// bit-reversed order included.
//
// A visual walkthrough of everything below — the wheel of powers, the
// expansion where shared brackets turn into butterflies, and this exact
// stage schedule animated — lives in docs/butterfly-explained.html.

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "ntt/mod_arith.hpp"

namespace ntt {

class NegacyclicNtt {
public:
  // n: power-of-two transform length; q: prime with (2n) | (q-1);
  // psi: element of order exactly 2n mod q (psi^n == q-1).
  NegacyclicNtt(uint64_t n, uint64_t q, uint64_t psi) : n_(n), q_(q), psi_(psi) {
    // Stage len pairs wire i with i + len/2, so n must be a power of two.
    if (n < 2 || (n & (n - 1)) != 0)
      throw std::invalid_argument("n must be a power of two >= 2");
    // The prime must contain a wheel of size 2n: its multiplicative group
    // has q-1 elements, so a primitive 2n-th root of unity exists only if
    // (2n) | (q-1). Real parameter sets pick limb primes to satisfy this at
    // n = 2^16..2^18; the vector generator does the picking here.
    if (q < 2 || (q - 1) % (2 * n) != 0)
      throw std::invalid_argument("q must be prime with (2n) | (q-1)");
    // psi^n must be -1: halfway around the wheel is minus. This one fact
    // is why the odd powers of psi are exactly the roots of x^n+1 (where
    // the negacyclic world plugs in) and why butterflies have +/- wings.
    if (pow_mod(psi, n, q) != q - 1)
      throw std::invalid_argument("psi^n != -1 mod q (psi must have order 2n)");
    // q > n keeps n invertible mod q; the inverse transform scales by n^-1.
    if (q <= n)
      throw std::invalid_argument("require q > n for N^-1 to exist");

    n_inv_ = inv_mod(n_ % q_, q_);
    // Inverse root for the backward sweep's ladder constants and strides.
    psi_inv_ = inv_mod(psi_, q_);
  }

  uint64_t n() const {
    return n_;
  }
  uint64_t q() const {
    return q_;
  }

  // In place; natural-order coefficients in [0, q) in, evaluation values
  // at the odd wheel points psi^(2j+1) out — with j visited in BIT-
  // REVERSED order (the DIF convention; see header for why that kills the
  // bit-reversal passes). Stage d scales the v leg by the ladder constant
  // psi^d, folding the input twist into the butterfly, and walks the
  // difference leg by
  // w = (psi^(n/d))^j = omega^(j·n/2d), the plain cyclic stride in
  // disguise. The first pair of each block has w = 1: free.
  void forward(std::vector<uint64_t> &values) const {
    check_size(values);
    for (uint64_t stage_pair_distance = n_ / 2; stage_pair_distance >= 1;
         stage_pair_distance >>= 1) {
      const uint64_t ladder_twiddle = pow_mod(psi_, stage_pair_distance, q_);
      const uint64_t twiddle_step = pow_mod(psi_, n_ / stage_pair_distance, q_);
      for (uint64_t group_start_index = 0; group_start_index < n_;
           group_start_index += 2 * stage_pair_distance) {
        uint64_t twiddle_power = 1;
        for (uint64_t pair_offset = 0; pair_offset < stage_pair_distance;
             ++pair_offset) {
          // One ladder butterfly: (u, v) -> (u + c·v, (u − c·v)·w).
          // The +/- pair uses (psi^2)^(n/2) == psi^n == -1; this body is
          // what the CUDA backend tiles across threads.
          const uint64_t upper_index = group_start_index + pair_offset;
          const uint64_t lower_index = upper_index + stage_pair_distance;
          const uint64_t upper_value = values[upper_index];
          const uint64_t scaled_lower_value =
              mul_mod(ladder_twiddle, values[lower_index], q_);
          values[upper_index] = add_mod(upper_value, scaled_lower_value, q_);
          values[lower_index] =
              mul_mod(sub_mod(upper_value, scaled_lower_value, q_), twiddle_power, q_);
          twiddle_power = mul_mod(twiddle_power, twiddle_step, q_);
        }
      }
    }
  }

  // In place, exact inverse of forward: the same stages from d = 1 up to
  // n/2 (DIT order), every butterfly undone — (x, y) = (u + c·v,
  // (u − c·v)·w) maps back to (x + y·w⁻¹, (x − y·w⁻¹)·c⁻¹) — then one
  // n^-1 scale. Each undone butterfly owes a factor 1/2; every element
  // crosses log2(n) of them, so the unpaid debt is exactly the n^-1.
  // Bit-reversed values in, natural coefficients out: no bit-reversal
  // pass exists anywhere in this class.
  void inverse(std::vector<uint64_t> &values) const {
    check_size(values);
    for (uint64_t stage_pair_distance = 1; stage_pair_distance <= n_ / 2;
         stage_pair_distance <<= 1) {
      const uint64_t inverse_ladder_twiddle =
          pow_mod(psi_inv_, stage_pair_distance, q_);
      const uint64_t inverse_twiddle_step =
          pow_mod(psi_inv_, n_ / stage_pair_distance, q_);
      for (uint64_t group_start_index = 0; group_start_index < n_;
           group_start_index += 2 * stage_pair_distance) {
        uint64_t inverse_twiddle_power = 1;
        for (uint64_t pair_offset = 0; pair_offset < stage_pair_distance;
             ++pair_offset) {
          const uint64_t upper_index = group_start_index + pair_offset;
          const uint64_t lower_index = upper_index + stage_pair_distance;
          const uint64_t upper_value = values[upper_index];
          const uint64_t twiddled_lower_value =
              mul_mod(values[lower_index], inverse_twiddle_power, q_);
          values[upper_index] = add_mod(upper_value, twiddled_lower_value, q_);
          values[lower_index] = mul_mod(sub_mod(upper_value, twiddled_lower_value, q_),
                                        inverse_ladder_twiddle, q_);
          inverse_twiddle_power =
              mul_mod(inverse_twiddle_power, inverse_twiddle_step, q_);
        }
      }
    }
    for (uint64_t value_index = 0; value_index < n_; ++value_index)
      values[value_index] = mul_mod(values[value_index], n_inv_, q_);
  }

private:
  void check_size(const std::vector<uint64_t> &values) const {
    if (values.size() != n_)
      throw std::length_error("vector size " + std::to_string(values.size()) +
                              " != n " + std::to_string(n_));
  }

  // --- the problem shape (identical for every kind of NTT) -------------
  uint64_t n_; // transform length, a power of two: n coefficients in,
               // n evaluation values out
  uint64_t q_; // the prime modulus — one RNS limb of the full modulus Q

  // --- the plain (cyclic) NTT core --------------------------------------
  // The cyclic root needs no member any more: every stage stride
  // psi^(n/d) = (psi^2)^(n/2d) = omega^(n/2d) IS the plain cyclic
  // twiddle, generated from psi on the fly. All that remains of the core
  // is the scale every DFT inverse shares:
  uint64_t n_inv_; // n^-1 mod q — the inverse's scale: each undone
                   // butterfly owes 1/2 and every element crosses
                   // log2(n) of them, so the total debt is exactly n.

  // --- the negacyclic extra: the twist, folded --------------------------
  // THE difference from a normal NTT, and all of it. A cyclic NTT
  // evaluates at omega^j — the roots of x^n - 1 (fold-with-plus world).
  // This transform evaluates at the odd wheel points psi^(2j+1) — the
  // roots of x^n + 1 (fold-with-minus world). Textbooks reach that by
  // twisting inputs by psi^i around a cyclic core; here the twist is
  // folded into the butterflies as ladder constants psi^d. These ladder
  // powers jointly implement the odd evaluation points. Same butterflies,
  // no extra pass, no psi^i tables. That is everything "negacyclic" adds.
  uint64_t psi_;     // primitive 2n-th root: psi^n == -1 (halfway
                     // around the doubled wheel is minus); source of
                     // both the ladder constants (psi^d) and the
                     // strides (psi^(n/d)) of every stage
  uint64_t psi_inv_; // psi^-1 mod q: the backward sweep's ladder
                     // constants and strides
};

// Ring product in Z_q[x]/(x^N+1) via the NTT: forward both operands (their
// evaluations at the odd wheel points — both in the same bit-reversed
// order, so the pointwise product needs no unscrambling), multiply
// pointwise — N plain multiplies, the entire payoff of the transform —
// then invert back to natural-order coefficients. Used by the unit tests
// as the fast side of the comparison against the O(N^2) schoolbook
// product.
inline std::vector<uint64_t> ntt_multiply(const NegacyclicNtt &ntt,
                                          std::vector<uint64_t> a,
                                          std::vector<uint64_t> b) {
  ntt.forward(a);
  ntt.forward(b);
  for (uint64_t i = 0; i < a.size(); ++i)
    a[i] = mul_mod(a[i], b[i], ntt.q());
  ntt.inverse(a);
  return a;
}

} // namespace ntt
