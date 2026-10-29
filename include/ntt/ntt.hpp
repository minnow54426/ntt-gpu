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
//   forward:  A_j = sum_i a_i * psi^((2j+1)*i)  mod q    (natural in/out)
//   inverse:  the exact inverse of the above; inv(fwd(a)) == a
//
// Evaluation at the odd powers psi^(2j+1) is where x^N + 1 splits into
// linear factors, so pointwise products in this domain are products mod
// (x^N + 1) — the property FHE key-switching and MSM/poly arguments rely on.
//
// Reference algorithm (no fused twiddles, no memory tricks — the GPU port
// optimizes later, correctness only):
//   forward  = twist by psi^i, then cyclic CT-DFT with omega (DIT:
//              bit-reversed input, natural output)
//   inverse  = cyclic CT^-1 with omega^-1, scale by N^-1, untwist by psi^-i

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "ntt/bitrev.hpp"
#include "ntt/mod_arith.hpp"

namespace ntt {

class NegacyclicNtt {
 public:
  // n: power-of-two transform length; q: prime with (2n) | (q-1);
  // psi: element of order exactly 2n mod q (psi^n == q-1).
  NegacyclicNtt(uint64_t n, uint64_t q, uint64_t psi) : n_(n), q_(q), psi_(psi) {
    if (n < 2 || (n & (n - 1)) != 0)
      throw std::invalid_argument("n must be a power of two >= 2");
    if (q < 2 || (q - 1) % (2 * n) != 0)
      throw std::invalid_argument("q must be prime with (2n) | (q-1)");
    if (pow_mod(psi, n, q) != q - 1)
      throw std::invalid_argument("psi^n != -1 mod q (psi must have order 2n)");
    if (q <= n) throw std::invalid_argument("require q > n for N^-1 to exist");

    omega_ = mul_mod(psi_, psi_, q_);
    n_inv_ = inv_mod(n_ % q_, q_);

    psi_powers_.resize(n_);
    psi_inv_powers_.resize(n_);
    const uint64_t psi_inv = inv_mod(psi_, q_);
    for (uint64_t i = 0; i < n_; ++i) {
      psi_powers_[i] = pow_mod(psi_, i, q_);
      psi_inv_powers_[i] = pow_mod(psi_inv, i, q_);
    }
  }

  uint64_t n() const { return n_; }
  uint64_t q() const { return q_; }

  // In place. Coefficients must already be reduced mod q.
  void forward(std::vector<uint64_t>& a) const {
    check_size(a);
    for (uint64_t i = 0; i < n_; ++i) a[i] = mul_mod(a[i], psi_powers_[i], q_);
    cyclic_ct(a, omega_);
  }

  // In place, exact inverse of forward.
  void inverse(std::vector<uint64_t>& a) const {
    check_size(a);
    const uint64_t omega_inv = inv_mod(omega_, q_);
    cyclic_ct(a, omega_inv);
    for (uint64_t i = 0; i < n_; ++i)
      a[i] = mul_mod(mul_mod(a[i], n_inv_, q_), psi_inv_powers_[i], q_);
  }

 private:
  void check_size(const std::vector<uint64_t>& a) const {
    if (a.size() != n_)
      throw std::length_error("vector size " + std::to_string(a.size()) +
                              " != n " + std::to_string(n_));
  }

  // Cyclic DFT of length n_ with primitive root `root`: DIT, bit-reversed
  // input, natural-order output. Called with root = omega it computes
  // y_j = sum_i a_i * root^(i*j); with root = omega^-1 the conjugate layout
  // used for inversion.
  void cyclic_ct(std::vector<uint64_t>& a, uint64_t root) const {
    bit_reverse_permute(a);
    for (uint64_t len = 2; len <= n_; len <<= 1) {
      const uint64_t wlen = pow_mod(root, n_ / len, q_);
      for (uint64_t i = 0; i < n_; i += len) {
        uint64_t w = 1;
        for (uint64_t j = 0; j < len / 2; ++j) {
          const uint64_t u = a[i + j];
          const uint64_t v = mul_mod(a[i + j + len / 2], w, q_);
          a[i + j] = add_mod(u, v, q_);
          a[i + j + len / 2] = sub_mod(u, v, q_);
          w = mul_mod(w, wlen, q_);
        }
      }
    }
  }

  uint64_t n_, q_, psi_, omega_, n_inv_;
  std::vector<uint64_t> psi_powers_, psi_inv_powers_;
};

// Polynomial multiplication in Z_q[x]/(x^N+1) via the NTT, for tests.
inline std::vector<uint64_t> ntt_multiply(const NegacyclicNtt& ntt,
                                          std::vector<uint64_t> a,
                                          std::vector<uint64_t> b) {
  ntt.forward(a);
  ntt.forward(b);
  for (uint64_t i = 0; i < a.size(); ++i) a[i] = mul_mod(a[i], b[i], ntt.q());
  ntt.inverse(a);
  return a;
}

}  // namespace ntt
