#pragma once
// Host-side parameter search: (q, psi) pairs at arbitrary n.
//
// A faithful port of the Python oracle's find_primes/find_psi
// (tests/oracle/ntt_oracle.py): 46-bit primes q with (2n) | (q-1) found
// by stepping through the residue class, psi of order exactly 2n
// (accepted only when psi^n == q-1). Used by bench and verify's big-N
// case, which need parameters at sizes the checked-in fixtures don't
// cover. Two deliberate equivalences, one deliberate difference:
//   - primality: Miller-Rabin with the oracle's same 12 bases
//     (deterministic far beyond 64-bit);
//   - acceptance: identical (psi^n == -1) test;
//   - candidates: a deterministic xorshift walk instead of Python's
//     seeded Random stream — any candidate passing the acceptance test
//     is a valid root, so the streams need not match.

#include <cstdint>
#include <stdexcept>
#include <vector>

#include "ntt/mod_arith.hpp"

namespace ntt {

namespace detail {

// Miller-Rabin with the oracle's bases (2..37): deterministic for every
// n this repo will ever see (< 2^64).
inline bool is_prime_mr(uint64_t n) {
  if (n < 2) return false;
  for (uint64_t p : {2u, 3u, 5u, 7u, 11u, 13u, 17u, 19u, 23u, 29u, 31u, 37u})
    if (n % p == 0) return n == p;
  uint64_t d = n - 1, s = 0;
  while ((d & 1) == 0) {
    d >>= 1;
    ++s;
  }
  for (uint64_t a : {2u, 3u, 5u, 7u, 11u, 13u, 17u, 19u, 23u, 29u, 31u, 37u}) {
    uint64_t x = pow_mod(a % n, d, n);
    if (x == 1 || x == n - 1) continue;
    bool witness = true;  // a says composite unless a pow hits n-1 in time
    for (uint64_t r = 0; r + 1 < s; ++r) {
      x = mul_mod(x, x, n);
      if (x == n - 1) {
        witness = false;
        break;
      }
    }
    if (witness) return false;
  }
  return true;
}

}  // namespace detail

// `count` distinct primes q with (2n) | (q-1) and 2^(bits-1) <= q < 2^bits.
// Same walk as the oracle: the first multiple of 2n in the range, +1,
// then step by 2n. Throws if the bit range runs dry (cannot happen for
// bits = 46 and the sizes this repo targets).
inline std::vector<uint64_t> find_primes(uint64_t n, uint64_t count,
                                         uint64_t bits = 46) {
  const uint64_t step = 2 * n;
  uint64_t q = ((uint64_t{1} << (bits - 1)) / step + 1) * step + 1;
  std::vector<uint64_t> primes;
  while (primes.size() < count) {
    if (q >= (uint64_t{1} << bits))
      throw std::runtime_error("prime search exhausted the bit range");
    if (detail::is_prime_mr(q)) primes.push_back(q);
    q += step;
  }
  return primes;
}

// An element of order exactly 2n mod q: zeta = g^((q-1)/2n) accepted when
// zeta^n == q-1 (halfway around the doubled wheel is minus — the same
// fact the reference constructor validates).
inline uint64_t find_psi(uint64_t q, uint64_t n) {
  uint64_t s = q ^ n;
  if (s == 0) s = 1;
  for (;;) {
    s ^= s << 13;
    s ^= s >> 7;
    s ^= s << 17;
    const uint64_t g = 2 + s % (q - 3);  // candidate in [2, q-2]
    const uint64_t zeta = pow_mod(g, (q - 1) / (2 * n), q);
    if (pow_mod(zeta, n, q) == q - 1) return zeta;
  }
}

}  // namespace ntt
