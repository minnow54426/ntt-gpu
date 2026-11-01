// Unit tests. With a vector file argument (produced by
// tests/oracle/gen_vectors.py) also runs NTT-domain cross-checks: roundtrip
// identity, and NTT multiplication vs. O(N^2) schoolbook mod x^N+1.

#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

#include "ntt/bitrev.hpp"
#include "ntt/mod_arith.hpp"
#include "ntt/ntt.hpp"
#include "ntt/vector_io.hpp"

namespace {

int failures = 0;

void check(bool ok, const std::string& what) {
  if (!ok) {
    ++failures;
    std::cerr << "FAIL: " << what << "\n";
  }
}

// Pins the primitives: wrapping behavior, Fermat's little theorem, and
// inverse roundtrips. If anything here fails, every failure above it is
// meaningless — these are the axioms the rest is built on.
void test_mod_arith() {
  const uint64_t q = 1000000007ULL;  // known prime
  check(ntt::add_mod(q - 1, 2, q) == 1, "add_mod wraps");
  check(ntt::sub_mod(1, 2, q) == q - 1, "sub_mod wraps");
  check(ntt::mul_mod(q - 1, q - 1, q) == 1, "(q-1)^2 == 1");
  check(ntt::pow_mod(2, q - 1, q) == 1, "Fermat: 2^(q-1) == 1");
  for (uint64_t x = 1; x < 50; ++x)
    check(ntt::mul_mod(ntt::inv_mod(x, q), x, q) == 1, "inv_mod roundtrip");
  check(ntt::pow_mod(5, 0, q) == 1, "pow_mod e=0");
  check(ntt::pow_mod(5, 0, 1) == 0, "pow_mod mod 1");
}

void test_bitrev() {
  std::vector<uint64_t> v = {0, 1, 2, 3, 4, 5, 6, 7};
  ntt::bit_reverse_permute(v);
  const std::vector<uint64_t> want = {0, 4, 2, 6, 1, 5, 3, 7};
  check(v == want, "bit_reverse_permute(8)");
  check(ntt::bit_reverse(0b110, 3) == 0b011, "bit_reverse(6,3)");
  check(ntt::log2_of_pow2(1) == 0 && ntt::log2_of_pow2(1024) == 10, "log2");
}

// Deterministic pseudo-random generator so failures reproduce exactly:
// Knuth's MMIX multiplier and increment, seeded explicitly per limb.
// Same seed -> same "random" operands -> a failing run can be replayed.
struct Lcg {
  uint64_t state;
  explicit Lcg(uint64_t seed) : state(seed) {}
  uint64_t next(uint64_t q) {
    state = state * 6364136223846793005ULL + 1442695040888963407ULL;
    return state % q;
  }
};

// O(N^2) schoolbook product mod x^N+1.
std::vector<uint64_t> schoolbook(const std::vector<uint64_t>& a,
                                 const std::vector<uint64_t>& b, uint64_t q) {
  const uint64_t n = a.size();
  std::vector<uint64_t> p(2 * n - 1, 0);
  for (uint64_t i = 0; i < n; ++i)
    for (uint64_t j = 0; j < n; ++j)
      p[i + j] = ntt::add_mod(p[i + j], ntt::mul_mod(a[i], b[j], q), q);
  std::vector<uint64_t> c(n);
  for (uint64_t k = 0; k < n; ++k)  // x^(k+N) == -x^k
    c[k] = ntt::sub_mod(p[k], k + n < 2 * n - 1 ? p[k + n] : 0, q);
  return c;
}

// The vector-file gates, per RNS limb: (1) inverse(forward(a)) == a, and
// (2) ntt_multiply == the O(N^2) schoolbook product mod x^N+1. The third
// gate — C++ vs the Python big-int oracle, with CRT reconstruction across
// limbs — lives in tests/oracle/test_vectors.py.
void test_with_vectors(const ntt::VectorFile& vf) {
  for (size_t k = 0; k < vf.q.size(); ++k) {
    const ntt::NegacyclicNtt ntt(vf.n, vf.q[k], vf.psi[k]);

    // Roundtrip: inverse(forward(a)) == a.
    std::vector<uint64_t> a(vf.n), orig(vf.n);
    for (uint64_t i = 0; i < vf.n; ++i)
      orig[i] = a[i] = ntt::reduce_decimal_mod(vf.coeffs[i], vf.q[k]);
    ntt.forward(a);
    ntt.inverse(a);
    check(a == orig, "roundtrip identity limb " + std::to_string(k));

    // NTT multiply vs schoolbook, random second operand.
    Lcg rng(0x9E3779B97F4A7C15ULL ^ vf.q[k]);
    std::vector<uint64_t> b(vf.n);
    for (uint64_t i = 0; i < vf.n; ++i) b[i] = rng.next(vf.q[k]);
    check(ntt::ntt_multiply(ntt, orig, b) == schoolbook(orig, b, vf.q[k]),
          "ntt multiply == schoolbook limb " + std::to_string(k));
  }
}

}  // namespace

int main(int argc, char** argv) {
  test_mod_arith();
  test_bitrev();
  if (argc > 1) {
    try {
      test_with_vectors(ntt::read_vector_file(argv[1]));
    } catch (const std::exception& e) {
      std::cerr << "vector tests skipped: " << e.what() << "\n";
      return 1;
    }
  }
  if (failures == 0) std::cout << "ALL UNIT TESTS PASSED\n";
  return failures == 0 ? 0 : 1;
}
