// Exact public-interface compatibility with OpenFHE 1.6.1.
//
// Seam under test, agreed before implementation: for the same N, modulus,
// primitive 2N-th root, and coefficient vector, NegacyclicNtt's forward and
// inverse arrays must equal OpenFHE's public NativeVector CRT-FTT arrays at
// every index, including its bit-reversed evaluation ordering. Repeating this
// per modulus is the DCRT-tower contract used by CKKS.

#include <cstdint>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "lattice/lat-hal.h"
#include "math/math-hal.h"
#include "math/nbtheory.h"
#include "ntt/ntt.hpp"
#include "ntt/param_gen.hpp"

namespace {

int failures = 0;

void fail(uint64_t n, uint64_t q, const std::string &operation, uint64_t index,
          uint64_t ours, uint64_t openfhe) {
  ++failures;
  std::cerr << "FAIL n=" << n << " q=" << q << " " << operation << " index=" << index
            << ": ours=" << ours << " openfhe=" << openfhe << "\n";
}

struct Lcg {
  uint64_t state;
  uint64_t next(uint64_t q) {
    state = state * 6364136223846793005ULL + 1442695040888963407ULL;
    return state % q;
  }
};

lbcrypto::NativeVector to_openfhe(const std::vector<uint64_t> &values, uint64_t q) {
  lbcrypto::NativeVector out(values.size(), lbcrypto::NativeInteger(q));
  for (size_t i = 0; i < values.size(); ++i)
    out[i] = lbcrypto::NativeInteger(values[i]);
  return out;
}

void compare(const std::vector<uint64_t> &ours, const lbcrypto::NativeVector &openfhe,
             uint64_t n, uint64_t q, const std::string &operation) {
  for (uint64_t i = 0; i < n; ++i) {
    const uint64_t want = openfhe[i].ConvertToInt();
    if (ours[i] != want) {
      fail(n, q, operation, i, ours[i], want);
      return; // one diagnostic per case keeps failures readable
    }
  }
}

void test_forward_input(uint64_t n, uint64_t q, uint64_t psi,
                        const std::vector<uint64_t> &input, const std::string &label) {
  const lbcrypto::NativeInteger modulus(q);
  const lbcrypto::NativeInteger root(psi);
  const uint32_t cyclo_order = static_cast<uint32_t>(2 * n);
  ChineseRemainderTransformFTT<lbcrypto::NativeVector>().PreCompute(root, cyclo_order,
                                                                    modulus);

  lbcrypto::NativeVector openfhe_input = to_openfhe(input, q);
  auto params =
      std::make_shared<lbcrypto::NativePoly::Params>(cyclo_order, modulus, root);
  lbcrypto::NativePoly openfhe_poly(params, Format::COEFFICIENT);
  openfhe_poly.SetValues(openfhe_input, Format::COEFFICIENT);
  openfhe_poly.SetFormat(Format::EVALUATION);
  const lbcrypto::NativeVector &openfhe_forward = openfhe_poly.GetValues();

  ntt::NegacyclicNtt transform(n, q, psi);
  std::vector<uint64_t> ours = input;
  transform.forward(ours);
  compare(ours, openfhe_forward, n, q, "forward/" + label);

  // The inverse of the exact shared forward array must also agree. This is
  // deliberately compared before asserting the original input, so a common
  // round-trip bug cannot hide a representation mismatch.
  std::vector<uint64_t> ours_inverse = ours;
  transform.inverse(ours_inverse);

  openfhe_poly.SetFormat(Format::COEFFICIENT);
  const lbcrypto::NativeVector &openfhe_inverse = openfhe_poly.GetValues();
  compare(ours_inverse, openfhe_inverse, n, q, "inverse-of-forward/" + label);
}

void test_inverse_input(uint64_t n, uint64_t q, uint64_t psi,
                        const std::vector<uint64_t> &evaluation,
                        const std::string &label) {
  const lbcrypto::NativeInteger modulus(q);
  const lbcrypto::NativeInteger root(psi);
  const uint32_t cyclo_order = static_cast<uint32_t>(2 * n);
  ChineseRemainderTransformFTT<lbcrypto::NativeVector>().PreCompute(root, cyclo_order,
                                                                    modulus);

  lbcrypto::NativeVector openfhe_input = to_openfhe(evaluation, q);
  auto params =
      std::make_shared<lbcrypto::NativePoly::Params>(cyclo_order, modulus, root);
  lbcrypto::NativePoly openfhe_poly(params, Format::EVALUATION);
  openfhe_poly.SetValues(openfhe_input, Format::EVALUATION);
  openfhe_poly.SetFormat(Format::COEFFICIENT);
  const lbcrypto::NativeVector &openfhe_inverse = openfhe_poly.GetValues();

  std::vector<uint64_t> ours = evaluation;
  ntt::NegacyclicNtt(n, q, psi).inverse(ours);
  compare(ours, openfhe_inverse, n, q, "independent-inverse/" + label);
}

void test_case(uint64_t n, uint64_t q, uint64_t psi, uint64_t seed) {
  std::vector<uint64_t> zero(n, 0);
  std::vector<uint64_t> max_value(n, q - 1);
  std::vector<uint64_t> ramp(n);
  std::vector<uint64_t> random(n);
  std::vector<uint64_t> evaluation(n);
  Lcg rng{seed};
  for (uint64_t i = 0; i < n; ++i) {
    ramp[i] = i % q;
    random[i] = rng.next(q);
    evaluation[i] = rng.next(q);
  }

  test_forward_input(n, q, psi, zero, "zero");
  test_forward_input(n, q, psi, max_value, "q-minus-one");
  test_forward_input(n, q, psi, ramp, "ramp");
  test_forward_input(n, q, psi, random, "random");
  test_inverse_input(n, q, psi, evaluation, "random-evaluation");
}

} // namespace

int main() {
  for (uint32_t exponent = 12; exponent <= 16; ++exponent) {
    const uint64_t n = uint64_t{1} << exponent;
    const uint32_t cyclo_order = static_cast<uint32_t>(2 * n);
    // OpenFHE owns parameter selection here: these exercise representative
    // CKKS tower widths and its exact primitive-root choice, rather than asking
    // OpenFHE merely to accept parameters generated by this repository.
    const std::vector<lbcrypto::NativeInteger> moduli = {
        lbcrypto::FirstPrime<lbcrypto::NativeInteger>(50, cyclo_order),
        lbcrypto::LastPrime<lbcrypto::NativeInteger>(59, cyclo_order)};
    for (size_t tower = 0; tower < moduli.size(); ++tower) {
      const uint64_t q = moduli[tower].ConvertToInt();
      const uint64_t psi =
          lbcrypto::RootOfUnity(cyclo_order, moduli[tower]).ConvertToInt();
      test_case(n, q, psi, 0x9E3779B97F4A7C15ULL ^ q ^ (tower << 32));
    }
  }
  if (failures == 0)
    std::cout << "OPENFHE EXACT COMPATIBILITY PASSED\n";
  return failures == 0 ? 0 : 1;
}
