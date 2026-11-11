// D7: public NativePoly/DCRTPoly representation against independent index formulas.
// Requires the exact pinned OpenFHE 1.6.1 core library; no CUDA execution.
#include <algorithm>
#include <array>
#include <cstdint>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "lattice/lat-hal.h"
#include "math/math-hal.h"
#include "math/nbtheory.h"
#include "ntt/ntt.hpp"

namespace {
constexpr uint64_t kN = 65536;
constexpr uint32_t kOrder = 2 * kN;

void require(bool ok, const std::string& context) {
  if (!ok) throw std::runtime_error(context);
}

// Independent of the production NTT (which performs no index reversal).
uint64_t reverse_index(uint64_t index) {
  uint64_t reversed = 0;
  for (uint64_t width = kN; width > 1; width >>= 1) {
    reversed = 2 * reversed + index % 2;
    index /= 2;
  }
  return reversed;
}

lbcrypto::NativePoly poly(const std::vector<uint64_t>& values, uint64_t q,
                          uint64_t psi, Format format) {
  auto params = std::make_shared<lbcrypto::NativePoly::Params>(
      kOrder, lbcrypto::NativeInteger(q), lbcrypto::NativeInteger(psi));
  lbcrypto::NativeVector data(kN, lbcrypto::NativeInteger(q));
  for (uint64_t i = 0; i < kN; ++i) data[i] = lbcrypto::NativeInteger(values[i]);
  lbcrypto::NativePoly result(params, format);
  result.SetValues(std::move(data), format);
  return result;
}

void equal(const lbcrypto::NativePoly& actual, const std::vector<uint64_t>& expected,
           const std::string& context) {
  require(actual.GetLength() == kN, context + ": dimension");
  for (uint64_t i = 0; i < kN; ++i)
    require(actual[i].ConvertToInt() == expected[i],
            context + ": index " + std::to_string(i));
}

void native_contract(uint64_t q, uint64_t psi) {
  require(ntt::pow_mod(psi, kN, q) == q - 1, "primitive 2N root");
  uint64_t minimum = psi, power = psi;
  const uint64_t square = ntt::mul_mod(psi, psi, q);
  for (uint64_t j = 0; j < kN; ++j) {
    minimum = std::min(minimum, power);
    power = ntt::mul_mod(power, square, q);
  }
  require(minimum == psi, "OpenFHE root must be the minimum primitive root");

  std::vector<uint64_t> points(kN);
  for (uint64_t k = 0; k < kN; ++k)
    points[k] = ntt::pow_mod(psi, 2 * reverse_index(k) + 1, q);
  require(points[0] == psi && points[1] == q - psi, "first bit-reversed points");
  require(ntt::pow_mod(psi, 3, q) != points[0], "another primitive root changes the array");

  // Closed-form monomials identify every evaluation address, not just a roundtrip.
  for (const uint64_t degree : std::array<uint64_t, 4>{0, 1, 17, kN - 1}) {
    std::vector<uint64_t> input(kN, 0), expected(kN);
    input[degree] = 1;
    for (uint64_t k = 0; k < kN; ++k)
      expected[k] = ntt::pow_mod(points[k], degree, q);
    auto actual = poly(input, q, psi, Format::COEFFICIENT);
    actual.SetFormat(Format::COEFFICIENT);
    equal(actual, input, "SetFormat same coefficient format is a no-op");
    actual.SetFormat(Format::EVALUATION);
    equal(actual, expected, "OpenFHE monomial order");
    actual.SetFormat(Format::EVALUATION);
    equal(actual, expected, "SetFormat same evaluation format is a no-op");
    auto ours = input;
    ntt::NegacyclicNtt(kN, q, psi).forward(ours);
    require(ours == expected, "reference monomial order");
    actual.SwitchFormat();
    equal(actual, input, "SwitchFormat inverse");
  }

  // Independent evaluation input: inverse basis vector at four distant slots.
  for (const uint64_t slot : std::array<uint64_t, 4>{0, 1, kN / 2, kN - 1}) {
    std::vector<uint64_t> input(kN, 0), expected(kN);
    input[slot] = 1;
    uint64_t value = ntt::inv_mod(kN, q);
    const uint64_t step = ntt::inv_mod(points[slot], q);
    for (uint64_t i = 0; i < kN; ++i) {
      expected[i] = value;
      value = ntt::mul_mod(value, step, q);
    }
    auto actual = poly(input, q, psi, Format::EVALUATION);
    equal(actual, input, "SetValues labels without transforming");
    actual.OverrideFormat(Format::COEFFICIENT);
    equal(actual, input, "OverrideFormat changes only metadata");
    actual.OverrideFormat(Format::EVALUATION); // restore truthful metadata
    actual.SetFormat(Format::COEFFICIENT);
    equal(actual, expected, "OpenFHE inverse basis and N-inverse");
    ntt::NegacyclicNtt(kN, q, psi).inverse(input);
    require(input == expected, "reference inverse basis and N-inverse");
  }
  std::cout << "NATIVE REPRESENTATION OK: N=" << kN << " q=" << q
            << " (minimum root, four monomials, four inverse slots, format APIs)\n";
}

void dcrt_contract(const std::vector<lbcrypto::NativeInteger>& moduli,
                   const std::vector<lbcrypto::NativeInteger>& roots) {
  const size_t towers = moduli.size();
  auto params = std::make_shared<lbcrypto::DCRTPoly::Params>(kOrder, moduli, roots);
  auto total = params->GetModulus();
  require(total.GetMSB() > 64, "composite modulus exceeds uint64");
  auto big_params = std::make_shared<lbcrypto::Poly::Params>(kOrder, total, lbcrypto::BigInteger(1));
  lbcrypto::BigVector big_values(kN, total);
  for (uint64_t i = 0; i < kN; ++i) {
    if (i % 4 == 0 || i == kN - 1) big_values[i] = total - lbcrypto::BigInteger(1);
    else if (i % 4 == 1) big_values[i] = lbcrypto::BigInteger("18446744073709551633");
    else big_values[i] = lbcrypto::BigInteger(i % 4 == 2 ? i : 0);
  }
  lbcrypto::Poly large(big_params, Format::COEFFICIENT);
  large.SetValues(big_values, Format::COEFFICIENT);
  lbcrypto::DCRTPoly actual(large, params);
  std::vector<std::vector<uint64_t>> coefficient(towers), evaluation(towers);
  std::vector<uint64_t> packed(towers * kN);
  for (size_t limb = 0; limb < towers; ++limb) {
    const uint64_t q = moduli[limb].ConvertToInt(), psi = roots[limb].ConvertToInt();
    coefficient[limb].resize(kN);
    for (uint64_t i = 0; i < kN; ++i) {
      coefficient[limb][i] = big_values[i].Mod(lbcrypto::BigInteger(q)).ConvertToInt();
      packed[limb * kN + i] = coefficient[limb][i];
    }
    require(actual.GetAllElements()[limb].GetModulus() == moduli[limb], "tower order");
    require(actual.GetAllElements()[limb].GetParams()->GetRootOfUnity() == roots[limb], "tower root");
    equal(actual.GetAllElements()[limb], coefficient[limb], "big coefficient to tower residues");
    evaluation[limb] = coefficient[limb];
    ntt::NegacyclicNtt(kN, q, psi).forward(evaluation[limb]);
  }
  auto interpolated = actual.CRTInterpolate();
  require(interpolated.GetValues() == big_values, "CRT interpolation of >64-bit coefficients");
  actual.SetFormat(Format::EVALUATION);
  for (size_t limb = 0; limb < towers; ++limb) {
    require(actual.GetAllElements()[limb].GetFormat() == Format::EVALUATION, "tower format");
    equal(actual.GetAllElements()[limb], evaluation[limb], "DCRT forward per tower");
    for (uint64_t i = 0; i < kN; ++i) packed[limb * kN + i] = evaluation[limb][i];
  }
  actual.SetFormat(Format::EVALUATION);
  // A separate object imports flat evaluation arrays by tagging, not transforming.
  lbcrypto::DCRTPoly imported(params, Format::EVALUATION, true);
  for (size_t limb = 0; limb < towers; ++limb) {
    std::vector<uint64_t> values(packed.begin() + limb * kN, packed.begin() + (limb + 1) * kN);
    imported.SetElementAtIndex(static_cast<uint32_t>(limb),
        poly(values, moduli[limb].ConvertToInt(), roots[limb].ConvertToInt(), Format::EVALUATION));
  }
  imported.SetFormat(Format::COEFFICIENT);
  for (size_t limb = 0; limb < towers; ++limb)
    equal(imported.GetAllElements()[limb], coefficient[limb], "flat GPU-layout import and inverse");

  // Multiply by x: independent closed-form wrap checks pointwise DCRT arithmetic.
  lbcrypto::DCRTPoly x(params, Format::COEFFICIENT, true);
  for (size_t limb = 0; limb < towers; ++limb) {
    std::vector<uint64_t> values(kN, 0); values[1] = 1;
    x.SetElementAtIndex(static_cast<uint32_t>(limb),
        poly(values, moduli[limb].ConvertToInt(), roots[limb].ConvertToInt(), Format::COEFFICIENT));
  }
  x.SetFormat(Format::EVALUATION);
  auto product = actual * x;
  product.SetFormat(Format::COEFFICIENT);
  for (size_t limb = 0; limb < towers; ++limb) {
    auto expected = coefficient[limb];
    for (uint64_t i = kN - 1; i > 0; --i) expected[i] = coefficient[limb][i - 1];
    const uint64_t last = coefficient[limb].back();
    expected[0] = last ? moduli[limb].ConvertToInt() - last : 0;
    equal(product.GetAllElements()[limb], expected, "DCRT pointwise multiply and negative wrap");
  }
  auto shortened = imported;
  shortened.DropLastElement();
  require(shortened.GetAllElements().size() == towers - 1, "active tower count");
  require(imported.GetAllElements().size() == towers &&
          imported.GetParams()->GetParams().size() == towers,
          "drop does not mutate original params");
  require(shortened.GetParams()->GetParams().size() == towers - 1, "shortened parameter count");
  lbcrypto::BigInteger remaining_modulus(1);
  for (size_t limb = 0; limb < towers - 1; ++limb)
    remaining_modulus *= lbcrypto::BigInteger(moduli[limb].ConvertToInt());
  require(shortened.GetParams()->GetModulus() == remaining_modulus, "shortened composite modulus");
  for (size_t limb = 0; limb < towers - 1; ++limb)
    equal(shortened.GetAllElements()[limb], coefficient[limb], "drop preserves prefix order and residues");
  auto relabeled = imported;
  relabeled.OverrideFormat(Format::EVALUATION);
  require(relabeled.GetFormat() == Format::EVALUATION &&
          relabeled.GetAllElements()[0].GetFormat() == Format::COEFFICIENT,
          "DCRT OverrideFormat only changes outer metadata; do not use for conversion");
  std::cout << "DCRT REPRESENTATION OK: N=" << kN << ", towers=" << towers
            << " (ordered packing, >64-bit coefficients, format conversion, multiply, level prefix)\n";
}
} // namespace

int main() {
  try {
    // Deliberately non-size-sorted order: adapters must retain the caller's list.
    std::vector<lbcrypto::NativeInteger> moduli, roots;
    for (const uint32_t bits : {55U, 50U, 60U}) {
      auto q = lbcrypto::LastPrime<lbcrypto::NativeInteger>(bits, kOrder);
      require(q.GetMSB() == bits, "exact modulus bit length");
      auto root = lbcrypto::RootOfUnity(kOrder, q);
      moduli.push_back(q); roots.push_back(root);
      native_contract(q.ConvertToInt(), root.ConvertToInt());
    }
    dcrt_contract(moduli, roots);
    std::cout << "OPENFHE REPRESENTATION CONTRACT PASSED\n";
  } catch (const std::exception& error) {
    std::cerr << "FAIL: " << error.what() << '\n';
    return 1;
  }
  return 0;
}
