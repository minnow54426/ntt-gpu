#include "lattice/lat-hal.h"
#include "math/math-hal.h"
#include "math/nbtheory.h"
#include "ntt/basis_conversion.hpp"
#ifdef NTT_BASIS_CUDA
#include "support/basis_cuda.hpp"
#endif
#include <iostream>

namespace {
using namespace lbcrypto;
using Params = std::shared_ptr<DCRTPoly::Params>;

Params parameters(uint64_t dimension, const std::vector<uint64_t> &moduli) {
  std::vector<NativeInteger> primes, roots;
  for (auto modulus : moduli) {
    primes.emplace_back(modulus);
    roots.push_back(RootOfUnity(2 * dimension, primes.back()));
  }
  return std::make_shared<DCRTPoly::Params>(2 * dimension, primes, roots);
}

uint64_t run_case(uint64_t dimension, std::vector<uint64_t> source,
                  std::vector<uint64_t> target, unsigned pattern) {
  // Deliberately use unsorted orders to detect accidental tower sorting.
  std::reverse(source.begin(), source.end());
  std::reverse(target.begin(), target.end());
  const ntt::ApproximateBasisPlan plan(source, target);
  const auto source_params = parameters(dimension, source),
             target_params = parameters(dimension, target);
  DCRTPoly polynomial(source_params, Format::COEFFICIENT, false);
  std::vector<uint64_t> input(dimension * source.size());
  uint64_t state = 0xB09;
  for (size_t tower = 0; tower < source.size(); ++tower) {
    NativePoly values(source_params->GetParams()[tower], Format::COEFFICIENT);
    NativeVector row(dimension, NativeInteger(source[tower]));
    for (size_t index = 0; index < dimension; ++index) {
      state = state * 6364136223846793005ULL + 1;
      const uint64_t value = pattern == 0   ? 0
                             : pattern == 1 ? source[tower] - 1
                             : pattern == 2 ? index == dimension - 1
                                            : state % source[tower];
      input[tower * dimension + index] = value;
      row[index] = NativeInteger(value);
    }
    values.SetValues(std::move(row), Format::COEFFICIENT);
    polynomial.SetElementAtIndex(tower, std::move(values));
  }

  // Construct the upstream tables with BigInteger products independently
  // of the candidate's modular-product table construction.
  BigInteger product(1);
  for (auto modulus : source)
    product *= BigInteger(modulus);
  std::vector<NativeInteger> inverses, preconditions;
  std::vector<std::vector<NativeInteger>> hats(source.size());
  std::vector<DoubleNativeInt> barrett;
  for (size_t tower = 0; tower < source.size(); ++tower) {
    const BigInteger hat = product / BigInteger(source[tower]);
    const NativeInteger modulus(source[tower]);
    inverses.emplace_back(hat.Mod(BigInteger(source[tower])).ConvertToInt());
    inverses.back() = inverses.back().ModInverse(modulus);
    preconditions.push_back(inverses.back().PrepModMulConst(modulus));
    for (auto target_modulus : target)
      hats[tower].emplace_back(hat.Mod(BigInteger(target_modulus)).ConvertToInt());
  }
  const BigInteger base = BigInteger(1).LShiftEq(128);
  for (auto modulus : target)
    barrett.push_back((base / BigInteger(modulus)).ConvertToInt<DoubleNativeInt>());
  const auto expected = polynomial.ApproxSwitchCRTBasis(
      source_params, target_params, inverses, preconditions, hats, barrett);
#ifdef NTT_BASIS_CUDA
  const auto actual = convert_basis_cuda(plan, dimension, input);
#else
  const auto actual = plan.convert(dimension, input);
#endif
  for (size_t tower = 0; tower < target.size(); ++tower)
    for (size_t index = 0; index < dimension; ++index)
      if (actual[tower * dimension + index] !=
          expected.GetElementAtIndex(tower)[index].ConvertToInt())
        throw std::runtime_error("basis mismatch at N=" + std::to_string(dimension) +
                                 " source=" + std::to_string(source.size()) +
                                 " target=" + std::to_string(target.size()) +
                                 " pattern=" + std::to_string(pattern));
  return actual.size();
}
} // namespace

int main(int argc, char **argv) try {
  const bool smoke = argc == 2 && std::string(argv[1]) == "--smoke";
  if (argc != 1 && !smoke)
    throw std::invalid_argument("usage: basis check [--smoke]");
  uint64_t comparisons = 0;
  for (uint64_t dimension : {4096ULL, 8192ULL, 16384ULL, 32768ULL, 65536ULL}) {
    if (smoke && dimension != 65536)
      continue;
    for (unsigned bits : {30U, 50U, 55U, 60U}) {
      if (smoke && bits != 60)
        continue;
      const auto primes = ntt::find_primes(dimension, 43, bits);
      for (size_t count : {size_t(1), size_t(3), size_t(40)}) {
        if (smoke && count != 40)
          continue;
        const std::vector<uint64_t> source(primes.begin(), primes.begin() + count);
        const std::vector<uint64_t> target(primes.end() - 3, primes.end());
        for (unsigned pattern = 0; pattern < 4; ++pattern) {
          comparisons += run_case(dimension, source, target, pattern);
          comparisons += run_case(dimension, target, source, pattern);
        }
      }
    }
    std::cout << "BASIS OPENFHE OK N=" << dimension << std::endl;
  }
#ifdef NTT_BASIS_CUDA
  std::cout << "NATIVE CUDA ";
#else
  std::cout << "CPU ";
#endif
  std::cout << "BASIS PASS: " << comparisons << " residues\n";
  return 0;
} catch (const std::exception &error) {
  std::cerr << error.what() << '\n';
  return 1;
}
