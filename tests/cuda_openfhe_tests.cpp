// Native GPU gate. Compiling this file without running it proves no GPU result.
#include "lattice/lat-hal.h"
#include "math/nbtheory.h"
#include "ntt/cuda_ntt.hpp"
#include "ntt/ntt.hpp"
#include <array>
#include <iostream>
#include <string>
namespace {
constexpr size_t N = 65536;
size_t comparisons = 0;
struct Basis {
  std::vector<uint64_t> q, roots;
};
void equal(const std::vector<uint64_t> &a, const std::vector<uint64_t> &b,
           const char *label) {
  if (a.size() != b.size())
    throw std::runtime_error("size mismatch");
  for (size_t i = 0; i < a.size(); ++i) {
    ++comparisons;
    if (a[i] != b[i])
      throw std::runtime_error(std::string(label) + " at " + std::to_string(i));
  }
}
std::vector<uint64_t> openfhe(const Basis &basis, const std::vector<uint64_t> &input,
                              bool inverse, bool square = false) {
  std::vector<uint64_t> out(input.size()), cpu = input;
  for (size_t k = 0; k < basis.q.size(); ++k) {
    auto params = std::make_shared<lbcrypto::NativePoly::Params>(
        2 * N, lbcrypto::NativeInteger(basis.q[k]),
        lbcrypto::NativeInteger(basis.roots[k]));
    const auto format = inverse ? Format::EVALUATION : Format::COEFFICIENT;
    lbcrypto::NativePoly poly(params, format);
    lbcrypto::NativeVector data(N, lbcrypto::NativeInteger(basis.q[k]));
    for (size_t i = 0; i < N; ++i)
      data[i] = input[k * N + i];
    poly.SetValues(std::move(data), format);
    poly.SetFormat(inverse ? Format::COEFFICIENT : Format::EVALUATION);
    if (square) {
      poly *= poly;
      poly.SetFormat(Format::COEFFICIENT);
    }
    for (size_t i = 0; i < N; ++i)
      out[k * N + i] = poly[i].ConvertToInt();
    std::vector<uint64_t> limb(input.begin() + k * N, input.begin() + (k + 1) * N);
    ntt::NegacyclicNtt reference(N, basis.q[k], basis.roots[k]);
    if (square)
      limb = ntt::ntt_multiply(reference, limb, limb);
    else if (inverse)
      reference.inverse(limb);
    else
      reference.forward(limb);
    std::copy(limb.begin(), limb.end(), cpu.begin() + k * N);
  }
  equal(cpu, out, "CPU versus OpenFHE");
  return out;
}
void run(const Basis &basis, bool smoke) {
  for (size_t k = 0; k < basis.q.size(); ++k)
    std::cout << "PARAM " << k << " q=" << basis.q[k] << " psi=" << basis.roots[k]
              << '\n';
  ntt::cuda::CudaNttTransform gpu(N, basis.q, basis.roots);
  for (unsigned pattern = 0; pattern < 7; ++pattern) {
    if (smoke && pattern != 0 && pattern != 6)
      continue;
    std::vector<uint64_t> input(N * basis.q.size());
    uint64_t state = 0x7251d9;
    for (size_t k = 0; k < basis.q.size(); ++k)
      for (size_t i = 0; i < N; ++i) {
        const auto q = basis.q[k];
        state = state * 6364136223846793005ULL + 1442695040888963407ULL;
        input[k * N + i] = pattern == 0   ? 0
                           : pattern == 1 ? 1
                           : pattern == 2 ? q - 1
                           : pattern == 3 ? (i % 2 ? q - 1 : 0)
                           : pattern == 4 ? (i == 17)
                           : pattern == 5 ? (i == N - 1)
                                          : state % q;
      }
    const auto original = input;
    auto values = input;
    gpu.forward(values);
    equal(values, openfhe(basis, input, false), "GPU forward");
    auto roundtrip = values;
    gpu.inverse(roundtrip);
    equal(roundtrip, input, "round trip");
    values = input;
    gpu.inverse(values);
    equal(values, openfhe(basis, input, true), "GPU independent inverse");
    values = input;
    gpu.forward(values);
    const auto right = values;
    gpu.pointwise_multiply(values, right);
    gpu.inverse(values);
    const auto product = openfhe(basis, input, false, true);
    equal(values, product, "GPU pointwise/inverse");
    equal(gpu.multiply(input, input), product, "GPU ring product");
    equal(input, original, "host source immutability");
    std::cout << "GPU/CPU/OpenFHE exact: N=" << N << " towers=" << basis.q.size()
              << " pattern=" << pattern << '\n';
  }
}
} // namespace
int main(int argc, char **argv) {
  try {
    const bool smoke = argc == 2 && std::string(argv[1]) == "--smoke";
    if (argc > 1 && !smoke)
      throw std::invalid_argument("usage: ntt_cuda_openfhe_tests [--smoke]");
    ntt::cuda::describe_device(std::cout);
    Basis widths;
    for (unsigned bits : {55U, 50U, 60U}) {
      auto q = lbcrypto::LastPrime<lbcrypto::NativeInteger>(bits, 2 * N);
      widths.q.push_back(q.ConvertToInt());
      widths.roots.push_back(lbcrypto::RootOfUnity(2 * N, q).ConvertToInt());
    }
    run(widths, smoke);
    if (!smoke) {
      const std::array<uint64_t, 8> q = {
          1152921504606584833ULL, 1125899902124033ULL,   1125899911168001ULL,
          1125899903827969ULL,    1125899908022273ULL,   1152921504598720513ULL,
          1152921504597016577ULL, 1152921504595968001ULL};
      const std::array<uint64_t, 8> roots = {
          18043022392882ULL, 15995550378ULL,  21670117260ULL,    938640682ULL,
          6511788946ULL,     800790938143ULL, 17749908910371ULL, 11469071954203ULL};
      for (size_t L : {size_t(5), size_t(4), size_t(3), size_t(1)})
        for (bool qp : {false, true}) {
          Basis active;
          for (size_t k = 0; k < 8; ++k)
            if (k < L || (qp && k >= 5)) {
              active.q.push_back(q[k]);
              active.roots.push_back(roots[k]);
            }
          run(active, false);
        }
    }
    std::cout << "NATIVE CUDA PASS: " << comparisons << " exact residue comparisons\n";
  } catch (const std::exception &e) {
    std::cerr << "NATIVE CUDA FAILED: " << e.what() << '\n';
    return 1;
  }
}
