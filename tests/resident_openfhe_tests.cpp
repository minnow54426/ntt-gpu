// Same exact-array cases on the deferred CPU transport and prepared CUDA driver.
#include "ntt/resident.hpp"
#ifndef NTT_RESIDENT_CUDA
#include "support/resident_host_driver.hpp"
#endif
#include "lattice/lat-hal.h"
#include "math/math-hal.h"
#include "math/nbtheory.h"
#include <bit>
#include <iostream>

using namespace ntt::proposal;
namespace {
size_t comparisons = 0;
ntt::resident::Session session(const PolynomialLayout &p) {
#ifdef NTT_RESIDENT_CUDA
  return ntt::resident::make_cuda_session(p);
#else
  return ntt::resident::Session(std::make_shared<resident_test::HostDriver>(p));
#endif
}
std::vector<uint64_t> reference(const PolynomialLayout &p, const std::vector<uint64_t> &a,
                               const std::vector<uint64_t> &b, int operation) {
  using namespace lbcrypto;
  const auto n = p.parameters->n(); std::vector<uint64_t> out;
  for (size_t k = 0; k < p.towers.size(); ++k) {
    const auto tower = p.parameters->tower(p.towers[k]);
    NativeInteger q(tower.q), root(tower.psi);
    ChineseRemainderTransformFTT<NativeVector>().PreCompute(root, 2*n, q);
    auto params = std::make_shared<NativePoly::Params>(2*n, q, root);
    NativeVector av(n, q), bv(n, q);
    for (size_t i = 0; i < n; ++i) { av[i] = a[k*n+i]; bv[i] = b[k*n+i]; }
    NativePoly left(params, operation == 2 ? Format::EVALUATION : Format::COEFFICIENT);
    left.SetValues(av, operation == 2 ? Format::EVALUATION : Format::COEFFICIENT);
    if (operation == 2) left.SetFormat(Format::COEFFICIENT);
    else {
      left.SetFormat(Format::EVALUATION);
      if (operation == 0) {
        NativePoly right(params, Format::COEFFICIENT); right.SetValues(bv, Format::COEFFICIENT);
        right.SetFormat(Format::EVALUATION);
        left = left * right; left.SetFormat(Format::COEFFICIENT);
      }
    }
    for (size_t i = 0; i < n; ++i) out.push_back(left.GetValues()[i].ConvertToInt());
  }
  return out;
}
void compare(std::span<const uint64_t> a, std::span<const uint64_t> b, const char *operation) {
  if (a.size() != b.size()) throw std::runtime_error("resident exact extent mismatch");
  for (size_t i = 0; i < a.size(); ++i) {
    ++comparisons;
    if (a[i] != b[i]) throw std::runtime_error(std::string(operation) + " mismatch at " + std::to_string(i));
  }
}
void run(bool smoke) {
  constexpr uint64_t n = 65536;
  std::vector<Tower> q;
  for (uint32_t bits : {55, 50, 60}) {
    auto prime = lbcrypto::LastPrime<lbcrypto::NativeInteger>(bits, 2*n);
    require(static_cast<uint32_t>(std::bit_width(prime.ConvertToInt())) == bits, "OpenFHE test prime width");
    q.push_back({prime.ConvertToInt(), lbcrypto::RootOfUnity(2*n, prime).ConvertToInt()});
  }
  auto parameters = Parameters::create(n, q, {}, 2);
  PolynomialLayout p{parameters, 3, q_prefix(*parameters, 3), Domain::Coefficient, Order::Natural};
  for (int pattern = 0; pattern < 4; ++pattern) {
    if (smoke && pattern != 3) continue;
    std::vector<uint64_t> a(n*3), b(n*3);
    uint64_t random = 0x12345678;
    for (size_t k = 0; k < 3; ++k) for (size_t i = 0; i < n; ++i) {
      random = random * 6364136223846793005ULL + 1;
      a[k*n+i] = pattern == 0 ? 0 : pattern == 1 ? q[k].q-1 : pattern == 2 ? (i == 17) : random % q[k].q;
      random = random * 6364136223846793005ULL + 1;
      b[k*n+i] = pattern == 0 ? random % q[k].q : pattern == 1 ? (i&1 ? q[k].q-1 : 0) :
                  pattern == 2 ? (i == n-1) : random % q[k].q;
    }
    const auto original_a = a, original_b = b;
    auto s = session(p); auto result = s.multiply(a, b);
    std::fill(a.begin(), a.end(), 0); std::fill(b.begin(), b.end(), 0);
    compare(result.wait_values(), reference(p, original_a, original_b, 0), "product");
    require(s.stats().uploads == 2 && s.stats().downloads == 1 && s.stats().device_copies == 4,
            "resident chain transfer/copy counts");
    s.close();
    auto checks = session(p); auto input = checks.upload(original_a);
    auto fwd = checks.forward(input);
    const auto expected_fwd = reference(p, original_a, original_b, 1);
    compare(checks.download(fwd).wait_values(), expected_fwd, "forward");
    compare(checks.download(input).wait_values(), original_a, "immutable input");
    compare(checks.download(checks.inverse(fwd)).wait_values(), original_a, "roundtrip");
    auto eval = checks.upload_evaluation(original_b);
    const auto expected_inverse = reference(p, original_b, original_a, 2);
    compare(checks.download(checks.inverse(eval)).wait_values(), expected_inverse, "independent inverse");
    auto view = checks.subview(input, 1, 1, 3);
    compare(checks.download(checks.forward(view)).wait_values(),
            std::span<const uint64_t>(expected_fwd).subspan(n,n), "tower view forward");
    auto eval_view = checks.subview(eval, 2, 1, 3);
    compare(checks.download(checks.inverse(eval_view)).wait_values(),
            std::span<const uint64_t>(expected_inverse).subspan(2*n,n), "tower view inverse");
    checks.close();
  }
}
} // namespace
int main(int argc, char **argv) {
  try {
    const bool smoke = argc == 2 && std::string(argv[1]) == "--smoke";
    if (argc != 1 && !smoke) throw std::invalid_argument("usage: resident exact test [--smoke]");
    run(smoke);
#ifdef NTT_RESIDENT_CUDA
    std::cout << "NATIVE CUDA RESIDENT EXACT CHECKS: ";
#else
    std::cout << "HOST TEST TRANSPORT / OPENFHE EXACT CHECKS (NO GPU): ";
#endif
    std::cout << comparisons << " residues passed\n";
  } catch (const std::exception &e) { std::cerr << e.what() << '\n'; return 1; }
}
