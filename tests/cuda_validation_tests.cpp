#include "ntt/cuda_validation.hpp"
#include <iostream>
#include <vector>
#ifdef NTT_TEST_CUDA_CONSTRUCTION
#include "ntt/cuda_ntt.hpp"
#endif
namespace {
size_t checks = 0;
void check(bool ok) {
  ++checks;
  if (!ok)
    throw std::runtime_error("CUDA preflight check failed");
}
template <class F> void rejects(F f) {
  ++checks;
  try {
    f();
  } catch (const std::invalid_argument &) {
    return;
  }
  throw std::runtime_error("invalid CUDA parameters accepted");
}
void run() {
  using namespace ntt::cuda::detail;
  const auto q = ntt::find_primes(65536, 3, 60);
  std::vector<uint64_t> roots;
  for (auto modulus : q)
    roots.push_back(ntt::find_psi(modulus, 65536));
  auto shape = validate_parameters(65536, q, roots);
  check(shape.values == 3 * 65536 && shape.twiddles == 3 * 65537);
  for (auto n :
       {uint64_t{0}, uint64_t{1}, uint64_t{65535}, uint64_t{131072}, uint64_t{1} << 63})
    rejects([&] { validate_parameters(n, q, roots); });
  rejects([&] { validate_parameters(65536, {}, {}); });
  rejects([&] { validate_parameters(65536, q, {}); });
  auto bad = q;
  bad[0] = 0;
  rejects([&] { validate_parameters(65536, bad, roots); });
  bad = q;
  bad[0] = uint64_t{1} << 63;
  rejects([&] { validate_parameters(65536, bad, roots); });
  bad = q;
  bad[0] = 65537ULL * 65537ULL;
  rejects([&] { validate_parameters(65536, bad, roots); });
  auto bad_root = roots;
  bad_root[0] = 1;
  rejects([&] { validate_parameters(65536, q, bad_root); });
  bad_root[0] = q[0] + roots[0];
  rejects([&] { validate_parameters(65536, q, bad_root); });
  std::vector<uint64_t> values(shape.values, 0);
  for (size_t k = 0; k < q.size(); ++k)
    values[k * 65536] = q[k] - 1;
  validate_values(65536, q, values);
  check(true);
  values[65536] = q[1];
  rejects([&] { validate_values(65536, q, values); });
  ++checks;
  try {
    validate_values(65536, q, {});
    throw std::runtime_error("short input accepted");
  } catch (const std::length_error &) {
  }
  ++checks;
  try {
    checked_bytes(std::numeric_limits<size_t>::max());
    throw std::runtime_error("overflow accepted");
  } catch (const std::length_error &) {
  }
#ifdef NTT_TEST_CUDA_CONSTRUCTION
  // Link and execute the real nvcc-compiled wrapper. All invalid constructors
  // must reject before device discovery, even on a machine without a driver.
  rejects([&] { ntt::cuda::CudaNttTransform t(65536, q, bad_root); });
  rejects([&] { ntt::cuda::CudaNttTransform t(131072, q, roots); });
  rejects([&] { ntt::cuda::CudaNttTransform t(65536, {}, {}); });
#endif
}
} // namespace
int main() {
  try {
    run();
    std::cout << checks << " CUDA host preflight checks passed; no GPU execution\n";
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
