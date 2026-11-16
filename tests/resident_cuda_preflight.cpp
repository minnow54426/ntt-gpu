// Real CUDA-linked factory, invalid inputs only. Safe without a driver.
#include "ntt/resident.hpp"
#include <iostream>
using namespace ntt::proposal;
template<class F> void rejects(F f) {
  try { f(); } catch (const ContractError &) { return; }
  throw std::runtime_error("invalid resident CUDA factory input reached runtime");
}
int main() {
  try {
    const auto q = ntt::find_primes(8, 1, 20).at(0);
    auto p = Parameters::create(8, {{q, ntt::find_psi(q, 8)}}, {}, 1);
    PolynomialLayout layout{p, 1, q_prefix(*p, 1), Domain::Coefficient, Order::Natural};
    rejects([&] { ntt::resident::make_cuda_session(transform_layout(layout, false)); });
    rejects([&] { ntt::resident::make_cuda_session(layout, -1); });
    rejects([&] { ntt::resident::make_cuda_session(layout, 0, {0, 128, 8}); });
    rejects([&] { ntt::resident::make_cuda_session(layout, 0, {1024, 0, 8}); });
    rejects([&] { ntt::resident::make_cuda_session(layout, 0, {1024, 1024, 0}); });
    std::cout << "5 CUDA-LINKED RESIDENT PREFLIGHT REJECTIONS PASSED; NO GPU EXECUTION\n";
  } catch (const std::exception &e) { std::cerr << e.what() << '\n'; return 1; }
}
