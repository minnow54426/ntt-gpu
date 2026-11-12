#pragma once
// Host-only preflight shared by the synchronous CUDA wrapper and CPU tests.
#include "ntt/param_gen.hpp"
#include <limits>
#include <span>
#include <stdexcept>

namespace ntt::cuda::detail {
inline size_t checked_bytes(size_t count) {
  if (count > std::numeric_limits<size_t>::max() / sizeof(uint64_t))
    throw std::length_error("CUDA buffer byte count overflow");
  return count * sizeof(uint64_t);
}
struct Shape {
  size_t values, twiddles;
};
inline Shape validate_parameters(uint64_t n, std::span<const uint64_t> q,
                                 std::span<const uint64_t> roots) {
  if (n < 2 || n > 65536 || (n & (n - 1)))
    throw std::invalid_argument("CUDA baseline requires power-of-two N in [2,65536]");
  if (q.empty() || q.size() != roots.size() || q.size() > 65535)
    throw std::invalid_argument("one (q, psi) pair per limb; at most 65535 limbs");
  for (size_t k = 0; k < q.size(); ++k) {
    if (q[k] <= n || q[k] >= (uint64_t{1} << 63) || q[k] % (2 * n) != 1 ||
        !::ntt::detail::is_prime_mr(q[k]))
      throw std::invalid_argument("CUDA requires prime q < 2^63 with 2N dividing q-1");
    if (roots[k] == 0 || roots[k] >= q[k] || pow_mod(roots[k], n, q[k]) != q[k] - 1)
      throw std::invalid_argument("CUDA requires a canonical primitive 2N-th root");
  }
  if (q.size() > std::numeric_limits<size_t>::max() / (n + 1))
    throw std::length_error("CUDA table element count overflow");
  const Shape shape{q.size() * static_cast<size_t>(n),
                    q.size() * static_cast<size_t>(n + 1)};
  checked_bytes(shape.values);
  checked_bytes(shape.twiddles);
  return shape;
}
inline void validate_values(uint64_t n, std::span<const uint64_t> q,
                            std::span<const uint64_t> values) {
  // n/q come from validated, immutable wrapper state.
  if (values.size() != q.size() * n)
    throw std::length_error("CUDA buffer must have limbs*N elements");
  for (size_t k = 0; k < q.size(); ++k)
    for (size_t i = 0; i < n; ++i)
      if (values[k * n + i] >= q[k])
        throw std::invalid_argument("CUDA requires canonical residues");
}
} // namespace ntt::cuda::detail
