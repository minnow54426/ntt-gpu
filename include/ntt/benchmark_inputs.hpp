#pragma once
// Shared, fixed operand identity for CPU and GPU measurements. Vector-file
// coefficients are deliberately unused: q/root/N identify the workload.
#include "ntt/cuda_validation.hpp"
#include <utility>
#include <vector>

namespace ntt::measurement {
inline constexpr uint64_t kInputSeed = 0x7251d9;
inline auto benchmark_inputs(uint64_t dimension, const std::vector<uint64_t> &moduli,
                             const std::vector<uint64_t> &roots) {
  const auto shape = cuda::detail::validate_parameters(dimension, moduli, roots);
  std::vector<uint64_t> left(shape.values), right(shape.values);
  uint64_t state = kInputSeed;
  for (size_t index = 0; index < shape.values; ++index) {
    state = state * 6364136223846793005ULL + 1442695040888963407ULL;
    left[index] = state % moduli[index / dimension];
    state = state * 6364136223846793005ULL + 1442695040888963407ULL;
    right[index] = state % moduli[index / dimension];
  }
  return std::make_pair(std::move(left), std::move(right));
}
} // namespace ntt::measurement
