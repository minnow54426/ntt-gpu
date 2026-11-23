#include "support/basis_cuda.hpp"
#include "ntt/barrett.hpp"
#include <cuda_runtime.h>
#include <string>

namespace {
void checked(cudaError_t status) {
  if (status != cudaSuccess)
    throw std::runtime_error(cudaGetErrorString(status));
}
class DeviceArray {
  uint64_t *data_ = nullptr;

public:
  explicit DeviceArray(size_t count) {
    checked(cudaMalloc(reinterpret_cast<void **>(&data_), count * sizeof(uint64_t)));
  }
  explicit DeviceArray(std::span<const uint64_t> input) : DeviceArray(input.size()) {
    try {
      checked(
          cudaMemcpy(data_, input.data(), input.size_bytes(), cudaMemcpyHostToDevice));
    } catch (...) {
      cudaFree(data_);
      data_ = nullptr;
      throw;
    }
  }
  DeviceArray(const DeviceArray &) = delete;
  DeviceArray &operator=(const DeviceArray &) = delete;
  ~DeviceArray() {
    if (data_)
      cudaFree(data_);
  }
  uint64_t *data() const {
    return data_;
  }
};
__global__ void basis_kernel(uint64_t *output, const uint64_t *input,
                             uint64_t dimension, uint64_t source_count,
                             uint64_t target_count, const uint64_t *source,
                             const uint64_t *target, const uint64_t *inverses,
                             const uint64_t *quotients, const uint64_t *hats) {
  const uint64_t index = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < dimension * target_count)
    output[index] = ntt::approximate_basis_value(
        input, dimension, index % dimension, index / dimension, source_count, source,
        target, target_count, inverses, quotients, hats);
}
__global__ void shoup_kernel(uint64_t *output, const uint64_t *queries,
                             uint64_t count) {
  const uint64_t index = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < count) {
    const auto *query = queries + 4 * index;
    output[index] = ntt::shoup_multiply(query[0], query[1], query[2], query[3]);
  }
}
__global__ void barrett_kernel(uint64_t *output, const uint64_t *queries,
                               uint64_t count) {
  const uint64_t index = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < count) {
    const auto *query = queries + 5 * index;
    output[index] = ntt::barrett_multiply(
        query[0], query[1], {query[2], query[3], query[4]});
  }
}
} // namespace

std::vector<uint64_t> convert_basis_cuda(const ntt::ApproximateBasisPlan &plan,
                                         uint64_t dimension,
                                         std::span<const uint64_t> input) {
  plan.validate_input(dimension, input);
  DeviceArray source(plan.source_moduli()), target(plan.target_moduli()),
      inverses(plan.inverses()), quotients(plan.inverse_quotients()),
      hats(plan.source_hats_mod_target()), values(input),
      output(dimension * plan.target_moduli().size());
  std::vector<uint64_t> result(dimension * plan.target_moduli().size());
  basis_kernel<<<static_cast<unsigned>((result.size() + 255) / 256), 256>>>(
      output.data(), values.data(), dimension, plan.source_moduli().size(),
      plan.target_moduli().size(), source.data(), target.data(), inverses.data(),
      quotients.data(), hats.data());
  checked(cudaGetLastError());
  checked(cudaDeviceSynchronize());
  checked(cudaMemcpy(result.data(), output.data(), result.size() * sizeof(uint64_t),
                     cudaMemcpyDeviceToHost));
  return result;
}

void check_shoup_cuda() {
  std::vector<uint64_t> queries, expected;
  uint64_t state = 0xB05;
  for (unsigned bits : {30U, 32U, 50U, 55U, 60U, 63U}) {
    const uint64_t modulus = (uint64_t{1} << bits) - 1;
    const std::vector<uint64_t> edges{0, 1, 2, modulus / 2, modulus - 2, modulus - 1};
    auto append = [&](uint64_t value, uint64_t constant) {
      queries.insert(
          queries.end(),
          {value, constant, ntt::shoup_precompute(constant, modulus), modulus});
      expected.push_back(ntt::mul_mod(value, constant, modulus));
    };
    for (auto value : edges)
      for (auto constant : edges)
        append(value, constant);
    for (size_t index = 0; index < 4096; ++index) {
      state = state * 6364136223846793005ULL + 1;
      const auto value = state % modulus;
      state = state * 6364136223846793005ULL + 1;
      append(value, state % modulus);
    }
  }
  DeviceArray input(queries), output(expected.size());
  shoup_kernel<<<static_cast<unsigned>((expected.size() + 255) / 256), 256>>>(
      output.data(), input.data(), expected.size());
  checked(cudaGetLastError());
  checked(cudaDeviceSynchronize());
  std::vector<uint64_t> actual(expected.size());
  checked(cudaMemcpy(actual.data(), output.data(), actual.size() * sizeof(uint64_t),
                     cudaMemcpyDeviceToHost));
  if (actual != expected)
    throw std::runtime_error("native Shoup differs from host 128-bit remainder");
}

void check_barrett_cuda() {
  std::vector<uint64_t> queries, expected;
  std::vector<uint64_t> moduli{2, 4, 17, uint64_t{1} << 32, uint64_t{1} << 62};
  uint64_t state = 0xBA2205;
  for (unsigned bits : {30U, 32U, 50U, 55U, 60U, 63U}) {
    moduli.push_back((uint64_t{1} << bits) - 1);
    moduli.push_back((uint64_t{1} << bits) - 25);
  }
  for (size_t index = 0; index < 24; ++index) {
    state = state * 6364136223846793005ULL + 1;
    auto modulus = state & ((uint64_t{1} << 63) - 1);
    moduli.push_back(modulus < 2 ? 2 : modulus);
  }
  for (auto modulus : moduli) {
    const auto plan = ntt::barrett_precompute(modulus);
    auto append = [&](uint64_t a, uint64_t b) {
      queries.insert(queries.end(), {a, b, modulus, plan.reciprocal_low, plan.reciprocal_high});
      expected.push_back(ntt::mul_mod(a, b, modulus));
    };
    const std::vector<uint64_t> edges{0, 1, 2 % modulus, modulus / 2,
                                     modulus - 2, modulus - 1,
                                     (uint64_t{1} << 32) % modulus};
    for (auto a : edges) for (auto b : edges) append(a, b);
    for (size_t index = 0; index < 4096; ++index) {
      state = state * 6364136223846793005ULL + 1;
      const auto a = state % modulus;
      state = state * 6364136223846793005ULL + 1;
      append(a, state % modulus);
    }
  }
  DeviceArray input(queries), output(expected.size());
  barrett_kernel<<<static_cast<unsigned>((expected.size() + 255) / 256), 256>>>(
      output.data(), input.data(), expected.size());
  checked(cudaGetLastError());
  checked(cudaDeviceSynchronize());
  std::vector<uint64_t> actual(expected.size());
  checked(cudaMemcpy(actual.data(), output.data(), actual.size() * sizeof(uint64_t),
                     cudaMemcpyDeviceToHost));
  if (actual != expected)
    throw std::runtime_error("native Barrett differs from host 128-bit remainder");
}
