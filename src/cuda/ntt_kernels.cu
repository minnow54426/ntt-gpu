// The CUDA backend: kernels, launchers, and CudaNttTransform.
//
// Design invariant: every index-heavy line lives in include/ntt/
// dev_core.hpp, shared with the device-schedule simulator (gate
// device_schedule_check, green on the Mac since Layer 2a). This file is
// deliberately only scaffolding around those bodies — allocate, launch,
// synchronize, copy — the class of code the GPU gates (cuda_matches_ref,
// cuda_verify) exist to check on the rented box.
//
// v1 arithmetic note: mul_mod is the reference __uint128_t division
// compiled for device as-is (mod_arith.hpp is __host__ __device__). On
// the GPU that lowering is emulated and slow; the bench table quantifies
// the pre-Barrett gap, and Barrett/Montgomery lands later behind the
// same signature with a dedicated cross-check.

#include <cuda_runtime.h>
#include "ntt/benchmark_inputs.hpp"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <ostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "ntt/cuda_ntt.hpp"
#include "ntt/cuda_validation.hpp"
#include "ntt/dev_core.hpp"
#include "ntt/mod_arith.hpp"
#include "ntt/ntt.hpp"
#include "ntt/param_gen.hpp"
#include "ntt/setup_resources.hpp"
#include "ntt/measurement.hpp"

// Every cudaRuntime call funnels through this: a failed allocation,
// launch, or copy must fail the gates loudly (nonzero exit + stderr),
// never surface as silently wrong numbers.
#define CHECK_CUDA(call)                                                               \
  do {                                                                                 \
    const cudaError_t cuda_status = (call);                                            \
    if (cuda_status != cudaSuccess)                                                    \
      throw std::runtime_error(std::string("CUDA: ") + #call + ": " +                  \
                               cudaGetErrorString(cuda_status));                       \
  } while (0)

namespace ntt::cuda {
namespace {

// Must match ScheduleSim (src/device_sim.cpp): the 256-value split width and
// the number of threads per block define the two-pass schedule checked by
// the simulator against the CPU reference.
constexpr uint32_t kThreadsPerBlock = 256;
constexpr uint64_t kPassSplitWidth = 256;

class DeviceGuard {
  int previous_;
  bool changed_;

public:
  explicit DeviceGuard(int owner) : previous_(-1), changed_(false) {
    CHECK_CUDA(cudaGetDevice(&previous_));
    if (previous_ != owner) {
      CHECK_CUDA(cudaSetDevice(owner));
      changed_ = true;
    }
  }
  ~DeviceGuard() noexcept {
    if (changed_)
      (void)cudaSetDevice(previous_);
  }
  DeviceGuard(const DeviceGuard &) = delete;
  DeviceGuard &operator=(const DeviceGuard &) = delete;
};

struct CudaSetupRuntime {
  int device = -1;
  uint64_t *allocate(size_t elements) {
    uint64_t *pointer = nullptr;
    CHECK_CUDA(cudaMalloc(&pointer, elements * sizeof(uint64_t)));
    return pointer;
  }
  void upload(uint64_t *destination, const uint64_t *source, size_t elements) {
    CHECK_CUDA(cudaMemcpy(destination, source, elements * sizeof(uint64_t),
                          cudaMemcpyHostToDevice));
  }
  bool release(uint64_t *pointer) noexcept {
    try {
      DeviceGuard guard(device);
      return cudaFree(pointer) == cudaSuccess;
    } catch (...) { return false; }
  }
};

// --- pass 1: one block per strided group ----------------------------------
// The far stages (pair distance is a multiple of 256) pair values with the
// same index modulo 256. Each block gathers one such strided group into
// shared memory, executes all far stages, then scatters it back. This pass
// runs only when the transform length is greater than 256.
__global__ void forward_strided_group_kernel(uint64_t *values,
                                             const uint64_t *twiddle_powers,
                                             const uint64_t *moduli,
                                             uint64_t transform_length,
                                             uint32_t strided_group_length) {
  extern __shared__ uint64_t shared_group_values[];
  const uint64_t strided_group_offset = blockIdx.x;
  uint64_t *limb_values = values + static_cast<uint64_t>(blockIdx.y) * transform_length;
  const uint64_t *limb_twiddle_powers =
      twiddle_powers + static_cast<uint64_t>(blockIdx.y) * (transform_length + 1);
  const uint64_t limb_modulus = moduli[blockIdx.y];
  for (uint32_t local_index = threadIdx.x; local_index < strided_group_length;
       local_index += blockDim.x)
    shared_group_values[local_index] =
        limb_values[strided_group_offset +
                    static_cast<uint64_t>(local_index) * kPassSplitWidth];
  __syncthreads();
  for (uint64_t local_pair_distance = strided_group_length / 2;
       local_pair_distance >= 1; local_pair_distance >>= 1) {
    dev_dif_col_stage(shared_group_values, strided_group_length, local_pair_distance,
                      transform_length, limb_twiddle_powers, limb_modulus, threadIdx.x,
                      blockDim.x, strided_group_offset);
    __syncthreads();
  }
  for (uint32_t local_index = threadIdx.x; local_index < strided_group_length;
       local_index += blockDim.x)
    limb_values[strided_group_offset +
                static_cast<uint64_t>(local_index) * kPassSplitWidth] =
        shared_group_values[local_index];
}

__global__ void inverse_strided_group_kernel(uint64_t *values,
                                             const uint64_t *inverse_twiddle_powers,
                                             const uint64_t *moduli,
                                             uint64_t transform_length,
                                             uint32_t strided_group_length) {
  extern __shared__ uint64_t shared_group_values[];
  const uint64_t strided_group_offset = blockIdx.x;
  uint64_t *limb_values = values + static_cast<uint64_t>(blockIdx.y) * transform_length;
  const uint64_t *limb_inverse_twiddle_powers =
      inverse_twiddle_powers +
      static_cast<uint64_t>(blockIdx.y) * (transform_length + 1);
  const uint64_t limb_modulus = moduli[blockIdx.y];
  for (uint32_t local_index = threadIdx.x; local_index < strided_group_length;
       local_index += blockDim.x)
    shared_group_values[local_index] =
        limb_values[strided_group_offset +
                    static_cast<uint64_t>(local_index) * kPassSplitWidth];
  __syncthreads();
  for (uint64_t local_pair_distance = 1;
       local_pair_distance <= strided_group_length / 2; local_pair_distance <<= 1) {
    dev_dit_col_stage(shared_group_values, strided_group_length, local_pair_distance,
                      transform_length, limb_inverse_twiddle_powers, limb_modulus,
                      threadIdx.x, blockDim.x, strided_group_offset);
    __syncthreads();
  }
  for (uint32_t local_index = threadIdx.x; local_index < strided_group_length;
       local_index += blockDim.x)
    limb_values[strided_group_offset +
                static_cast<uint64_t>(local_index) * kPassSplitWidth] =
        shared_group_values[local_index];
}

// --- pass 2: one block per contiguous 256-tile -----------------------------
// The near stages (pair distance <= 128) stay inside one contiguous
// 256-value tile. A transform shorter than 256 uses one smaller tile.
__global__ void forward_contiguous_tile_kernel(uint64_t *values,
                                               const uint64_t *twiddle_powers,
                                               const uint64_t *moduli,
                                               uint64_t transform_length,
                                               uint32_t tile_length) {
  extern __shared__ uint64_t shared_tile_values[];
  uint64_t *limb_values = values + static_cast<uint64_t>(blockIdx.y) * transform_length;
  const uint64_t *limb_twiddle_powers =
      twiddle_powers + static_cast<uint64_t>(blockIdx.y) * (transform_length + 1);
  const uint64_t limb_modulus = moduli[blockIdx.y];
  const uint64_t tile_start_index = static_cast<uint64_t>(blockIdx.x) * tile_length;
  for (uint32_t tile_index = threadIdx.x; tile_index < tile_length;
       tile_index += blockDim.x)
    shared_tile_values[tile_index] = limb_values[tile_start_index + tile_index];
  __syncthreads();
  for (uint64_t pair_distance = tile_length / 2; pair_distance >= 1;
       pair_distance >>= 1) {
    dev_dif_stage(shared_tile_values, tile_length, pair_distance, transform_length,
                  limb_twiddle_powers, limb_modulus, threadIdx.x, blockDim.x);
    __syncthreads();
  }
  for (uint32_t tile_index = threadIdx.x; tile_index < tile_length;
       tile_index += blockDim.x)
    limb_values[tile_start_index + tile_index] = shared_tile_values[tile_index];
}

__global__ void inverse_contiguous_tile_kernel(uint64_t *values,
                                               const uint64_t *inverse_twiddle_powers,
                                               const uint64_t *moduli,
                                               uint64_t transform_length,
                                               uint32_t tile_length) {
  extern __shared__ uint64_t shared_tile_values[];
  uint64_t *limb_values = values + static_cast<uint64_t>(blockIdx.y) * transform_length;
  const uint64_t *limb_inverse_twiddle_powers =
      inverse_twiddle_powers +
      static_cast<uint64_t>(blockIdx.y) * (transform_length + 1);
  const uint64_t limb_modulus = moduli[blockIdx.y];
  const uint64_t tile_start_index = static_cast<uint64_t>(blockIdx.x) * tile_length;
  for (uint32_t tile_index = threadIdx.x; tile_index < tile_length;
       tile_index += blockDim.x)
    shared_tile_values[tile_index] = limb_values[tile_start_index + tile_index];
  __syncthreads();
  for (uint64_t pair_distance = 1; pair_distance <= tile_length / 2;
       pair_distance <<= 1) {
    dev_dit_stage(shared_tile_values, tile_length, pair_distance, transform_length,
                  limb_inverse_twiddle_powers, limb_modulus, threadIdx.x, blockDim.x);
    __syncthreads();
  }
  for (uint32_t tile_index = threadIdx.x; tile_index < tile_length;
       tile_index += blockDim.x)
    limb_values[tile_start_index + tile_index] = shared_tile_values[tile_index];
}

__global__ void pointwise_multiply_kernel(uint64_t *left_values,
                                          const uint64_t *right_values,
                                          const uint64_t *moduli,
                                          uint64_t transform_length,
                                          uint64_t limb_count) {
  const uint64_t value_index =
      static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const uint64_t total_value_count = limb_count * transform_length;
  if (value_index >= total_value_count)
    return;

  const uint64_t limb_index = value_index / transform_length;
  left_values[value_index] =
      mul_mod(left_values[value_index], right_values[value_index], moduli[limb_index]);
}

__global__ void inverse_scale_kernel(uint64_t *values,
                                     const uint64_t *inverse_transform_lengths,
                                     const uint64_t *moduli, uint64_t transform_length,
                                     uint64_t limb_count) {
  const uint64_t value_index =
      static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const uint64_t total_value_count = limb_count * transform_length;
  if (value_index >= total_value_count)
    return;

  const uint64_t limb_index = value_index / transform_length;
  values[value_index] = mul_mod(
      values[value_index], inverse_transform_lengths[limb_index], moduli[limb_index]);
}

} // namespace

// --- Impl: device buffers + launch geometry ------------------------------
struct CudaNttTransform::Impl {
  detail::SetupResources<CudaSetupRuntime> setup;
  explicit Impl(size_t failure_step, detail::SetupTrace *trace)
      : setup(CudaSetupRuntime{}, failure_step, trace) {}
  int device = -1;
  uint64_t transform_length = 0;
  uint64_t limb_count = 0;
  uint64_t contiguous_tile_length = 0;
  uint64_t strided_group_length = 0;
  uint32_t threads_per_block = kThreadsPerBlock;

  std::vector<uint64_t> moduli;
  std::vector<NegacyclicNtt> cpu_references;

  uint64_t *device_operand_a = nullptr;
  uint64_t *device_operand_b = nullptr;
  uint64_t *device_twiddle_powers = nullptr;
  uint64_t *device_inverse_twiddle_powers = nullptr;
  uint64_t *device_moduli = nullptr;
  uint64_t *device_inverse_transform_lengths = nullptr;

  uint32_t contiguous_tile_block_count = 0;
  uint32_t elementwise_block_count = 0;

};

namespace {

// Device-resident drivers, one per public operation — exactly the launch
// sequences src/device_sim.cpp walks: forward = strided pass then tile
// pass; inverse = tile pass, strided pass, n^-1 scale. Forward uses two
// launches and inverse uses three (one fewer each when n <= 256).
template <class Implementation>
void launch_forward_transform(Implementation &impl, uint64_t *device_values) {
  if (impl.transform_length > kPassSplitWidth) {
    forward_strided_group_kernel<<<
        dim3(kPassSplitWidth, static_cast<unsigned>(impl.limb_count)),
        impl.threads_per_block, impl.strided_group_length * sizeof(uint64_t)>>>(
        device_values, impl.device_twiddle_powers, impl.device_moduli,
        impl.transform_length, static_cast<uint32_t>(impl.strided_group_length));
    CHECK_CUDA(cudaGetLastError());
  }
  forward_contiguous_tile_kernel<<<
      dim3(impl.contiguous_tile_block_count, static_cast<unsigned>(impl.limb_count)),
      impl.threads_per_block, impl.contiguous_tile_length * sizeof(uint64_t)>>>(
      device_values, impl.device_twiddle_powers, impl.device_moduli,
      impl.transform_length, static_cast<uint32_t>(impl.contiguous_tile_length));
  CHECK_CUDA(cudaGetLastError());
}

template <class Implementation>
void launch_inverse_transform(Implementation &impl, uint64_t *device_values) {
  inverse_contiguous_tile_kernel<<<
      dim3(impl.contiguous_tile_block_count, static_cast<unsigned>(impl.limb_count)),
      impl.threads_per_block, impl.contiguous_tile_length * sizeof(uint64_t)>>>(
      device_values, impl.device_inverse_twiddle_powers, impl.device_moduli,
      impl.transform_length, static_cast<uint32_t>(impl.contiguous_tile_length));
  CHECK_CUDA(cudaGetLastError());
  if (impl.transform_length > kPassSplitWidth) {
    inverse_strided_group_kernel<<<
        dim3(kPassSplitWidth, static_cast<unsigned>(impl.limb_count)),
        impl.threads_per_block, impl.strided_group_length * sizeof(uint64_t)>>>(
        device_values, impl.device_inverse_twiddle_powers, impl.device_moduli,
        impl.transform_length, static_cast<uint32_t>(impl.strided_group_length));
    CHECK_CUDA(cudaGetLastError());
  }
  inverse_scale_kernel<<<impl.elementwise_block_count, impl.threads_per_block>>>(
      device_values, impl.device_inverse_transform_lengths, impl.device_moduli,
      impl.transform_length, impl.limb_count);
  CHECK_CUDA(cudaGetLastError());
}

} // namespace

// --- CudaNttTransform ----------------------------------------------------
CudaNttTransform::CudaNttTransform(uint64_t transform_length,
                                   const std::vector<uint64_t> &moduli,
                                   const std::vector<uint64_t> &primitive_roots)
    : CudaNttTransform(transform_length, moduli, primitive_roots, 0, nullptr) {}

CudaNttTransform::CudaNttTransform(uint64_t transform_length,
                                   const std::vector<uint64_t> &moduli,
                                   const std::vector<uint64_t> &primitive_roots,
                                   size_t failure_step, detail::SetupTrace *trace)
    : impl_(std::make_unique<Impl>(failure_step, trace)) {
  Impl &impl = *impl_;
  const auto shape =
      detail::validate_parameters(transform_length, moduli, primitive_roots);
  impl.transform_length = transform_length;
  impl.limb_count = moduli.size();
  impl.moduli = moduli;
  impl.contiguous_tile_length = std::min<uint64_t>(kPassSplitWidth, transform_length);
  impl.strided_group_length = transform_length / kPassSplitWidth;
  // Tables from the reference arithmetic; NegacyclicNtt's constructor
  // validates every wheel/root (throws) before any allocation happens.
  std::vector<uint64_t> twiddle_powers;
  std::vector<uint64_t> inverse_twiddle_powers;
  std::vector<uint64_t> inverse_transform_lengths;
  twiddle_powers.reserve(shape.twiddles);
  inverse_twiddle_powers.reserve(shape.twiddles);
  for (uint64_t limb_index = 0; limb_index < impl.limb_count; ++limb_index) {
    const uint64_t modulus = moduli[limb_index];
    const uint64_t primitive_root = primitive_roots[limb_index];
    impl.cpu_references.emplace_back(transform_length, modulus, primitive_root);
    const uint64_t inverse_primitive_root = inv_mod(primitive_root, modulus);
    uint64_t twiddle_power = 1;
    uint64_t inverse_twiddle_power = 1;
    for (uint64_t exponent = 0; exponent <= transform_length; ++exponent) {
      twiddle_powers.push_back(twiddle_power);
      inverse_twiddle_powers.push_back(inverse_twiddle_power);
      twiddle_power = mul_mod(twiddle_power, primitive_root, modulus);
      inverse_twiddle_power =
          mul_mod(inverse_twiddle_power, inverse_primitive_root, modulus);
    }
    inverse_transform_lengths.push_back(inv_mod(transform_length, modulus));
  }
  const size_t value_count = shape.values;
  const size_t twiddle_count = shape.twiddles;
  CHECK_CUDA(cudaGetDevice(&impl.device));
  impl.setup.runtime().device = impl.device;
  impl.device_operand_a = impl.setup.allocate(value_count);
  impl.device_operand_b = impl.setup.allocate(value_count);
  impl.device_twiddle_powers = impl.setup.allocate(twiddle_count);
  impl.device_inverse_twiddle_powers = impl.setup.allocate(twiddle_count);
  impl.device_moduli = impl.setup.allocate(impl.limb_count);
  impl.device_inverse_transform_lengths = impl.setup.allocate(impl.limb_count);
  impl.setup.upload(impl.device_twiddle_powers, twiddle_powers.data(), twiddle_count);
  impl.setup.upload(impl.device_inverse_twiddle_powers,
                    inverse_twiddle_powers.data(), twiddle_count);
  impl.setup.upload(impl.device_moduli, moduli.data(), impl.limb_count);
  impl.setup.upload(impl.device_inverse_transform_lengths,
                    inverse_transform_lengths.data(), impl.limb_count);
  impl.contiguous_tile_block_count =
      static_cast<uint32_t>(std::max<uint64_t>(1, transform_length / kPassSplitWidth));
  impl.elementwise_block_count =
      static_cast<uint32_t>((value_count + kThreadsPerBlock - 1) / kThreadsPerBlock);
}

CudaNttTransform::~CudaNttTransform() = default;

void check_native_setup_failures(std::ostream &output) {
  const auto moduli = find_primes(65536, 3, 50);
  std::vector<uint64_t> roots;
  for (auto modulus : moduli) roots.push_back(find_psi(modulus, 65536));
  for (size_t step = 0; step <= 10; ++step) {
    detail::SetupTrace trace;
    bool injected = false;
    try {
      CudaNttTransform transform(65536, moduli, roots, step, &trace);
    } catch (const detail::InjectedSetupFailure &) { injected = true; }
    const size_t expected = step ? std::min(step - 1, size_t{6}) : 6;
    if (injected != (step != 0) || trace.allocations != expected ||
        trace.releases != expected || trace.release_failures != 0 ||
        trace.uploads != (step ? (step > 7 ? step - 7 : 0) : 4))
      throw std::runtime_error("native setup fault cleanup mismatch");
    output << "SETUP_NATIVE step=" << step << " allocations=" << trace.allocations
           << " releases=" << trace.releases << " passed\n";
  }
}

uint64_t CudaNttTransform::n() const {
  return impl_->transform_length;
}
uint64_t CudaNttTransform::limbs() const {
  return impl_->limb_count;
}
uint64_t CudaNttTransform::q(uint64_t limb_index) const {
  return impl_->moduli.at(limb_index);
}
const NegacyclicNtt &CudaNttTransform::ref(uint64_t limb_index) const {
  return impl_->cpu_references.at(limb_index);
}

void CudaNttTransform::forward(std::vector<uint64_t> &values) {
  const size_t value_count = impl_->limb_count * impl_->transform_length;
  detail::validate_values(impl_->transform_length, impl_->moduli, values);
  DeviceGuard guard(impl_->device);
  CHECK_CUDA(cudaMemcpy(impl_->device_operand_a, values.data(),
                        value_count * sizeof(uint64_t), cudaMemcpyHostToDevice));
  launch_forward_transform(*impl_, impl_->device_operand_a);
  CHECK_CUDA(cudaMemcpy(values.data(), impl_->device_operand_a,
                        value_count * sizeof(uint64_t), cudaMemcpyDeviceToHost));
}

void CudaNttTransform::inverse(std::vector<uint64_t> &values) {
  const size_t value_count = impl_->limb_count * impl_->transform_length;
  detail::validate_values(impl_->transform_length, impl_->moduli, values);
  DeviceGuard guard(impl_->device);
  CHECK_CUDA(cudaMemcpy(impl_->device_operand_a, values.data(),
                        value_count * sizeof(uint64_t), cudaMemcpyHostToDevice));
  launch_inverse_transform(*impl_, impl_->device_operand_a);
  CHECK_CUDA(cudaMemcpy(values.data(), impl_->device_operand_a,
                        value_count * sizeof(uint64_t), cudaMemcpyDeviceToHost));
}

void CudaNttTransform::pointwise_multiply(std::vector<uint64_t> &left_values,
                                          const std::vector<uint64_t> &right_values) {
  const size_t value_count = impl_->limb_count * impl_->transform_length;
  detail::validate_values(impl_->transform_length, impl_->moduli, left_values);
  detail::validate_values(impl_->transform_length, impl_->moduli, right_values);
  DeviceGuard guard(impl_->device);
  CHECK_CUDA(cudaMemcpy(impl_->device_operand_a, left_values.data(),
                        value_count * sizeof(uint64_t), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(impl_->device_operand_b, right_values.data(),
                        value_count * sizeof(uint64_t), cudaMemcpyHostToDevice));
  pointwise_multiply_kernel<<<impl_->elementwise_block_count,
                              impl_->threads_per_block>>>(
      impl_->device_operand_a, impl_->device_operand_b, impl_->device_moduli,
      impl_->transform_length, impl_->limb_count);
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaMemcpy(left_values.data(), impl_->device_operand_a,
                        value_count * sizeof(uint64_t), cudaMemcpyDeviceToHost));
}

std::vector<uint64_t>
CudaNttTransform::multiply(std::vector<uint64_t> left_values,
                           const std::vector<uint64_t> &right_values) {
  const size_t value_count = impl_->limb_count * impl_->transform_length;
  detail::validate_values(impl_->transform_length, impl_->moduli, left_values);
  detail::validate_values(impl_->transform_length, impl_->moduli, right_values);
  DeviceGuard guard(impl_->device);
  // Everything below stays on device: two forwards, one pointwise, one
  // inverse — the round trips this class exists to remove.
  CHECK_CUDA(cudaMemcpy(impl_->device_operand_a, left_values.data(),
                        value_count * sizeof(uint64_t), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(impl_->device_operand_b, right_values.data(),
                        value_count * sizeof(uint64_t), cudaMemcpyHostToDevice));
  launch_forward_transform(*impl_, impl_->device_operand_a);
  launch_forward_transform(*impl_, impl_->device_operand_b);
  pointwise_multiply_kernel<<<impl_->elementwise_block_count,
                              impl_->threads_per_block>>>(
      impl_->device_operand_a, impl_->device_operand_b, impl_->device_moduli,
      impl_->transform_length, impl_->limb_count);
  CHECK_CUDA(cudaGetLastError());
  launch_inverse_transform(*impl_, impl_->device_operand_a);
  CHECK_CUDA(cudaMemcpy(left_values.data(), impl_->device_operand_a,
                        value_count * sizeof(uint64_t), cudaMemcpyDeviceToHost));
  return left_values;
}

// --- bench ---------------------------------------------------------------
void describe_device(std::ostream &output) {
  int count = 0, device = 0, driver = 0, runtime = 0;
  CHECK_CUDA(cudaGetDeviceCount(&count));
  if (count == 0)
    throw std::runtime_error("no visible CUDA device");
  CHECK_CUDA(cudaGetDevice(&device));
  cudaDeviceProp props{};
  CHECK_CUDA(cudaGetDeviceProperties(&props, device));
  CHECK_CUDA(cudaDriverGetVersion(&driver));
  CHECK_CUDA(cudaRuntimeGetVersion(&runtime));
  output << "visible devices: " << count << "\nselected device: " << device
         << "\nname: " << props.name << "\ncompute capability: " << props.major << '.'
         << props.minor << "\nmemory bytes: " << props.totalGlobalMem
         << "\ndriver API: " << driver << "\nruntime API: " << runtime << '\n';
}
void run_bench(uint32_t minimum_size_exponent, uint32_t maximum_size_exponent,
               uint64_t limb_count, std::ostream &output) {
  if (minimum_size_exponent < 1 || maximum_size_exponent > 16 ||
      minimum_size_exponent > maximum_size_exponent || limb_count == 0 ||
      limb_count > 65535)
    throw std::invalid_argument("bench requires 1<=min<=max<=16 and 1<=limbs<=65535");
  cudaDeviceProp device_properties{};
  int device_index = 0;
  CHECK_CUDA(cudaGetDevice(&device_index));
  CHECK_CUDA(cudaGetDeviceProperties(&device_properties, device_index));
  int driver_version = 0;
  int runtime_version = 0;
  CHECK_CUDA(cudaDriverGetVersion(&driver_version));
  CHECK_CUDA(cudaRuntimeGetVersion(&runtime_version));
  output << "device: " << device_properties.name << " (sm_" << device_properties.major
         << device_properties.minor << ", " << device_properties.totalGlobalMem
         << " device bytes)\n";
  output << "driver/runtime API: " << driver_version / 1000 << "."
         << (driver_version % 1000) / 10 << " / " << runtime_version / 1000 << "."
         << (runtime_version % 1000) / 10 << "\n";
  output << "v1 arithmetic: __uint128_t division (pre-Barrett; see README)\n";
  output << "columns are end-to-end per call: transfers included\n\n";
  output
      << "baseline parameters: 46-bit primes; host validation/device guards included\n";
  output << "| n | limbs | ref fwd ms | gpu fwd ms | gpu inv ms | gpu mul ms |\n";
  output << "|---|---|---|---|---|---|\n";

  constexpr uint32_t kWarmupRepetitions = 2;
  constexpr uint32_t kTimedRepetitions = 10;
  auto elapsed_milliseconds = [](const auto &start_time, const auto &end_time) {
    return std::chrono::duration<double, std::milli>(end_time - start_time).count();
  };
  for (uint32_t size_exponent = minimum_size_exponent;
       size_exponent <= maximum_size_exponent; ++size_exponent) {
    const uint64_t transform_length = uint64_t{1} << size_exponent;
    const std::vector<uint64_t> moduli = find_primes(transform_length, limb_count);
    std::vector<uint64_t> primitive_roots;
    for (uint64_t modulus : moduli)
      primitive_roots.push_back(find_psi(modulus, transform_length));
    CudaNttTransform transform(transform_length, moduli, primitive_roots);

    // Deterministic per-limb random operands (bench only needs data, not
    // provenance — the correctness gates elsewhere own verification).
    std::vector<uint64_t> left_operand(limb_count * transform_length);
    std::vector<uint64_t> right_operand(limb_count * transform_length);
    uint64_t random_state = 0x9E3779B97F4A7C15ULL;
    for (uint64_t limb_index = 0; limb_index < limb_count; ++limb_index)
      for (uint64_t value_index = 0; value_index < transform_length; ++value_index) {
        random_state = random_state * 6364136223846793005ULL + 1442695040888963407ULL;
        left_operand[limb_index * transform_length + value_index] =
            random_state % moduli[limb_index];
        random_state = random_state * 6364136223846793005ULL + 1442695040888963407ULL;
        right_operand[limb_index * transform_length + value_index] =
            random_state % moduli[limb_index];
      }

    // CPU reference forward, same data, host wall clock.
    double reference_forward_milliseconds = 0;
    {
      std::vector<std::vector<uint64_t>> reference_values(limb_count);
      for (uint64_t limb_index = 0; limb_index < limb_count; ++limb_index)
        reference_values[limb_index].assign(
            left_operand.begin() + limb_index * transform_length,
            left_operand.begin() + (limb_index + 1) * transform_length);
      for (uint32_t repetition = 0; repetition < kWarmupRepetitions; ++repetition)
        for (uint64_t limb_index = 0; limb_index < limb_count; ++limb_index)
          transform.ref(limb_index).forward(reference_values[limb_index]);
      double total_milliseconds = 0;
      for (uint32_t repetition = 0; repetition < kTimedRepetitions; ++repetition) {
        const auto start_time = std::chrono::steady_clock::now();
        for (uint64_t limb_index = 0; limb_index < limb_count; ++limb_index)
          transform.ref(limb_index).forward(reference_values[limb_index]);
        total_milliseconds +=
            elapsed_milliseconds(start_time, std::chrono::steady_clock::now());
      }
      reference_forward_milliseconds = total_milliseconds / kTimedRepetitions;
    }

    // GPU columns: wall clock around each public call — the final device
    // copy blocks, so the measurement is honest end-to-end. Buffers are
    // re-filled outside the timed region; content does not affect timing.
    double gpu_forward_milliseconds = 0;
    double gpu_inverse_milliseconds = 0;
    double gpu_multiply_milliseconds = 0;
    {
      std::vector<uint64_t> working_values(left_operand);
      for (uint32_t repetition = 0; repetition < kWarmupRepetitions; ++repetition)
        transform.forward(working_values);
      double total_milliseconds = 0;
      for (uint32_t repetition = 0; repetition < kTimedRepetitions; ++repetition) {
        working_values = left_operand;
        const auto start_time = std::chrono::steady_clock::now();
        transform.forward(working_values);
        total_milliseconds +=
            elapsed_milliseconds(start_time, std::chrono::steady_clock::now());
      }
      gpu_forward_milliseconds = total_milliseconds / kTimedRepetitions;
      total_milliseconds = 0;
      for (uint32_t repetition = 0; repetition < kTimedRepetitions; ++repetition) {
        const auto start_time = std::chrono::steady_clock::now();
        transform.inverse(working_values);
        total_milliseconds +=
            elapsed_milliseconds(start_time, std::chrono::steady_clock::now());
      }
      gpu_inverse_milliseconds = total_milliseconds / kTimedRepetitions;
      for (uint32_t repetition = 0; repetition < kWarmupRepetitions; ++repetition)
        transform.multiply(left_operand, right_operand);
      total_milliseconds = 0;
      for (uint32_t repetition = 0; repetition < kTimedRepetitions; ++repetition) {
        const auto start_time = std::chrono::steady_clock::now();
        transform.multiply(left_operand, right_operand);
        total_milliseconds +=
            elapsed_milliseconds(start_time, std::chrono::steady_clock::now());
      }
      gpu_multiply_milliseconds = total_milliseconds / kTimedRepetitions;
    }

    output << "| 2^" << size_exponent << " (" << transform_length << ") | "
           << limb_count << " | " << reference_forward_milliseconds << " | "
           << gpu_forward_milliseconds << " | " << gpu_inverse_milliseconds << " | "
           << gpu_multiply_milliseconds << " |\n";
    output.flush();
  }
}

void run_bench_samples(uint64_t dimension, const std::vector<uint64_t> &moduli,
                       const std::vector<uint64_t> &roots, size_t warmups,
                       size_t samples, std::ostream &output, bool event_timing) {
  measurement::validate_counts(warmups, samples);
  CudaNttTransform transform(dimension, moduli, roots);
  cudaDeviceProp properties{};
  int device, driver, runtime;
  CHECK_CUDA(cudaGetDevice(&device));
  CHECK_CUDA(cudaGetDeviceProperties(&properties, device));
  CHECK_CUDA(cudaDriverGetVersion(&driver));
  CHECK_CUDA(cudaRuntimeGetVersion(&runtime));
  const size_t count = dimension * moduli.size();
  const auto [left, right] = measurement::benchmark_inputs(dimension, moduli, roots);
  auto cpu_transform = [&](std::vector<uint64_t> &values) {
    for (size_t tower = 0; tower < moduli.size(); ++tower) {
      std::vector<uint64_t> limb(values.begin() + tower * dimension,
                                 values.begin() + (tower + 1) * dimension);
      transform.ref(tower).forward(limb);
      std::copy(limb.begin(), limb.end(), values.begin() + tower * dimension);
    }
  };
  auto evaluation = left, right_evaluation = right;
  cpu_transform(evaluation);
  cpu_transform(right_evaluation);
  auto expected_product = evaluation;
  for (size_t index = 0; index < count; ++index)
    expected_product[index] = mul_mod(evaluation[index], right_evaluation[index],
                                     moduli[index / dimension]);
  auto expected_coefficients = expected_product;
  for (size_t tower = 0; tower < moduli.size(); ++tower) {
    std::vector<uint64_t> limb(expected_coefficients.begin() + tower * dimension,
                               expected_coefficients.begin() + (tower + 1) * dimension);
    transform.ref(tower).inverse(limb);
    std::copy(limb.begin(), limb.end(), expected_coefficients.begin() + tower * dimension);
  }
  output << "{\"type\":\"workload\",\"schema\":1,\"n\":" << dimension
         << ",\"towers\":" << moduli.size() << ",\"warmups\":" << warmups
         << ",\"samples\":" << samples << ",\"timing\":\""
         << (event_timing ? "cuda_event_resident_and_pinned_copy" : "host_wall_public_call") << "\","
            "\"input_policy\":\"fixed_seed_reset_outside_timer\",\"seed\":7492057,"
            "\"setup_included\":false,\"device\":";
  measurement::json_string(output, properties.name);
  output << ",\"compute_major\":" << properties.major << ",\"compute_minor\":" << properties.minor
         << ",\"device_bytes\":" << properties.totalGlobalMem
         << ",\"driver_api\":" << driver << ",\"runtime_api\":" << runtime << ",\"moduli\":[";
  for (size_t tower = 0; tower < moduli.size(); ++tower) {
    if (tower) output << ',';
    output << moduli[tower];
  }
  output << "],\"roots\":[";
  for (size_t tower = 0; tower < roots.size(); ++tower) {
    if (tower) output << ',';
    output << roots[tower];
  }
  output << "]}\n";
  if (event_timing) {
    // Standalone event experiments. Public methods are deliberately untouched.
    // These scopes are not additive components of host-wall call latency.
    struct Event {
      cudaEvent_t handle{};
      Event() { CHECK_CUDA(cudaEventCreate(&handle)); }
      ~Event() { (void)cudaEventDestroy(handle); }
      Event(const Event &) = delete;
      Event &operator=(const Event &) = delete;
    };
    struct PinnedBuffer {
      uint64_t *values = nullptr;
      explicit PinnedBuffer(size_t bytes) {
        CHECK_CUDA(cudaMallocHost(reinterpret_cast<void **>(&values), bytes));
      }
      ~PinnedBuffer() { (void)cudaFreeHost(values); }
      PinnedBuffer(const PinnedBuffer &) = delete;
      PinnedBuffer &operator=(const PinnedBuffer &) = delete;
    };
    auto &impl = *transform.impl_;
    DeviceGuard guard(impl.device);
    Event start, finish;
    const size_t bytes = count * sizeof(uint64_t);
    PinnedBuffer pinned(bytes);
    std::vector<uint64_t> observed(count);
    auto upload = [&](uint64_t *destination, const std::vector<uint64_t> &source) {
      CHECK_CUDA(cudaMemcpy(destination, source.data(), bytes, cudaMemcpyHostToDevice));
    };
    auto verify_device = [&](const std::vector<uint64_t> &expected) {
      CHECK_CUDA(cudaMemcpy(observed.data(), impl.device_operand_a, bytes, cudaMemcpyDeviceToHost));
      if (observed != expected) throw std::runtime_error("event benchmark result mismatch");
    };
    auto measure_event = [&](const std::string &name, auto prepare, auto operation, auto verify) {
      auto timings = measurement::collect_elapsed(warmups, samples, [&] {
        prepare();
        CHECK_CUDA(cudaStreamSynchronize(0));
      }, [&] {
        CHECK_CUDA(cudaEventRecord(start.handle, 0));
        operation();
        CHECK_CUDA(cudaEventRecord(finish.handle, 0));
        CHECK_CUDA(cudaEventSynchronize(finish.handle));
        float milliseconds = 0;
        CHECK_CUDA(cudaEventElapsedTime(&milliseconds, start.handle, finish.handle));
        return static_cast<double>(milliseconds);
      }, verify);
      measurement::write_samples(output, name, timings);
    };
    auto launch_pointwise = [&] {
      pointwise_multiply_kernel<<<impl.elementwise_block_count, impl.threads_per_block>>>(
          impl.device_operand_a, impl.device_operand_b, impl.device_moduli,
          dimension, impl.limb_count);
      CHECK_CUDA(cudaGetLastError());
    };
    measure_event("gpu_forward", [&] { upload(impl.device_operand_a, left); },
        [&] { launch_forward_transform(impl, impl.device_operand_a); },
        [&] { verify_device(evaluation); });
    measure_event("gpu_inverse", [&] { upload(impl.device_operand_a, evaluation); },
        [&] { launch_inverse_transform(impl, impl.device_operand_a); },
        [&] { verify_device(left); });
    measure_event("gpu_pointwise", [&] {
      upload(impl.device_operand_a, evaluation); upload(impl.device_operand_b, right_evaluation);
    }, launch_pointwise, [&] { verify_device(expected_product); });
    measure_event("gpu_fused_multiply", [&] {
      upload(impl.device_operand_a, left); upload(impl.device_operand_b, right);
    }, [&] {
      launch_forward_transform(impl, impl.device_operand_a);
      launch_forward_transform(impl, impl.device_operand_b);
      launch_pointwise();
      launch_inverse_transform(impl, impl.device_operand_a);
    }, [&] { verify_device(expected_coefficients); });
    measure_event("h2d_pinned", [&] {
      std::copy(left.begin(), left.end(), pinned.values);
      CHECK_CUDA(cudaMemset(impl.device_operand_a, 0, bytes));
    }, [&] {
      CHECK_CUDA(cudaMemcpyAsync(impl.device_operand_a, pinned.values, bytes, cudaMemcpyHostToDevice, 0));
    }, [&] { verify_device(left); });
    measure_event("d2h_pinned", [&] {
      upload(impl.device_operand_a, left);
      std::fill(pinned.values, pinned.values + count, 0);
    }, [&] {
      CHECK_CUDA(cudaMemcpyAsync(pinned.values, impl.device_operand_a, bytes, cudaMemcpyDeviceToHost, 0));
    }, [&] {
      if (!std::equal(left.begin(), left.end(), pinned.values))
        throw std::runtime_error("pinned D2H benchmark result mismatch");
    });
    output << "{\"type\":\"complete\",\"operations\":6}\n";
    return;
  }
  std::vector<uint64_t> working, other, product;
  auto measure = [&](const std::string &name, auto prepare, auto operation,
                     const std::vector<uint64_t> &expected, std::vector<uint64_t> &result) {
    auto timings = measurement::collect(warmups, samples, prepare, operation);
    if (result != expected)
      throw std::runtime_error("benchmark result mismatch: " + name);
    measurement::write_samples(output, name, timings);
  };
  std::vector<std::vector<uint64_t>> cpu_values(moduli.size(),
                                                std::vector<uint64_t>(dimension));
  auto cpu_samples = measurement::collect(warmups, samples, [&] {
    for (size_t tower = 0; tower < moduli.size(); ++tower)
      std::copy(left.begin() + tower * dimension, left.begin() + (tower + 1) * dimension,
                cpu_values[tower].begin());
  }, [&] {
    for (size_t tower = 0; tower < moduli.size(); ++tower)
      transform.ref(tower).forward(cpu_values[tower]);
  });
  for (size_t tower = 0; tower < moduli.size(); ++tower)
    if (!std::equal(cpu_values[tower].begin(), cpu_values[tower].end(),
                    evaluation.begin() + tower * dimension))
      throw std::runtime_error("CPU benchmark result mismatch");
  measurement::write_samples(output, "cpu_reference_forward", cpu_samples);
  measure("gpu_forward", [&] { working = left; },
          [&] { transform.forward(working); }, evaluation, working);
  measure("gpu_inverse", [&] { working = evaluation; },
          [&] { transform.inverse(working); }, left, working);
  measure("gpu_pointwise", [&] { working = evaluation; },
          [&] { transform.pointwise_multiply(working, right_evaluation); }, expected_product, working);
  measure("gpu_unfused_multiply", [&] { working = left; other = right; }, [&] {
    transform.forward(working); transform.forward(other);
    transform.pointwise_multiply(working, other); transform.inverse(working);
  }, expected_coefficients, working);
  measure("gpu_fused_multiply", [&] { std::vector<uint64_t>().swap(product); }, [&] {
    product = transform.multiply(left, right);
  }, expected_coefficients, product);
  output << "{\"type\":\"complete\",\"operations\":6}\n";
}

} // namespace ntt::cuda
