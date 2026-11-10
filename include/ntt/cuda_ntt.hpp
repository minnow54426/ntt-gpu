#pragma once
// The CUDA backend's public face — deliberately free of CUDA headers so
// ordinary C++ TUs (src/cuda_cli.cpp) can drive it; the implementation and
// every cudaRuntime call live in src/cuda/ntt_kernels.cu (pimpl).
//
// Contract: identical to the CPU reference (include/ntt/ntt.hpp).
// forward consumes natural-order coefficients and emits values in
// bit-reversed evaluation order; inverse is its exact inverse; multiply
// is the ring product mod x^n+1. The GPU gates assert ARRAY equality
// with NegacyclicNtt on identical vectors — never approximate agreement.
//
// Data layout: one flat limb-major buffer of limbs()*n values; every
// transform launch batches RNS limbs as blockIdx.y. Elementwise kernels
// use a flat K*n grid and recover the limb from index/n. One public call
// covers all limbs; pointwise products stay limb-local.

#include <cstdint>
#include <iosfwd>
#include <vector>

namespace ntt {
class NegacyclicNtt; // complete type in ntt/ntt.hpp; a reference here
}

namespace ntt::cuda {

class CudaNttTransform {
public:
  // n: power-of-two transform length; q[k]/psi[k]: per-limb prime and
  // primitive 2n-th root. Parameters are validated by constructing the
  // CPU reference per limb (throws the same exceptions as
  // NegacyclicNtt) before device buffers are allocated; twiddle tables are
  // built with the reference arithmetic and uploaded once.
  CudaNttTransform(uint64_t transform_length, const std::vector<uint64_t> &moduli,
                   const std::vector<uint64_t> &primitive_roots);
  ~CudaNttTransform();

  CudaNttTransform(const CudaNttTransform &) = delete;
  CudaNttTransform &operator=(const CudaNttTransform &) = delete;

  uint64_t n() const;
  uint64_t limbs() const;
  uint64_t q(uint64_t limb_index) const;
  // The CPU reference for limb k on the exact same parameters — verify
  // and bench compare against this instead of re-deriving it, so GPU and
  // ref can never disagree about the inputs they were given.
  const ntt::NegacyclicNtt &ref(uint64_t limb_index) const;

  // In place on limb-major host data (size limbs()*n): upload, run the
  // whole stage schedule on device, download. Not const: the device
  // buffers are scratch state.
  void forward(std::vector<uint64_t> &values);
  void inverse(std::vector<uint64_t> &values);
  // a *= b elementwise in the value domain (both already transformed,
  // same bit-reversed order — so no unscrambling between them).
  void pointwise_multiply(std::vector<uint64_t> &left_values,
                          const std::vector<uint64_t> &right_values);
  // The full ring product: forward(a), forward(b), pointwise, inverse.
  std::vector<uint64_t> multiply(std::vector<uint64_t> left_values,
                                 const std::vector<uint64_t> &right_values);

private:
  struct Impl; // CUDA pointers and launch geometry live in the .cu
  Impl *impl_;
};

// Timing sweep for the bench subcommand (implemented in the .cu): for
// each n = 2^min_exp .. 2^max_exp, fresh 46-bit parameters (param_gen.hpp,
// `limbs` of them), warmup + timed repetitions of forward / inverse /
// multiply, wall clock around each public call (transfers included — the
// table header says so) plus the CPU reference for scale. Prints to
// `out`, headed by the device/driver/toolkit names — numbers are only
// meaningful attached to that hardware header, and are never quoted from
// an unnamed machine (README rule).
void run_bench(uint32_t minimum_size_exponent, uint32_t maximum_size_exponent,
               uint64_t limb_count, std::ostream &output);

} // namespace ntt::cuda
