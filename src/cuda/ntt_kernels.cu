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

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <ostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "ntt/cuda_ntt.hpp"
#include "ntt/dev_core.hpp"
#include "ntt/mod_arith.hpp"
#include "ntt/ntt.hpp"
#include "ntt/param_gen.hpp"

// Every cudaRuntime call funnels through this: a failed allocation,
// launch, or copy must fail the gates loudly (nonzero exit + stderr),
// never surface as silently wrong numbers.
#define CHECK_CUDA(call)                                              \
  do {                                                                \
    const cudaError_t _e = (call);                                    \
    if (_e != cudaSuccess)                                            \
      throw std::runtime_error(std::string("CUDA: ") + #call + ": " + \
                               cudaGetErrorString(_e));               \
  } while (0)

namespace ntt::cuda {
namespace {

// Must match ScheduleSim (src/device_sim.cpp): the tile width B and
// block size T define the two-phase split the simulator already proved
// against the CPU reference at n = 64 / 1024 / 4096.
constexpr uint32_t kThreads = 256;
constexpr uint64_t kTile = 1024;

// --- shared-memory phase: one block per B-wide tile per limb ------------
// The tile hosts stages of the GLOBAL transform: dev_dif/dev_dit take
// the global n for twiddle scales (the trap dev_core.hpp pins). All
// threads reach every __syncthreads(): the barriers sit between stages,
// never inside conditionals or the pair loop.
__global__ void dif_shared_kernel(uint64_t* a, const uint64_t* pp,
                                  const uint64_t* q, uint64_t n, uint32_t B) {
  extern __shared__ uint64_t smem[];
  uint64_t* al = a + static_cast<uint64_t>(blockIdx.y) * n;
  const uint64_t* pl = pp + static_cast<uint64_t>(blockIdx.y) * (n + 1);
  const uint64_t ql = q[blockIdx.y];
  const uint64_t base = static_cast<uint64_t>(blockIdx.x) * B;
  for (uint32_t t = threadIdx.x; t < B; t += blockDim.x)
    smem[t] = al[base + t];
  __syncthreads();
  for (uint64_t d = B / 2; d >= 1; d >>= 1) {
    dev_dif_stage(smem, B, d, n, pl, ql, threadIdx.x, blockDim.x);
    __syncthreads();
  }
  for (uint32_t t = threadIdx.x; t < B; t += blockDim.x)
    al[base + t] = smem[t];
}

__global__ void dit_shared_kernel(uint64_t* a, const uint64_t* pp,
                                  const uint64_t* q, uint64_t n, uint32_t B) {
  extern __shared__ uint64_t smem[];
  uint64_t* al = a + static_cast<uint64_t>(blockIdx.y) * n;
  const uint64_t* pl = pp + static_cast<uint64_t>(blockIdx.y) * (n + 1);
  const uint64_t ql = q[blockIdx.y];
  const uint64_t base = static_cast<uint64_t>(blockIdx.x) * B;
  for (uint32_t t = threadIdx.x; t < B; t += blockDim.x)
    smem[t] = al[base + t];
  __syncthreads();
  for (uint64_t d = 1; d <= B / 2; d <<= 1) {
    dev_dit_stage(smem, B, d, n, pl, ql, threadIdx.x, blockDim.x);
    __syncthreads();
  }
  for (uint32_t t = threadIdx.x; t < B; t += blockDim.x)
    al[base + t] = smem[t];
}

// --- global phase: one launch per stage, grid-stride over the pairs -----
// The shared body treats the global thread id/count as tid/nthr — the
// same disjoint pair set the simulator walks serially, so the result is
// identical regardless of launch geometry. n/2 pairs fit 32 bits for
// every n this repo targets (n <= 2^18).
__global__ void dif_global_kernel(uint64_t* a, const uint64_t* pp,
                                  const uint64_t* q, uint64_t n, uint64_t d) {
  uint64_t* al = a + static_cast<uint64_t>(blockIdx.y) * n;
  const uint64_t* pl = pp + static_cast<uint64_t>(blockIdx.y) * (n + 1);
  const uint64_t tid =
      static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const uint64_t nthr = static_cast<uint64_t>(gridDim.x) * blockDim.x;
  dev_dif_stage(al, n, d, n, pl, q[blockIdx.y],
                static_cast<uint32_t>(tid), static_cast<uint32_t>(nthr));
}

__global__ void dit_global_kernel(uint64_t* a, const uint64_t* pp,
                                  const uint64_t* q, uint64_t n, uint64_t d) {
  uint64_t* al = a + static_cast<uint64_t>(blockIdx.y) * n;
  const uint64_t* pl = pp + static_cast<uint64_t>(blockIdx.y) * (n + 1);
  const uint64_t tid =
      static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const uint64_t nthr = static_cast<uint64_t>(gridDim.x) * blockDim.x;
  dev_dit_stage(al, n, d, n, pl, q[blockIdx.y],
                static_cast<uint32_t>(tid), static_cast<uint32_t>(nthr));
}

// --- elementwise: the n^-1 scale and the pointwise product --------------
__global__ void scale_kernel(uint64_t* a, const uint64_t* n_inv,
                             const uint64_t* q, uint64_t n, uint64_t limbs) {
  const uint64_t total = limbs * n;
  const uint64_t stride =
      static_cast<uint64_t>(gridDim.x) * blockDim.x;
  for (uint64_t i = static_cast<uint64_t>(blockIdx.x) * blockDim.x +
                    threadIdx.x;
       i < total; i += stride) {
    const uint64_t k = i / n;  // limb-major: the limb owns n contiguous
    a[i] = mul_mod(a[i], n_inv[k], q[k]);
  }
}

__global__ void pointwise_kernel(uint64_t* a, const uint64_t* b,
                                 const uint64_t* q, uint64_t n,
                                 uint64_t limbs) {
  const uint64_t total = limbs * n;
  const uint64_t stride =
      static_cast<uint64_t>(gridDim.x) * blockDim.x;
  for (uint64_t i = static_cast<uint64_t>(blockIdx.x) * blockDim.x +
                    threadIdx.x;
       i < total; i += stride) {
    a[i] = mul_mod(a[i], b[i], q[i / n]);
  }
}

}  // namespace

// --- Impl: device buffers + launch geometry ------------------------------
struct CudaNttTransform::Impl {
  uint64_t n = 0, P = 0, B = 0;
  uint32_t T = kThreads;
  std::vector<uint64_t> q;            // host copy (reporting)
  std::vector<NegacyclicNtt> refs;    // per-limb CPU reference: the contract
  uint64_t* d_a = nullptr;            // scratch operands, limb-major [P][n]
  uint64_t* d_b = nullptr;
  uint64_t* d_pp = nullptr;           // psi^e tables,      [P][n+1]
  uint64_t* d_pp_inv = nullptr;       // psi^-e tables,     [P][n+1]
  uint64_t* d_q = nullptr;            // per-limb primes,   [P]
  uint64_t* d_ninv = nullptr;         // per-limb n^-1,     [P]
  uint32_t pairs_blocks = 0;          // grid.x for global stages: ceil(n/2/T)
  uint32_t flat_blocks = 0;           // grid.x for elementwise: ceil(P*n/T)

  ~Impl() {  // noexcept by rule: frees ignore errors
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_pp);
    cudaFree(d_pp_inv);
    cudaFree(d_q);
    cudaFree(d_ninv);
  }
};

namespace {

// Device-resident drivers, one per public operation — exactly the launch
// sequences src/device_sim.cpp walks serially (global phase, then tiles
// for forward; tiles, then global phase, then scale for inverse).
void forward_dev(CudaNttTransform::Impl& m, uint64_t* buf) {
  for (uint64_t d = m.n / 2; d >= m.B; d >>= 1) {
    dif_global_kernel<<<dim3(m.pairs_blocks, static_cast<unsigned>(m.P)),
                        m.T>>>(buf, m.d_pp, m.d_q, m.n, d);
    CHECK_CUDA(cudaGetLastError());
  }
  dif_shared_kernel<<<dim3(static_cast<unsigned>(m.n / m.B),
                           static_cast<unsigned>(m.P)),
                      m.T, m.B * sizeof(uint64_t)>>>(
      buf, m.d_pp, m.d_q, m.n, static_cast<uint32_t>(m.B));
  CHECK_CUDA(cudaGetLastError());
}

void inverse_dev(CudaNttTransform::Impl& m, uint64_t* buf) {
  dit_shared_kernel<<<dim3(static_cast<unsigned>(m.n / m.B),
                           static_cast<unsigned>(m.P)),
                      m.T, m.B * sizeof(uint64_t)>>>(
      buf, m.d_pp_inv, m.d_q, m.n, static_cast<uint32_t>(m.B));
  CHECK_CUDA(cudaGetLastError());
  for (uint64_t d = m.B; d <= m.n / 2; d <<= 1) {
    dit_global_kernel<<<dim3(m.pairs_blocks, static_cast<unsigned>(m.P)),
                        m.T>>>(buf, m.d_pp_inv, m.d_q, m.n, d);
    CHECK_CUDA(cudaGetLastError());
  }
  scale_kernel<<<m.flat_blocks, m.T>>>(buf, m.d_ninv, m.d_q, m.n, m.P);
  CHECK_CUDA(cudaGetLastError());
}

}  // namespace

// --- CudaNttTransform ----------------------------------------------------
CudaNttTransform::CudaNttTransform(uint64_t n, const std::vector<uint64_t>& q,
                                   const std::vector<uint64_t>& psi)
    : impl_(new Impl{}) {
  Impl& m = *impl_;
  if (q.size() != psi.size() || q.empty())
    throw std::invalid_argument("one (q, psi) pair per limb required");
  m.n = n;
  m.P = q.size();
  m.q = q;
  m.B = std::min<uint64_t>(kTile, n);
  // Tables from the reference arithmetic; NegacyclicNtt's constructor
  // validates every wheel/root (throws) before any allocation happens.
  std::vector<uint64_t> pp, pp_inv, ninv;
  pp.reserve(m.P * (n + 1));
  pp_inv.reserve(m.P * (n + 1));
  for (uint64_t k = 0; k < m.P; ++k) {
    m.refs.emplace_back(n, q[k], psi[k]);
    const uint64_t psi_inv = inv_mod(psi[k], q[k]);
    uint64_t w = 1, wi = 1;
    for (uint64_t e = 0; e <= n; ++e) {  // pp[e] = psi^e, pp_inv[e] = psi^-e
      pp.push_back(w);
      pp_inv.push_back(wi);
      w = mul_mod(w, psi[k], q[k]);
      wi = mul_mod(wi, psi_inv, q[k]);
    }
    ninv.push_back(inv_mod(n, q[k]));
  }
  const size_t words = m.P * n, tbl = m.P * (n + 1);
  CHECK_CUDA(cudaMalloc(&m.d_a, words * sizeof(uint64_t)));
  CHECK_CUDA(cudaMalloc(&m.d_b, words * sizeof(uint64_t)));
  CHECK_CUDA(cudaMalloc(&m.d_pp, tbl * sizeof(uint64_t)));
  CHECK_CUDA(cudaMalloc(&m.d_pp_inv, tbl * sizeof(uint64_t)));
  CHECK_CUDA(cudaMalloc(&m.d_q, m.P * sizeof(uint64_t)));
  CHECK_CUDA(cudaMalloc(&m.d_ninv, m.P * sizeof(uint64_t)));
  CHECK_CUDA(cudaMemcpy(m.d_pp, pp.data(), tbl * sizeof(uint64_t),
                        cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(m.d_pp_inv, pp_inv.data(), tbl * sizeof(uint64_t),
                        cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(m.d_q, q.data(), m.P * sizeof(uint64_t),
                        cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(m.d_ninv, ninv.data(), m.P * sizeof(uint64_t),
                        cudaMemcpyHostToDevice));
  m.pairs_blocks =
      static_cast<uint32_t>((n / 2 + kThreads - 1) / kThreads);
  m.flat_blocks =
      static_cast<uint32_t>((words + kThreads - 1) / kThreads);
}

CudaNttTransform::~CudaNttTransform() { delete impl_; }

uint64_t CudaNttTransform::n() const { return impl_->n; }
uint64_t CudaNttTransform::limbs() const { return impl_->P; }
uint64_t CudaNttTransform::q(uint64_t k) const { return impl_->q.at(k); }
const NegacyclicNtt& CudaNttTransform::ref(uint64_t k) const {
  return impl_->refs.at(k);
}

void CudaNttTransform::forward(std::vector<uint64_t>& a) {
  const size_t words = impl_->P * impl_->n;
  if (a.size() != words)
    throw std::length_error("buffer size " + std::to_string(a.size()) +
                            " != limbs*n " + std::to_string(words));
  CHECK_CUDA(cudaMemcpy(impl_->d_a, a.data(), words * sizeof(uint64_t),
                        cudaMemcpyHostToDevice));
  forward_dev(*impl_, impl_->d_a);
  CHECK_CUDA(cudaMemcpy(a.data(), impl_->d_a, words * sizeof(uint64_t),
                        cudaMemcpyDeviceToHost));
}

void CudaNttTransform::inverse(std::vector<uint64_t>& a) {
  const size_t words = impl_->P * impl_->n;
  if (a.size() != words)
    throw std::length_error("buffer size " + std::to_string(a.size()) +
                            " != limbs*n " + std::to_string(words));
  CHECK_CUDA(cudaMemcpy(impl_->d_a, a.data(), words * sizeof(uint64_t),
                        cudaMemcpyHostToDevice));
  inverse_dev(*impl_, impl_->d_a);
  CHECK_CUDA(cudaMemcpy(a.data(), impl_->d_a, words * sizeof(uint64_t),
                        cudaMemcpyDeviceToHost));
}

void CudaNttTransform::pointwise_multiply(std::vector<uint64_t>& a,
                                          const std::vector<uint64_t>& b) {
  const size_t words = impl_->P * impl_->n;
  if (a.size() != words || b.size() != words)
    throw std::length_error("pointwise operands must both be limbs*n");
  CHECK_CUDA(cudaMemcpy(impl_->d_a, a.data(), words * sizeof(uint64_t),
                        cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(impl_->d_b, b.data(), words * sizeof(uint64_t),
                        cudaMemcpyHostToDevice));
  pointwise_kernel<<<impl_->flat_blocks, impl_->T>>>(
      impl_->d_a, impl_->d_b, impl_->d_q, impl_->n, impl_->P);
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaMemcpy(a.data(), impl_->d_a, words * sizeof(uint64_t),
                        cudaMemcpyDeviceToHost));
}

std::vector<uint64_t> CudaNttTransform::multiply(std::vector<uint64_t> a,
                                                 const std::vector<uint64_t>& b) {
  const size_t words = impl_->P * impl_->n;
  if (a.size() != words || b.size() != words)
    throw std::length_error("multiply operands must both be limbs*n");
  // Everything below stays on device: two forwards, one pointwise, one
  // inverse — the round trips this class exists to remove.
  CHECK_CUDA(cudaMemcpy(impl_->d_a, a.data(), words * sizeof(uint64_t),
                        cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(impl_->d_b, b.data(), words * sizeof(uint64_t),
                        cudaMemcpyHostToDevice));
  forward_dev(*impl_, impl_->d_a);
  forward_dev(*impl_, impl_->d_b);
  pointwise_kernel<<<impl_->flat_blocks, impl_->T>>>(
      impl_->d_a, impl_->d_b, impl_->d_q, impl_->n, impl_->P);
  CHECK_CUDA(cudaGetLastError());
  inverse_dev(*impl_, impl_->d_a);
  CHECK_CUDA(cudaMemcpy(a.data(), impl_->d_a, words * sizeof(uint64_t),
                        cudaMemcpyDeviceToHost));
  return a;
}

// --- bench ---------------------------------------------------------------
void run_bench(uint32_t min_exp, uint32_t max_exp, uint64_t limbs,
               std::ostream& out) {
  cudaDeviceProp prop{};
  int device = 0;
  CHECK_CUDA(cudaGetDevice(&device));
  CHECK_CUDA(cudaGetDeviceProperties(&prop, device));
  int driver = 0, runtime = 0;
  CHECK_CUDA(cudaDriverGetVersion(&driver));
  CHECK_CUDA(cudaRuntimeGetVersion(&runtime));
  out << "device: " << prop.name << " (sm_" << prop.major << prop.minor
      << ", " << prop.clockRate / 1000 << " MHz)\n";
  out << "driver/runtime API: " << driver / 1000 << "." << (driver % 1000) / 10
      << " / " << runtime / 1000 << "." << (runtime % 1000) / 10 << "\n";
  out << "v1 arithmetic: __uint128_t division (pre-Barrett; see README)\n";
  out << "columns are end-to-end per call: transfers included\n\n";
  out << "| n | limbs | ref fwd ms | gpu fwd ms | gpu inv ms | gpu mul ms |\n";
  out << "|---|---|---|---|---|---|\n";

  constexpr uint32_t kWarmup = 2, kReps = 10;
  auto clock_ms = [](const auto& t0, const auto& t1) {
    return std::chrono::duration<double, std::milli>(t1 - t0).count();
  };
  for (uint32_t e = min_exp; e <= max_exp; ++e) {
    const uint64_t n = uint64_t{1} << e;
    const std::vector<uint64_t> qs = find_primes(n, limbs);
    std::vector<uint64_t> psis;
    for (uint64_t p : qs) psis.push_back(find_psi(p, n));
    CudaNttTransform t(n, qs, psis);

    // Deterministic per-limb random operands (bench only needs data, not
    // provenance — the correctness gates elsewhere own verification).
    std::vector<uint64_t> a(limbs * n), b(limbs * n);
    uint64_t s = 0x9E3779B97F4A7C15ULL;
    for (uint64_t k = 0; k < limbs; ++k)
      for (uint64_t i = 0; i < n; ++i) {
        s = s * 6364136223846793005ULL + 1442695040888963407ULL;
        a[k * n + i] = s % qs[k];
        s = s * 6364136223846793005ULL + 1442695040888963407ULL;
        b[k * n + i] = s % qs[k];
      }

    // CPU reference forward, same data, host wall clock.
    double ref_ms = 0;
    {
      std::vector<std::vector<uint64_t>> ra(limbs);
      for (uint64_t k = 0; k < limbs; ++k)
        ra[k].assign(a.begin() + k * n, a.begin() + (k + 1) * n);
      for (uint32_t r = 0; r < kWarmup; ++r)
        for (uint64_t k = 0; k < limbs; ++k) t.ref(k).forward(ra[k]);
      double total = 0;
      for (uint32_t r = 0; r < kReps; ++r) {
        const auto t0 = std::chrono::steady_clock::now();
        for (uint64_t k = 0; k < limbs; ++k) t.ref(k).forward(ra[k]);
        total += clock_ms(t0, std::chrono::steady_clock::now());
      }
      ref_ms = total / kReps;
    }

    // GPU columns: wall clock around each public call — the final device
    // copy blocks, so the measurement is honest end-to-end. Buffers are
    // re-filled outside the timed region; content does not affect timing.
    double fwd_ms = 0, inv_ms = 0, mul_ms = 0;
    {
      std::vector<uint64_t> x(a);
      for (uint32_t r = 0; r < kWarmup; ++r) t.forward(x);
      double total = 0;
      for (uint32_t r = 0; r < kReps; ++r) {
        x = a;
        const auto t0 = std::chrono::steady_clock::now();
        t.forward(x);
        total += clock_ms(t0, std::chrono::steady_clock::now());
      }
      fwd_ms = total / kReps;  // x is now bit-reversed values: valid input
      total = 0;               // for the inverse (any values are)
      for (uint32_t r = 0; r < kReps; ++r) {
        const auto t0 = std::chrono::steady_clock::now();
        t.inverse(x);
        total += clock_ms(t0, std::chrono::steady_clock::now());
      }
      inv_ms = total / kReps;
      for (uint32_t r = 0; r < kWarmup; ++r) t.multiply(a, b);
      total = 0;
      for (uint32_t r = 0; r < kReps; ++r) {
        const auto t0 = std::chrono::steady_clock::now();
        t.multiply(a, b);
        total += clock_ms(t0, std::chrono::steady_clock::now());
      }
      mul_ms = total / kReps;
    }

    out << "| 2^" << e << " (" << n << ") | " << limbs << " | " << ref_ms
        << " | " << fwd_ms << " | " << inv_ms << " | " << mul_ms << " |\n";
    out.flush();
  }
}
