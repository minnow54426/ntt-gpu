// ntt_device_sim: portable simulator of the CUDA kernel schedule.
//
// Runs the SAME stage bodies the GPU kernels run (include/ntt/dev_core.hpp,
// the single source of schedule truth) with std::vector tiles and a serial
// loop over thread ids standing in for thread blocks, and checks the result
// against the CPU reference on identical vectors. Green here means the
// index arithmetic, stage order, two-phase split, and twiddle tables are
// right — proven on this machine, before any GPU is rented.
//
// Catches: index bugs, wrong stage sequence, tile-local-NTT mistakes,
// table-building errors. Cannot catch (it is serial): missing-barrier
// races, nvcc compile errors, launch-config mistakes — the kernels are
// thin wrappers around the shared bodies precisely to keep that surface
// small; the GPU gates cover it on the rented box.
//
// Usage: ntt_device_sim <vector-file> [<vector-file> ...]

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "ntt/dev_core.hpp"
#include "ntt/ntt.hpp"
#include "ntt/vector_io.hpp"

using ntt::dev_dif_stage;
using ntt::dev_dit_stage;
using ntt::inv_mod;
using ntt::mul_mod;

namespace {

int failures = 0;

void check(bool ok, const std::string& what) {
  if (!ok) {
    ++failures;
    std::cerr << "FAIL: " << what << "\n";
  }
}

// Deterministic pseudo-random generator, same scheme as unit_tests.cpp so
// any failure replays exactly.
struct Lcg {
  uint64_t state;
  explicit Lcg(uint64_t seed) : state(seed) {}
  uint64_t next(uint64_t q) {
    state = state * 6364136223846793005ULL + 1442695040888963407ULL;
    return state % q;
  }
};

// The launch schedule of CudaNttTransform (src/cuda/ntt_kernels.cu),
// mirrored line for line: global-phase "launches" over the whole limb,
// then a shared phase per B-wide tile. Threads are emulated by looping
// tid = 0..T-1 through the shared stage body; because one stage's pairs
// are disjoint, serial execution is exactly equivalent to any parallel
// interleaving the GPU might choose.
class ScheduleSim {
 public:
  ScheduleSim(uint64_t n, const std::vector<uint64_t>& q,
              const std::vector<uint64_t>& psi)
      : n_(n), q_(q), T_(kThreads), B_(std::min<uint64_t>(kTile, n)) {
    for (size_t k = 0; k < q_.size(); ++k) {
      // Parameter validation identical to the CUDA path: constructing the
      // reference throws on any bad wheel/root before tables are built.
      refs_.emplace_back(n_, q_[k], psi[k]);
      // pp[e] = psi^e and pp_inv[e] = psi^-e for e in [0, n]: every
      // ladder constant (pp[d]) and twiddle (pp[(n/d)*j]) is a table
      // lookup, so no on-device twiddle generation can be wrong.
      std::vector<uint64_t> pp(n_ + 1), pi(n_ + 1);
      const uint64_t psi_inv = inv_mod(psi[k], q_[k]);
      uint64_t w = 1, wi = 1;
      for (uint64_t e = 0; e <= n_; ++e) {
        pp[e] = w;
        pi[e] = wi;
        w = mul_mod(w, psi[k], q_[k]);
        wi = mul_mod(wi, psi_inv, q_[k]);
      }
      pp_.push_back(std::move(pp));
      pp_inv_.push_back(std::move(pi));
      n_inv_.push_back(inv_mod(n_, q_[k]));  // the inverse's final scale
    }
  }

  const ntt::NegacyclicNtt& ref(size_t k) const { return refs_[k]; }

  // Natural coefficients in, values out in bit-reversed order — same
  // contract as NegacyclicNtt::forward, which the gates assert exactly.
  void forward(std::vector<uint64_t>& a, size_t k) const {
    check_size(a);
    // Global phase: d = n/2 down to B (vacuous when n <= B — the fixture
    // size n = 1024 IS this edge). On GPU: one dif_global launch per stage,
    // grid-stride over the n/2 pairs; here: T threads cover the same pairs.
    for (uint64_t d = n_ / 2; d >= B_; d >>= 1)
      launch(dev_dif_stage, a.data(), n_, d, n_, pp_[k].data(), q_[k]);
    // Shared phase: d = B/2 down to 1, one B-wide tile at a time. On GPU:
    // tile staged in dynamic shared memory with __syncthreads() between
    // stages; here: the tile is a vector and stages run serially.
    for (uint64_t base = 0; base < n_; base += B_) {
      std::vector<uint64_t> smem(a.begin() + base, a.begin() + base + B_);
      for (uint64_t d = B_ / 2; d >= 1; d >>= 1)
        launch(dev_dif_stage, smem.data(), B_, d, n_, pp_[k].data(), q_[k]);
      std::copy(smem.begin(), smem.end(), a.begin() + base);
    }
  }

  // Bit-reversed values in, natural coefficients out: the shared phase
  // first (d = 1 up to B/2), then the global phase (d = B up to n/2), then
  // the elementwise n^-1 scale — the mirror of forward's stage order.
  void inverse(std::vector<uint64_t>& a, size_t k) const {
    check_size(a);
    for (uint64_t base = 0; base < n_; base += B_) {
      std::vector<uint64_t> smem(a.begin() + base, a.begin() + base + B_);
      for (uint64_t d = 1; d <= B_ / 2; d <<= 1)
        launch(dev_dit_stage, smem.data(), B_, d, n_, pp_inv_[k].data(),
               q_[k]);
      std::copy(smem.begin(), smem.end(), a.begin() + base);
    }
    for (uint64_t d = B_; d <= n_ / 2; d <<= 1)
      launch(dev_dit_stage, a.data(), n_, d, n_, pp_inv_[k].data(), q_[k]);
    for (uint64_t i = 0; i < n_; ++i)
      a[i] = mul_mod(a[i], n_inv_[k], q_[k]);
  }

  // The full ring product on this schedule: forward both, N pointwise
  // multiplies (the entire payoff), inverse — checked against
  // ntt_multiply, which unit_tests.cpp already pins to schoolbook.
  std::vector<uint64_t> multiply(std::vector<uint64_t> a,
                                 std::vector<uint64_t> b, size_t k) const {
    check_size(a);
    check_size(b);
    forward(a, k);
    forward(b, k);
    for (uint64_t i = 0; i < n_; ++i) a[i] = mul_mod(a[i], b[i], q_[k]);
    inverse(a, k);
    return a;
  }

 private:
  using StageFn = void (*)(uint64_t*, uint64_t, uint64_t, uint64_t,
                           const uint64_t*, uint64_t, uint32_t, uint32_t);

  // One "kernel launch": every thread of the block runs its slice. The
  // global phase on GPU uses a grid-stride pair index instead of
  // threadIdx — same coverage of a disjoint-pair map, same result.
  void launch(StageFn fn, uint64_t* tile, uint64_t tile_len, uint64_t d,
              uint64_t n, const uint64_t* pp, uint64_t q) const {
    for (uint32_t tid = 0; tid < T_; ++tid)
      fn(tile, tile_len, d, n, pp, q, tid, T_);
  }

  void check_size(const std::vector<uint64_t>& a) const {
    if (a.size() != n_)
      throw std::length_error("vector size " + std::to_string(a.size()) +
                              " != n " + std::to_string(n_));
  }

  // Must match the kernel constants in src/cuda/ntt_kernels.cu.
  static constexpr uint32_t kThreads = 256;  // T: threads per block
  static constexpr uint64_t kTile = 1024;    // B: shared-phase tile width

  uint64_t n_;
  std::vector<uint64_t> q_;
  uint32_t T_;
  uint64_t B_;
  std::vector<ntt::NegacyclicNtt> refs_;  // per-limb CPU reference
  std::vector<std::vector<uint64_t>> pp_;      // psi^e tables per limb
  std::vector<std::vector<uint64_t>> pp_inv_;  // psi^-e tables per limb
  std::vector<uint64_t> n_inv_;                // n^-1 per limb
};

// Per RNS limb: schedule-forward == reference-forward element for element
// (bit-reversed order included — this is the equality the CUDA gates later
// assert on the GPU), schedule-roundtrip == input, and the full schedule
// multiply == the reference ntt_multiply.
void test_with_vectors(const ntt::VectorFile& vf) {
  const ScheduleSim sim(vf.n, vf.q, vf.psi);
  for (size_t k = 0; k < vf.q.size(); ++k) {
    std::vector<uint64_t> orig(vf.n), a(vf.n), ref(vf.n);
    for (uint64_t i = 0; i < vf.n; ++i)
      orig[i] = a[i] = ref[i] = ntt::reduce_decimal_mod(vf.coeffs[i], vf.q[k]);

    sim.forward(a, k);
    sim.ref(k).forward(ref);
    check(a == ref, "schedule forward == reference forward, limb " +
                        std::to_string(k));

    sim.inverse(a, k);
    check(a == orig, "schedule roundtrip identity, limb " + std::to_string(k));

    Lcg rng(0x9E3779B97F4A7C15ULL ^ vf.q[k]);
    std::vector<uint64_t> b(vf.n);
    for (uint64_t i = 0; i < vf.n; ++i) b[i] = rng.next(vf.q[k]);
    check(sim.multiply(orig, b, k) == ntt::ntt_multiply(sim.ref(k), orig, b),
          "schedule multiply == reference multiply, limb " +
              std::to_string(k));
  }
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::cerr << "usage: ntt_device_sim <vector-file> [<vector-file> ...]\n";
    return 2;
  }
  for (int i = 1; i < argc; ++i) {
    try {
      const ntt::VectorFile vf = ntt::read_vector_file(argv[i]);
      test_with_vectors(vf);
      std::cout << "DEVICE SCHEDULE OK: n=" << vf.n << ", limbs="
                << vf.q.size() << " (" << argv[i] << ")\n";
    } catch (const std::exception& e) {
      std::cerr << argv[i] << ": " << e.what() << "\n";
      return 1;
    }
  }
  return failures == 0 ? 0 : 1;
}
