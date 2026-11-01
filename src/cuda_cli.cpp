// ntt_cuda_cli: the CUDA backend behind the same CLI contract as ntt_cli.
//
//   ntt_cuda_cli <vector-file> <transform|identity>   (byte-identical to
//     ntt_cli — tests/oracle/test_vectors.py runs unchanged against this
//     binary, which is the cuda_oracle_crosscheck gate)
//   ntt_cuda_cli verify <vector-file> [...]           (cuda_verify gate)
//   ntt_cuda_cli bench [min_exp max_exp limbs]        (manual; report
//     numbers only attached to named hardware — README rule)
//
// A plain C++ TU: everything CUDA lives behind ntt::cuda::CudaNttTransform
// (src/cuda/ntt_kernels.cu).

#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

#include "ntt/cuda_ntt.hpp"
#include "ntt/ntt.hpp"
#include "ntt/param_gen.hpp"
#include "ntt/vector_io.hpp"

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

// Limb-major staging of a vector file: buf[k*n + i] = coefficient i
// reduced mod q_k — the flat layout CudaNttTransform consumes.
std::vector<uint64_t> stage_limbs(const ntt::VectorFile& vf) {
  std::vector<uint64_t> buf(vf.q.size() * vf.n);
  for (size_t k = 0; k < vf.q.size(); ++k)
    for (uint64_t i = 0; i < vf.n; ++i)
      buf[k * vf.n + i] = ntt::reduce_decimal_mod(vf.coeffs[i], vf.q[k]);
  return buf;
}

std::vector<uint64_t> limb_slice(const std::vector<uint64_t>& buf,
                                 uint64_t n, uint64_t k) {
  return std::vector<uint64_t>(buf.begin() + k * n, buf.begin() + (k + 1) * n);
}

// Forward transform, one "LIMB k" section per RNS prime — the exact output
// sequence of src/cli.cpp, so the oracle parser needs no changes.
int cmd_transform(const ntt::VectorFile& vf) {
  ntt::cuda::CudaNttTransform t(vf.n, vf.q, vf.psi);
  std::vector<uint64_t> buf = stage_limbs(vf);
  t.forward(buf);
  for (size_t k = 0; k < vf.q.size(); ++k) {
    std::cout << "LIMB " << k << "\n";
    for (uint64_t i = 0; i < vf.n; ++i)
      std::cout << buf[k * vf.n + i] << "\n";
  }
  return 0;
}

// Cheapest end-to-end check, same contract as ntt_cli: forward followed
// by inverse must reproduce the input exactly, limb by limb.
int cmd_identity(const ntt::VectorFile& vf) {
  ntt::cuda::CudaNttTransform t(vf.n, vf.q, vf.psi);
  const std::vector<uint64_t> orig = stage_limbs(vf);
  std::vector<uint64_t> a = orig;
  t.forward(a);
  t.inverse(a);
  for (size_t k = 0; k < vf.q.size(); ++k)
    for (uint64_t i = 0; i < vf.n; ++i)
      if (a[k * vf.n + i] != orig[k * vf.n + i]) {
        std::cerr << "IDENTITY FAIL limb " << k << " coeff " << i << ": "
                  << a[k * vf.n + i] << " != " << orig[k * vf.n + i] << "\n";
        return 1;
      }
  std::cout << "IDENTITY OK\n";
  return 0;
}

// The in-process cross-check, per case (a staged operand set): GPU forward
// must equal the CPU reference forward element for element (bit-reversed
// order included), GPU roundtrip must restore the input, and the GPU
// multiply must equal the reference ntt_multiply on the same operands.
// The reference comes from the transform object itself (t.ref(k)), so the
// two sides can never disagree about their parameters.
void verify_case(const std::string& label, uint64_t n,
                 const std::vector<uint64_t>& buf,
                 ntt::cuda::CudaNttTransform& t) {
  const uint64_t limbs = t.limbs();

  std::vector<uint64_t> gpu = buf;
  t.forward(gpu);
  for (uint64_t k = 0; k < limbs; ++k) {
    std::vector<uint64_t> r = limb_slice(buf, n, k);
    t.ref(k).forward(r);
    check(limb_slice(gpu, n, k) == r,
          label + ": gpu forward == ref forward, limb " + std::to_string(k));
  }

  t.inverse(gpu);
  check(gpu == buf, label + ": gpu roundtrip identity");

  // Deterministic second operand per limb; a full big-int RNS operand is
  // unnecessary here — this case proves schedule equality at scale, while
  // CRT composition across limbs is the n=1024 oracle's job.
  std::vector<uint64_t> b(limbs * n);
  for (uint64_t k = 0; k < limbs; ++k) {
    Lcg rng(0x9E3779B97F4A7C15ULL ^ t.q(k));
    for (uint64_t i = 0; i < n; ++i) b[k * n + i] = rng.next(t.q(k));
  }
  const std::vector<uint64_t> gm = t.multiply(buf, b);
  for (uint64_t k = 0; k < limbs; ++k) {
    check(limb_slice(gm, n, k) ==
              ntt::ntt_multiply(t.ref(k), limb_slice(buf, n, k),
                                limb_slice(b, n, k)),
          label + ": gpu multiply == ref multiply, limb " + std::to_string(k));
  }
}

int cmd_verify(int argc, char** argv) {
  for (int i = 2; i < argc; ++i) {
    const ntt::VectorFile vf = ntt::read_vector_file(argv[i]);
    ntt::cuda::CudaNttTransform t(vf.n, vf.q, vf.psi);
    verify_case(argv[i], vf.n, stage_limbs(vf), t);
    std::cout << "VERIFIED: " << argv[i] << " (n=" << vf.n << ", limbs="
              << vf.q.size() << ")\n";
  }
  // The big case: 2^16 with fresh parameters — beyond the fixtures,
  // far beyond the Python oracle's O(N^2) reach, and exactly the size
  // the roadmap's v0.1 benchmark row reports.
  constexpr uint64_t kBigN = 65536;
  const std::vector<uint64_t> qs = ntt::find_primes(kBigN, 3);
  std::vector<uint64_t> psis;
  for (uint64_t q : qs) psis.push_back(ntt::find_psi(q, kBigN));
  std::vector<uint64_t> buf(3 * kBigN);
  for (uint64_t k = 0; k < 3; ++k) {
    Lcg rng(0x9E3779B97F4A7C15ULL ^ qs[k]);
    for (uint64_t i = 0; i < kBigN; ++i) buf[k * kBigN + i] = rng.next(qs[k]);
  }
  ntt::cuda::CudaNttTransform t(kBigN, qs, psis);
  verify_case("generated-n65536", kBigN, buf, t);
  std::cout << "VERIFIED: generated n=" << kBigN << ", limbs=3\n";

  if (failures != 0) return 1;
  std::cout << "VERIFIED OK\n";
  return 0;
}

int cmd_bench(int argc, char** argv) {
  uint32_t min_exp = 12, max_exp = 18;
  uint64_t limbs = 3;
  if (argc >= 4) {
    min_exp = static_cast<uint32_t>(std::stoul(argv[2]));
    max_exp = static_cast<uint32_t>(std::stoul(argv[3]));
  }
  if (argc >= 5) limbs = std::stoul(argv[4]);
  ntt::cuda::run_bench(min_exp, max_exp, limbs, std::cout);
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc >= 2 && std::string(argv[1]) == "verify") {
      if (argc < 3) {
        std::cerr << "usage: ntt_cuda_cli verify <vector-file> [...]\n";
        return 2;
      }
      return cmd_verify(argc, argv);
    }
    if (argc >= 2 && std::string(argv[1]) == "bench") return cmd_bench(argc, argv);
    if (argc == 3) {
      const ntt::VectorFile vf = ntt::read_vector_file(argv[1]);
      const std::string cmd = argv[2];
      if (cmd == "transform") return cmd_transform(vf);
      if (cmd == "identity") return cmd_identity(vf);
      std::cerr << "unknown command: " << cmd << "\n";
      return 2;
    }
    std::cerr << "usage: ntt_cuda_cli <vector-file> <transform|identity>\n"
                 "       ntt_cuda_cli verify <vector-file> [...]\n"
                 "       ntt_cuda_cli bench [min_exp max_exp limbs]\n";
    return 2;
  } catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << "\n";
    return 1;
  }
}
