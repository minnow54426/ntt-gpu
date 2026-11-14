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

void check(bool condition, const std::string &description) {
  if (!condition) {
    ++failures;
    std::cerr << "FAIL: " << description << "\n";
  }
}

// Deterministic pseudo-random generator, same scheme as unit_tests.cpp so
// any failure replays exactly.
struct Lcg {
  uint64_t state;
  explicit Lcg(uint64_t seed) : state(seed) {}
  uint64_t next(uint64_t modulus) {
    state = state * 6364136223846793005ULL + 1442695040888963407ULL;
    return state % modulus;
  }
};

// Limb-major staging: limb_values[limb_index * transform_length +
// coefficient_index] stores one coefficient reduced modulo that limb's prime.
std::vector<uint64_t> stage_limbs(const ntt::VectorFile &vector_file) {
  std::vector<uint64_t> limb_values(vector_file.q.size() * vector_file.n);
  for (size_t limb_index = 0; limb_index < vector_file.q.size(); ++limb_index)
    for (uint64_t coefficient_index = 0; coefficient_index < vector_file.n;
         ++coefficient_index)
      limb_values[limb_index * vector_file.n + coefficient_index] =
          ntt::reduce_decimal_mod(vector_file.coeffs[coefficient_index],
                                  vector_file.q[limb_index]);
  return limb_values;
}

std::vector<uint64_t> limb_slice(const std::vector<uint64_t> &limb_values,
                                 uint64_t transform_length, uint64_t limb_index) {
  return std::vector<uint64_t>(limb_values.begin() + limb_index * transform_length,
                               limb_values.begin() +
                                   (limb_index + 1) * transform_length);
}

// Forward transform, one "LIMB k" section per RNS prime — the exact output
// sequence of src/cli.cpp, so the oracle parser needs no changes.
int cmd_transform(const ntt::VectorFile &vector_file) {
  ntt::cuda::CudaNttTransform transform(vector_file.n, vector_file.q, vector_file.psi);
  std::vector<uint64_t> limb_values = stage_limbs(vector_file);
  transform.forward(limb_values);
  for (size_t limb_index = 0; limb_index < vector_file.q.size(); ++limb_index) {
    std::cout << "LIMB " << limb_index << "\n";
    for (uint64_t coefficient_index = 0; coefficient_index < vector_file.n;
         ++coefficient_index)
      std::cout << limb_values[limb_index * vector_file.n + coefficient_index] << "\n";
  }
  return 0;
}

// Cheapest end-to-end check, same contract as ntt_cli: forward followed
// by inverse must reproduce the input exactly, limb by limb.
int cmd_identity(const ntt::VectorFile &vector_file) {
  ntt::cuda::CudaNttTransform transform(vector_file.n, vector_file.q, vector_file.psi);
  const std::vector<uint64_t> original_values = stage_limbs(vector_file);
  std::vector<uint64_t> roundtrip_values = original_values;
  transform.forward(roundtrip_values);
  transform.inverse(roundtrip_values);
  for (size_t limb_index = 0; limb_index < vector_file.q.size(); ++limb_index)
    for (uint64_t coefficient_index = 0; coefficient_index < vector_file.n;
         ++coefficient_index) {
      const uint64_t flat_index = limb_index * vector_file.n + coefficient_index;
      if (roundtrip_values[flat_index] != original_values[flat_index]) {
        std::cerr << "IDENTITY FAIL limb " << limb_index << " coeff "
                  << coefficient_index << ": " << roundtrip_values[flat_index]
                  << " != " << original_values[flat_index] << "\n";
        return 1;
      }
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
void verify_case(const std::string &case_label, uint64_t transform_length,
                 const std::vector<uint64_t> &input_values,
                 ntt::cuda::CudaNttTransform &transform) {
  const uint64_t limb_count = transform.limbs();

  std::vector<uint64_t> gpu_values = input_values;
  transform.forward(gpu_values);
  for (uint64_t limb_index = 0; limb_index < limb_count; ++limb_index) {
    std::vector<uint64_t> reference_values =
        limb_slice(input_values, transform_length, limb_index);
    transform.ref(limb_index).forward(reference_values);
    check(limb_slice(gpu_values, transform_length, limb_index) == reference_values,
          case_label + ": gpu forward == ref forward, limb " +
              std::to_string(limb_index));
  }

  transform.inverse(gpu_values);
  check(gpu_values == input_values, case_label + ": gpu roundtrip identity");

  // Deterministic second operand per limb; a full big-int RNS operand is
  // unnecessary here — this case proves schedule equality at scale, while
  // CRT composition across limbs is the n=1024 oracle's job.
  std::vector<uint64_t> right_operand(limb_count * transform_length);
  for (uint64_t limb_index = 0; limb_index < limb_count; ++limb_index) {
    Lcg random_generator(0x9E3779B97F4A7C15ULL ^ transform.q(limb_index));
    for (uint64_t coefficient_index = 0; coefficient_index < transform_length;
         ++coefficient_index)
      right_operand[limb_index * transform_length + coefficient_index] =
          random_generator.next(transform.q(limb_index));
  }
  const std::vector<uint64_t> gpu_product =
      transform.multiply(input_values, right_operand);
  for (uint64_t limb_index = 0; limb_index < limb_count; ++limb_index) {
    check(
        limb_slice(gpu_product, transform_length, limb_index) ==
            ntt::ntt_multiply(transform.ref(limb_index),
                              limb_slice(input_values, transform_length, limb_index),
                              limb_slice(right_operand, transform_length, limb_index)),
        case_label + ": gpu multiply == ref multiply, limb " +
            std::to_string(limb_index));
  }
}

int cmd_verify(int argc, char **argv) {
  for (int argument_index = 2; argument_index < argc; ++argument_index) {
    const ntt::VectorFile vector_file = ntt::read_vector_file(argv[argument_index]);
    ntt::cuda::CudaNttTransform transform(vector_file.n, vector_file.q,
                                          vector_file.psi);
    verify_case(argv[argument_index], vector_file.n, stage_limbs(vector_file),
                transform);
    std::cout << "VERIFIED: " << argv[argument_index] << " (n=" << vector_file.n
              << ", limbs=" << vector_file.q.size() << ")\n";
  }
  // The big case: 2^16 with fresh parameters — beyond the fixtures,
  // far beyond the Python oracle's O(N^2) reach, and exactly the size
  // the roadmap's v0.1 benchmark row reports.
  constexpr uint64_t kLargeTransformLength = 65536;
  constexpr uint64_t kLargeCaseLimbCount = 3;
  const std::vector<uint64_t> moduli =
      ntt::find_primes(kLargeTransformLength, kLargeCaseLimbCount);
  std::vector<uint64_t> primitive_roots;
  for (uint64_t modulus : moduli)
    primitive_roots.push_back(ntt::find_psi(modulus, kLargeTransformLength));
  std::vector<uint64_t> large_case_values(kLargeCaseLimbCount * kLargeTransformLength);
  for (uint64_t limb_index = 0; limb_index < kLargeCaseLimbCount; ++limb_index) {
    Lcg random_generator(0x9E3779B97F4A7C15ULL ^ moduli[limb_index]);
    for (uint64_t coefficient_index = 0; coefficient_index < kLargeTransformLength;
         ++coefficient_index)
      large_case_values[limb_index * kLargeTransformLength + coefficient_index] =
          random_generator.next(moduli[limb_index]);
  }
  ntt::cuda::CudaNttTransform large_transform(kLargeTransformLength, moduli,
                                              primitive_roots);
  verify_case("generated-n65536", kLargeTransformLength, large_case_values,
              large_transform);
  std::cout << "VERIFIED: generated n=" << kLargeTransformLength
            << ", limbs=" << kLargeCaseLimbCount << "\n";

  if (failures != 0)
    return 1;
  std::cout << "VERIFIED OK\n";
  return 0;
}

int cmd_bench(int argc, char **argv) {
  uint32_t minimum_size_exponent = 16;
  uint32_t maximum_size_exponent = 16;
  uint64_t limb_count = 3;
  if (argc >= 4) {
    minimum_size_exponent = static_cast<uint32_t>(std::stoul(argv[2]));
    maximum_size_exponent = static_cast<uint32_t>(std::stoul(argv[3]));
  }
  if (argc >= 5)
    limb_count = std::stoul(argv[4]);
  ntt::cuda::run_bench(minimum_size_exponent, maximum_size_exponent, limb_count,
                       std::cout);
  return 0;
}

} // namespace

int main(int argc, char **argv) {
  try {
    if (argc >= 3 && (std::string(argv[1]) == "bench-jsonl" ||
                      std::string(argv[1]) == "events-jsonl")) {
      if (argc != 3 && argc != 5)
        throw std::invalid_argument("usage: bench-jsonl|events-jsonl <parameter-vector> [warmups samples]");
      const auto parameters = ntt::read_vector_file(argv[2]);
      ntt::cuda::run_bench_samples(parameters.n, parameters.q, parameters.psi,
          argc == 5 ? std::stoul(argv[3]) : 5,
          argc == 5 ? std::stoul(argv[4]) : 100, std::cout,
          std::string(argv[1]) == "events-jsonl");
      return 0;
    }
    if (argc == 2 && std::string(argv[1]) == "info") {
      ntt::cuda::describe_device(std::cout);
      return 0;
    }
    if (argc >= 2 && std::string(argv[1]) == "verify") {
      if (argc < 3) {
        std::cerr << "usage: ntt_cuda_cli verify <vector-file> [...]\n";
        return 2;
      }
      return cmd_verify(argc, argv);
    }
    if (argc >= 2 && std::string(argv[1]) == "bench")
      return cmd_bench(argc, argv);
    if (argc == 3) {
      const ntt::VectorFile vector_file = ntt::read_vector_file(argv[1]);
      const std::string command = argv[2];
      if (command == "transform")
        return cmd_transform(vector_file);
      if (command == "identity")
        return cmd_identity(vector_file);
      std::cerr << "unknown command: " << command << "\n";
      return 2;
    }
    std::cerr << "usage: ntt_cuda_cli <vector-file> <transform|identity>\n"
                 "       ntt_cuda_cli info\n"
                 "       ntt_cuda_cli verify <vector-file> [...]\n"
                 "       ntt_cuda_cli bench [min_exp max_exp limbs]\n"
                 "       ntt_cuda_cli bench-jsonl|events-jsonl <parameter-vector> [warmups samples]\n";
    return 2;
  } catch (const std::exception &error) {
    std::cerr << "error: " << error.what() << "\n";
    return 1;
  }
}
