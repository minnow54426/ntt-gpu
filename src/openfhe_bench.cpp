// Sequential per-tower OpenFHE NativeVector baseline. Uses OpenFHE's
// preconditioned CRT-FTT implementation, not our scalar NTT. Not a DCRT/OpenMP,
// HEXL, full CKKS or fastest-possible CPU benchmark.
#include "lattice/lat-hal.h"
#include "math/math-hal.h"
#include "ntt/benchmark_inputs.hpp"
#include "ntt/measurement.hpp"
#include "ntt/ntt.hpp"
#include "ntt/vector_io.hpp"
#include <iostream>

int main(int argc, char **argv) {
  try {
    if (argc != 2 && argc != 4)
      throw std::invalid_argument("usage: ntt_openfhe_bench <parameter-vector> [warmups samples]");
    const auto parameters = ntt::read_vector_file(argv[1]);
    const size_t warmups = argc == 4 ? std::stoul(argv[2]) : 5;
    const size_t samples = argc == 4 ? std::stoul(argv[3]) : 100;
    ntt::measurement::validate_counts(warmups, samples);
    const auto [left, right] = ntt::measurement::benchmark_inputs(
        parameters.n, parameters.q, parameters.psi);
    const size_t dimension = parameters.n, towers = parameters.q.size();
    std::vector<lbcrypto::NativeVector> coefficients, evaluations, right_evaluations, working, other;
    std::vector<std::vector<uint64_t>> expected_forward(towers), expected_product(towers);
    for (size_t tower = 0; tower < towers; ++tower) {
      const lbcrypto::NativeInteger modulus(parameters.q[tower]), root(parameters.psi[tower]);
      ChineseRemainderTransformFTT<lbcrypto::NativeVector>().PreCompute(root, 2 * dimension, modulus);
      lbcrypto::NativeVector input(dimension, modulus), right_input(dimension, modulus);
      std::vector<uint64_t> reference_left(dimension), reference_right(dimension);
      for (size_t index = 0; index < dimension; ++index) {
        input[index] = reference_left[index] = left[tower * dimension + index];
        right_input[index] = reference_right[index] = right[tower * dimension + index];
      }
      ntt::NegacyclicNtt reference(dimension, parameters.q[tower], parameters.psi[tower]);
      reference.forward(reference_left);
      reference.forward(reference_right);
      expected_forward[tower] = reference_left;
      for (size_t index = 0; index < dimension; ++index)
        reference_left[index] = ntt::mul_mod(reference_left[index], reference_right[index], parameters.q[tower]);
      reference.inverse(reference_left);
      expected_product[tower] = std::move(reference_left);
      coefficients.push_back(input);
      ChineseRemainderTransformFTT<lbcrypto::NativeVector>().ForwardTransformToBitReverseInPlace(root, 2 * dimension, &input);
      ChineseRemainderTransformFTT<lbcrypto::NativeVector>().ForwardTransformToBitReverseInPlace(root, 2 * dimension, &right_input);
      evaluations.push_back(std::move(input));
      right_evaluations.push_back(std::move(right_input));
    }
    auto transform = [&](std::vector<lbcrypto::NativeVector> &values, bool inverse) {
      for (size_t tower = 0; tower < towers; ++tower) {
        const lbcrypto::NativeInteger root(parameters.psi[tower]);
        auto crt = ChineseRemainderTransformFTT<lbcrypto::NativeVector>();
        if (inverse) crt.InverseTransformFromBitReverseInPlace(root, 2 * dimension, &values[tower]);
        else crt.ForwardTransformToBitReverseInPlace(root, 2 * dimension, &values[tower]);
      }
    };
    auto pointwise = [&] {
      for (size_t tower = 0; tower < towers; ++tower)
        working[tower].ModMulEq(other[tower]);
    };
    auto verify = [&](auto expected) {
      for (size_t tower = 0; tower < towers; ++tower)
        for (size_t index = 0; index < dimension; ++index)
          if (working[tower][index].ConvertToInt() != expected(tower, index))
            throw std::runtime_error("OpenFHE benchmark exact-result mismatch");
    };
    std::cout << "{\"type\":\"workload\",\"schema\":1,\"n\":" << dimension
              << ",\"towers\":" << towers << ",\"warmups\":" << warmups << ",\"samples\":" << samples
              << ",\"timing\":\"host_wall_openfhe_native\",\"input_policy\":\"fixed_seed_reset_outside_timer\","
                 "\"seed\":7492057,\"setup_included\":false,\"openfhe_version\":\"1.6.1\","
                 "\"tower_execution\":\"sequential\",\"openfhe_openmp\":";
    ntt::measurement::json_string(std::cout, NTT_OPENFHE_OPENMP);
    std::cout << ",\"openfhe_nativeopt\":";
    ntt::measurement::json_string(std::cout, NTT_OPENFHE_NATIVEOPT);
    std::cout << ",\"openfhe_cxx_flags\":";
    ntt::measurement::json_string(std::cout, NTT_OPENFHE_CXX_FLAGS);
    std::cout << ",\"benchmark_compiler\":";
    ntt::measurement::json_string(std::cout, __VERSION__);
    std::cout << ",\"moduli\":[";
    for (size_t tower = 0; tower < towers; ++tower) {
      if (tower) std::cout << ',';
      std::cout << parameters.q[tower];
    }
    std::cout << "],\"roots\":[";
    for (size_t tower = 0; tower < towers; ++tower) {
      if (tower) std::cout << ',';
      std::cout << parameters.psi[tower];
    }
    std::cout << "]}\n";
    auto measure = [&](const std::string &name, auto prepare, auto operation, auto expected) {
      const auto timings = ntt::measurement::collect(warmups, samples, prepare, operation);
      verify(expected);
      ntt::measurement::write_samples(std::cout, name, timings);
    };
    measure("cpu_openfhe_forward", [&] { working = coefficients; },
        [&] { transform(working, false); },
        [&](size_t tower, size_t index) { return expected_forward[tower][index]; });
    measure("cpu_openfhe_inverse", [&] { working = evaluations; },
        [&] { transform(working, true); },
        [&](size_t tower, size_t index) { return left[tower * dimension + index]; });
    measure("cpu_openfhe_pointwise", [&] { working = evaluations; other = right_evaluations; },
        pointwise, [&](size_t tower, size_t index) {
          return ntt::mul_mod(evaluations[tower][index].ConvertToInt(),
                              right_evaluations[tower][index].ConvertToInt(), parameters.q[tower]);
        });
    measure("cpu_openfhe_multiply", [&] { working = coefficients; other = coefficients;
      for (size_t tower = 0; tower < towers; ++tower)
        for (size_t index = 0; index < dimension; ++index)
          other[tower][index] = right[tower * dimension + index];
    }, [&] { transform(working, false); transform(other, false); pointwise(); transform(working, true); },
        [&](size_t tower, size_t index) { return expected_product[tower][index]; });
    std::cout << "{\"type\":\"complete\",\"operations\":4}\n";
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
