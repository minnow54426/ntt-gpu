#include "ntt/benchmark_inputs.hpp"
#include "ntt/transform_stage.hpp"
#include "ntt/vector_io.hpp"
#include "resident_benchmark_common.hpp"
// Reuse the native correctness fixture's pinned OpenFHE table/packing helpers.
#include "support/resident_conversion_cases.hpp"

namespace {
using namespace ntt::proposal;
using namespace ntt::resident::benchmark;
using ntt::resident::Polynomial;
using ntt::resident::Session;
namespace oracle = resident_conversion_cases;

lbcrypto::DCRTPoly polynomial(const oracle::Params &parameters, Format format,
                              std::span<const uint64_t> input) {
  const size_t dimension = parameters->GetRingDimension();
  lbcrypto::DCRTPoly result(parameters, format, false);
  for (size_t tower = 0; tower < parameters->GetParams().size(); ++tower) {
    const auto &parameter = parameters->GetParams()[tower];
    lbcrypto::NativeVector values(dimension, parameter->GetModulus());
    for (size_t index = 0; index < dimension; ++index)
      values[index] = lbcrypto::NativeInteger(input[tower * dimension + index]);
    lbcrypto::NativePoly limb(parameter, format);
    limb.SetValues(std::move(values), format);
    result.SetElementAtIndex(tower, std::move(limb));
  }
  return result;
}

std::vector<Tower> special_primes(const ntt::VectorFile &vector) {
  std::vector<Tower> result;
  for (auto modulus : ntt::find_primes(vector.n, vector.q.size() + 3, 60))
    if (std::find(vector.q.begin(), vector.q.end(), modulus) == vector.q.end()) {
      result.push_back({modulus, ntt::find_psi(modulus, vector.n)});
      if (result.size() == 3)
        break;
    }
  require(result.size() == 3, "three disjoint special primes required");
  return result;
}
} // namespace

int main(int argc, char **argv) try {
  if (argc != 4)
    throw std::invalid_argument(
        "usage: ntt_resident_conversion_bench VECTOR WARMUPS SAMPLES");
  const size_t warmups = count_argument(argv[2]), samples = count_argument(argv[3]);
  ntt::measurement::validate_counts(warmups, samples);
  const size_t iterations = warmups + samples;
  if (iterations > 200)
    throw std::invalid_argument(
        "resident benchmark allows at most 200 total iterations");
  const auto vector = ntt::read_vector_file(argv[1]);
  require(vector.q.size() <= 40, "conversion benchmark supports at most 40 Q towers");
  for (auto modulus : vector.q)
    require(modulus < (uint64_t{1} << 60),
            "conversion benchmark requires Q primes of at most 60 bits");
  const auto [coefficients, unused] =
      ntt::measurement::benchmark_inputs(vector.n, vector.q, vector.psi);
  std::vector<Tower> q;
  for (size_t index = 0; index < vector.q.size(); ++index)
    q.push_back({vector.q[index], vector.psi[index]});
  const auto p = special_primes(vector);
  auto qp = q;
  qp.insert(qp.end(), p.begin(), p.end());
  const auto parameters = Parameters::create(vector.n, q, p, 3);
  const PolynomialLayout full{parameters, q.size(), qp_basis(*parameters, q.size()),
                              Domain::Coefficient, Order::Natural};
  const PolynomialLayout source{parameters, q.size(), q_prefix(*parameters, q.size()),
                                Domain::Coefficient, Order::Natural};
  const auto source_eval = transform_layout(source, false),
             full_eval = transform_layout(full, false);
  const auto q_params = oracle::params(vector.n, q),
             p_params = oracle::params(vector.n, p),
             qp_params = oracle::params(vector.n, qp);
  const oracle::Tables up_tables(q_params, p_params), down_tables(p_params, q_params);
  lbcrypto::BigInteger product_p(1);
  std::vector<uint64_t> p_moduli, p_roots, full_moduli = vector.q,
                                           full_roots = vector.psi;
  for (auto tower : p) {
    product_p *= lbcrypto::BigInteger(tower.q);
    p_moduli.push_back(tower.q);
    p_roots.push_back(tower.psi);
    full_moduli.push_back(tower.q);
    full_roots.push_back(tower.psi);
  }
  std::vector<lbcrypto::NativeInteger> inverses, preconditions;
  for (auto modulus : vector.q) {
    const lbcrypto::NativeInteger prime(modulus);
    inverses.emplace_back(product_p.Mod(lbcrypto::BigInteger(modulus)).ConvertToInt());
    inverses.back() = inverses.back().ModInverse(prime);
    preconditions.push_back(inverses.back().PrepModMulConst(prime));
  }
  auto down = [&](const lbcrypto::DCRTPoly &input) {
    return input.ApproxModDown(q_params, p_params, inverses, preconditions,
                               down_tables.inverses, down_tables.preconditions,
                               down_tables.hats, down_tables.barrett, {}, {},
                               lbcrypto::NativeInteger(0), {});
  };
  auto original = polynomial(q_params, Format::COEFFICIENT, coefficients);
  auto evaluation = original;
  evaluation.SetFormat(Format::EVALUATION);
  const auto evaluation_values = oracle::flatten(evaluation);
  auto lifted = original;
  lifted.ApproxModUp(q_params, p_params, qp_params, up_tables.inverses,
                     up_tables.preconditions, up_tables.hats, up_tables.barrett);
  const auto expected_up = oracle::flatten(lifted);
  const auto expected_chain = oracle::flatten(down(lifted));
  const auto arbitrary =
      ntt::measurement::benchmark_inputs(vector.n, full_moduli, full_roots).first;
  const auto expected_down =
      oracle::flatten(down(polynomial(qp_params, Format::EVALUATION, arbitrary)));
  const auto check = [](std::span<const uint64_t> actual,
                        const std::vector<uint64_t> &expected) {
    require(std::ranges::equal(actual, expected),
            "conversion benchmark differs from pinned OpenFHE");
  };

  std::cout
      << "{\"type\":\"workload\",\"schema\":1,\"timing\":\"host_wall_resident_"
         "conversion\""
      << ",\"n\":" << vector.n << ",\"towers\":" << q.size()
      << ",\"parameter_setup_included\":true"
      << ",\"seed\":" << ntt::measurement::kInputSeed << ",\"warmups\":" << warmups
      << ",\"samples\":" << samples
      << ",\"input_policy\":\"fixed_seed_reset_outside_timer\",\"setup_included\":false"
      << ",\"transform_schedule\":\"" << ntt::kTransformSchedule << '"'
      << ",\"p_policy\":\"first_disjoint_60_bit_primes\",\"down_input_policy\":"
         "\"seeded_arbitrary_qp_evaluations\"";
#ifdef NTT_RESIDENT_BENCH_HOST
  std::cout << ",\"simulation\":true";
#endif
  write_parameters("moduli", vector.q);
  write_parameters("roots", vector.psi);
  write_parameters("p_moduli", p_moduli);
  write_parameters("p_roots", p_roots);
  std::cout << "}\n";

  auto device_case = [&](const char *name, const PolynomialLayout &input_layout,
                         const std::vector<uint64_t> &input, const auto &expected,
                         size_t parameter_uploads, auto operation) {
    auto session = make_session(full, iterations);
    const auto resident_input = session.upload(input_layout, input);
    resident_input.readiness().wait();
    Polynomial result;
    measure(
        name, warmups, samples,
        [&] {
          result = operation(session, resident_input);
          result.readiness().wait();
        },
        [&] {
          check(session.download(result).wait_values(), expected);
          result = {};
        });
    const auto counts = session.stats();
    require(counts.uploads == 1 && counts.downloads == iterations &&
                counts.parameter_uploads == parameter_uploads * iterations,
            "resident conversion changed its transfer boundary");
    session.close();
  };
  auto up = [](Session &session, const Polynomial &input) {
    return session.mod_up(input);
  };
  auto reduce = [](Session &session, const Polynomial &input) {
    return session.mod_down(input);
  };
  auto compose = [](Session &session, const Polynomial &input) {
    return session.mod_down(session.mod_up(input));
  };
  device_case("resident_modup_coefficient", source, coefficients, expected_up, 1, up);
  device_case("resident_modup_evaluation", source_eval, evaluation_values, expected_up,
              1, up);
  device_case("resident_moddown_evaluation", full_eval, arbitrary, expected_down, 2,
              reduce);
  device_case("resident_modup_moddown", source, coefficients, expected_chain, 3,
              compose);
  for (bool roundtrip : {false, true}) {
    auto session = make_session(full, iterations);
    ntt::resident::Download result;
    std::span<const uint64_t> values;
    measure(
        roundtrip ? "resident_conversion_host_roundtrip"
                  : "resident_conversion_host_boundary",
        warmups, samples,
        [&] {
          const auto input = session.upload(source, coefficients);
          const auto lifted_device = session.mod_up(input);
          if (roundtrip) {
            const auto downloaded = session.download(lifted_device);
            const auto intermediate = downloaded.wait_values();
            result = session.download(
                session.mod_down(session.upload(full_eval, intermediate)));
          } else {
            result = session.download(session.mod_down(lifted_device));
          }
          values = result.wait_values();
        },
        [&] {
          check(values, expected_chain);
          values = {};
          result = {};
        });
    const auto counts = session.stats();
    const size_t transfers = roundtrip ? 2 : 1;
    require(counts.uploads == transfers * iterations &&
                counts.downloads == transfers * iterations &&
                counts.parameter_uploads == 3 * iterations,
            "host conversion control changed its transfer boundary");
    session.close();
  }
  std::cout << "{\"type\":\"complete\",\"operations\":6}\n";
  return 0;
} catch (const std::exception &error) {
  std::cerr << error.what() << '\n';
  return 1;
}
