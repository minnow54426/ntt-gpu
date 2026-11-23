#include "ntt/benchmark_inputs.hpp"
#include "ntt/measurement.hpp"
#include "ntt/ntt.hpp"
#include "ntt/resident.hpp"
#include "ntt/transform_stage.hpp"
#include "ntt/vector_io.hpp"
#include "resident_benchmark_common.hpp"

#include <iostream>

namespace {
using namespace ntt::proposal;
using ntt::resident::Download;
using ntt::resident::Session;
using namespace ntt::resident::benchmark;

Download roundtrip(Session &session, std::span<const uint64_t> left,
                   std::span<const uint64_t> right) {
  auto forward_to_host = [&](std::span<const uint64_t> input) {
    return session.download(session.forward(session.upload(input)));
  };
  const auto left_download = forward_to_host(left);
  const auto left_eval = left_download.wait_values();
  const auto right_download = forward_to_host(right);
  const auto right_eval = right_download.wait_values();
  const auto left_device = session.upload_evaluation(left_eval);
  const auto right_device = session.upload_evaluation(right_eval);
  const auto product_download =
      session.download(session.pointwise(left_device, right_device));
  const auto product = product_download.wait_values();
  return session.download(session.inverse(session.upload_evaluation(product)));
}

} // namespace

int main(int argc, char **argv) try {
  if (argc != 4)
    throw std::invalid_argument("usage: ntt_resident_bench VECTOR WARMUPS SAMPLES");
  const size_t warmups = count_argument(argv[2]), samples = count_argument(argv[3]);
  ntt::measurement::validate_counts(warmups, samples);
  const size_t iterations = warmups + samples;
  if (iterations > 200)
    throw std::invalid_argument(
        "resident benchmark allows at most 200 total iterations");

  ntt::profile::Range setup("ntt.host_setup");
  const auto vector = ntt::read_vector_file(argv[1]);
  const auto [left, right] =
      ntt::measurement::benchmark_inputs(vector.n, vector.q, vector.psi);
  const size_t tower_count = vector.q.size();
  std::vector<Tower> towers;
  std::vector<uint64_t> expected;
  expected.reserve(left.size());
  for (size_t tower = 0; tower < tower_count; ++tower) {
    towers.push_back({vector.q[tower], vector.psi[tower]});
    const size_t first = tower * vector.n, end = first + vector.n;
    const auto product = ntt::ntt_multiply(
        ntt::NegacyclicNtt(vector.n, vector.q[tower], vector.psi[tower]),
        {left.begin() + first, left.begin() + end},
        {right.begin() + first, right.begin() + end});
    expected.insert(expected.end(), product.begin(), product.end());
  }
  const auto parameters = Parameters::create(vector.n, towers, {}, 2);
  const PolynomialLayout layout{parameters, tower_count,
                                q_prefix(*parameters, tower_count), Domain::Coefficient,
                                Order::Natural};
  auto check = [&](std::span<const uint64_t> result) {
    if (!std::ranges::equal(result, expected))
      throw std::runtime_error("resident chain differs from exact CPU coefficients");
  };
  setup.end();

  std::cout
      << "{\"type\":\"workload\",\"schema\":1,\"timing\":\"host_wall_resident_chain\""
      << ",\"n\":" << vector.n << ",\"towers\":" << tower_count
      << ",\"seed\":" << ntt::measurement::kInputSeed << ",\"warmups\":" << warmups
      << ",\"samples\":" << samples
      << ",\"input_policy\":\"fixed_seed_reset_outside_timer\",\"setup_included\":false"
      << ",\"transform_schedule\":\"" << ntt::kTransformSchedule << '"';
#ifdef NTT_RESIDENT_BENCH_HOST
  std::cout << ",\"simulation\":true";
#endif
#ifdef NTT_BENCH_PROFILE_NVTX
  std::cout << ",\"profiled\":true";
#endif
  write_parameters("moduli", vector.q);
  write_parameters("roots", vector.psi);
  std::cout << "}\n";

  {
    auto session = make_session(layout, iterations);
    ntt::profile::Range prepare("ntt.inputs");
    const auto left_device = session.upload(left);
    const auto right_device = session.upload(right);
    right_device.readiness().wait();
    prepare.end();
    ntt::resident::Polynomial result;
    measure(
        "resident_device_chain", warmups, samples,
        [&] {
          const auto left_eval = session.forward(left_device);
          const auto right_eval = session.forward(right_device);
          result = session.inverse(session.pointwise(left_eval, right_eval));
          result.readiness().wait();
        },
        [&] {
          check(session.download(result).wait_values());
          result = {};
        });
    close_session(session);
  }

  const auto measure_host_scope = [&](const char *name, auto operation) {
    auto session = make_session(layout, iterations);
    Download result;
    std::span<const uint64_t> values;
    measure(
        name, warmups, samples,
        [&] {
          result = operation(session, left, right);
          values = result.wait_values();
        },
        [&] {
          check(values);
          values = {};
          result = {};
        });
    close_session(session);
  };
  measure_host_scope(
      "resident_host_boundary",
      [](Session &session, std::span<const uint64_t> left,
         std::span<const uint64_t> right) { return session.multiply(left, right); });
  measure_host_scope("resident_host_roundtrip_chain", roundtrip);
  std::cout << "{\"type\":\"complete\",\"operations\":3}\n";
  return 0;
} catch (const std::exception &error) {
  std::cerr << error.what() << '\n';
  return 1;
}
