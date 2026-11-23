#pragma once
// Shared timing protocol for the NTT and conversion resident experiments.
#include "ntt/measurement.hpp"
#include "ntt/benchmark_profile.hpp"
#include "ntt/resident.hpp"
#ifdef NTT_RESIDENT_BENCH_HOST
#include "support/resident_host_driver.hpp"
#endif
#include <iostream>

namespace ntt::resident::benchmark {
using namespace ntt::proposal;
using ntt::resident::Session;
using Clock = std::chrono::steady_clock;
inline Session make_session(const PolynomialLayout &layout, size_t iterations) {
  ntt::profile::Range setup("ntt.session_setup");
  ntt::resident::Limits limits;
  // The prototype retains operation history. Allow at most 13 submissions
  // per iteration (the explicit host-roundtrip chain) in each fresh session.
  limits.operations = 2 + 13 * iterations;
#ifdef NTT_RESIDENT_BENCH_HOST
  return Session(std::make_shared<resident_test::HostDriver>(layout), limits);
#else
  return ntt::resident::make_cuda_session(layout, 0, limits);
#endif
}

inline void close_session(Session &session) {
  ntt::profile::Range close("ntt.session_close");
  session.close();
}

inline void write_parameters(const char *name, const std::vector<uint64_t> &values) {
  std::cout << ",\"" << name << "\":[";
  for (size_t index = 0; index < values.size(); ++index)
    std::cout << (index ? "," : "") << values[index];
  std::cout << ']';
}

inline size_t count_argument(const char *argument) {
  const std::string value(argument);
  if (value.empty() || value.find_first_not_of("0123456789") != std::string::npos)
    throw std::invalid_argument("iteration counts must be decimal integers");
  const auto result = std::stoull(value);
  if (result > 200)
    throw std::invalid_argument(
        "resident benchmark allows at most 200 total iterations");
  return static_cast<size_t>(result);
}

template <class Operation, class Verify>
void measure(const char *name, size_t warmups, size_t samples, Operation operation,
             Verify verify) {
  size_t iteration = 0;
  auto elapsed = ntt::measurement::collect_elapsed(
      warmups, samples, [] {},
      [&] {
        ntt::profile::Range phase(iteration < warmups ? "ntt.warmup" : "ntt.sample");
        ntt::profile::Range scope(name);
        const auto start = Clock::now();
        operation();
        return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
      },
      [&] {
        ntt::profile::Range verification("ntt.verify");
        verify();
        ++iteration;
      });
  ntt::profile::Range reporting("ntt.report");
  ntt::measurement::write_samples(std::cout, name, elapsed);
}
} // namespace ntt::resident::benchmark
