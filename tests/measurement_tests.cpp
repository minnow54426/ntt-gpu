#include "ntt/measurement.hpp"
#include "ntt/benchmark_inputs.hpp"
#include <iostream>
#include <sstream>
namespace {
void require(bool condition) {
  if (!condition) throw std::runtime_error("measurement check failed");
}
struct FakeClock {
  static inline int ticks = 0;
  static inline bool inside = false;
  static auto now() {
    inside = !inside;
    return std::chrono::steady_clock::time_point(std::chrono::milliseconds(ticks++));
  }
};
}
int main(int argc, char **argv) {
  try {
    if (argc == 2 && std::string(argv[1]) == "--emit") {
      std::cout << "{\"type\":\"workload\",\"schema\":1,\"n\":65536,\"towers\":1,"
                   "\"samples\":4,\"warmups\":2,\"moduli\":[562949954338817],\"roots\":[2],"
                   "\"timing\":\"host_wall_public_call\",\"simulation\":true,"
                   "\"input_policy\":\"fixed_seed_reset_outside_timer\",\"setup_included\":false}\n";
      for (const std::string name : {"cpu_reference_forward", "gpu_forward", "gpu_inverse",
                                    "gpu_pointwise", "gpu_unfused_multiply", "gpu_fused_multiply"})
        ntt::measurement::write_samples(std::cout, name, {1, 2, 3, 4});
      std::cout << "{\"type\":\"complete\",\"operations\":6}\n";
      return 0;
    }
    int resets = 0, runs = 0, value = 0;
    const auto samples = ntt::measurement::collect<FakeClock>(2, 4,
        [&] { require(!FakeClock::inside); ++resets; value = 7; },
        [&] { require(value == 7 && FakeClock::inside == (runs >= 2)); ++runs; value = 0; });
    require(resets == 6 && runs == 6 && samples == std::vector<double>(4, 1));
    std::string protocol;
    auto external = ntt::measurement::collect_elapsed(1, 2,
        [&] { protocol += 'P'; }, [&] { protocol += 'T'; return 2.5; },
        [&] { protocol += 'V'; });
    require(protocol == "PTVPTVPTV" && external == std::vector<double>(2, 2.5));
    bool bad_external = false;
    try {
      ntt::measurement::collect_elapsed(1, 2, [] {}, [] { return -1.; }, [] {});
    } catch (const std::runtime_error &) { bad_external = true; }
    require(bad_external);
    const auto inputs = ntt::measurement::benchmark_inputs(2, {17, 41}, {4, 9});
    require(inputs == ntt::measurement::benchmark_inputs(2, {17, 41}, {4, 9}));
    uint64_t state = ntt::measurement::kInputSeed;
    for (size_t index = 0; index < 4; ++index) {
      const uint64_t modulus = index < 2 ? 17 : 41;
      state = state * 6364136223846793005ULL + 1442695040888963407ULL;
      require(inputs.first[index] == state % modulus);
      state = state * 6364136223846793005ULL + 1442695040888963407ULL;
      require(inputs.second[index] == state % modulus);
    }
    require(ntt::measurement::percentile({4, 1, 3, 2}, .5) == 2.5);
    require(std::abs(ntt::measurement::percentile({4, 1, 3, 2}, .05) - 1.15) < 1e-12);
    std::ostringstream output;
    ntt::measurement::write_samples(output, "test", samples);
    require(output.str().find("\"count\":4") != std::string::npos);
    std::ostringstream escaped;
    ntt::measurement::json_string(escaped, "a\n\"\\");
    require(escaped.str() == "\"a\\u000a\\\"\\\\\"");
    bool rejected = false;
    try { ntt::measurement::validate_counts(0, 1); }
    catch (const std::invalid_argument &) { rejected = true; }
    require(rejected);
    std::cout << "MEASUREMENT HOST PASS\n";
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n'; return 1;
  }
}
