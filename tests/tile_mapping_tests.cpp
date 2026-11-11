// CPU execution of the production contiguous-stage helpers. Tagged power
// tables distinguish exponent selection from numerical NTT correctness.
#include <algorithm>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "ntt/dev_core.hpp"

namespace {
constexpr uint64_t kSplit = 256;
constexpr uint32_t kThreads = 256;
constexpr uint64_t kModulus = 1000000007;

void require(bool condition, const std::string &message) {
  if (!condition)
    throw std::runtime_error(message);
}

void check_tile_mapping(uint64_t n, bool inverse) {
  const uint64_t length = std::min(kSplit, n);
  std::vector<uint64_t> tags(n + 1);
  for (uint64_t exponent = 0; exponent <= n; ++exponent)
    tags[exponent] = exponent + 1;
  uint64_t pairs = 0;
  const auto stage = inverse ? ntt::dev_dit_stage : ntt::dev_dif_stage;
  for (uint64_t d = inverse ? 1 : length / 2;
       inverse ? d <= length / 2 : d >= 1;
       d = inverse ? d * 2 : d / 2) {
    std::vector<uint8_t> seen(n, 0);
    std::vector<uint8_t> owners(n / 2, 0);
    // CPU global group/offset order, not a copy of the tile helper's map.
    for (uint64_t group_start = 0; group_start < n; group_start += 2 * d) {
      for (uint64_t offset = 0; offset < d; ++offset) {
        const uint64_t upper = group_start + offset;
        const uint64_t lower = upper + d;
        const uint64_t row = upper / length;
        const uint64_t upper_local = upper % length;
        const uint64_t lower_local = lower % length;
        const uint32_t thread = static_cast<uint32_t>(
            (group_start - row * length) / 2 + offset);
        const uint64_t exponent = (n / d) * offset;
        const std::string context = "N=" + std::to_string(n) + " d=" +
            std::to_string(d) + " row=" + std::to_string(row) +
            " thread=" + std::to_string(thread);
        require(lower / length == row && thread < length / 2,
                "tile closure: " + context);
        require(seen[upper]++ == 0 && seen[lower]++ == 0,
                "disjoint pairs: " + context);
        require(owners[row * (length / 2) + thread]++ == 0,
                "unique owner: " + context);
        for (const bool lower_input : {false, true}) {
          std::vector<uint64_t> actual(length, 0);
          actual[lower_input ? lower_local : upper_local] = 1;
          auto expected = actual;
          const uint64_t x = lower_input ? 0 : 1;
          const uint64_t scaled = lower_input ? tags[inverse ? exponent : d] : 0;
          expected[upper_local] = (x + scaled) % kModulus;
          expected[lower_local] =
              ((x + kModulus - scaled) * tags[inverse ? d : exponent]) % kModulus;
          stage(actual.data(), length, d, n, tags.data(), kModulus, thread, kThreads);
          require(actual == expected, "tagged input/table probe: " + context);
        }
        ++pairs;
      }
    }
    for (const uint8_t count : seen)
      require(count == 1, "every coefficient covered once");
    for (const uint8_t count : owners)
      require(count == 1, "every active owner covered once");
    for (uint64_t row = 0; row < n / length; ++row) {
      for (uint32_t thread = static_cast<uint32_t>(length / 2);
           thread < kThreads; ++thread) {
        std::vector<uint64_t> values(length, 11);
        stage(values.data(), length, d, n, tags.data(), kModulus, thread, kThreads);
        require(values == std::vector<uint64_t>(length, 11), "idle thread wrote values");
      }
    }
  }
  std::cout << (inverse ? "INVERSE" : "FORWARD") << " TILE MAPPING OK: N="
            << n << ", pairs=" << pairs << ", probes=" << 2 * pairs << '\n';
}

void check_stage_coverage(uint64_t n) {
  std::vector<uint64_t> cpu_forward;
  for (uint64_t d = n / 2; d >= 1; d /= 2)
    cpu_forward.push_back(d);
  std::vector<uint64_t> forward;
  std::vector<uint64_t> inverse;
  const uint64_t tile_length = std::min(kSplit, n);
  // Mirror the launcher's two loops; compare with the unsplit CPU loop.
  if (n > kSplit)
    for (uint64_t local = (n / kSplit) / 2; local >= 1; local /= 2)
      forward.push_back(kSplit * local);
  for (uint64_t d = tile_length / 2; d >= 1; d /= 2)
    forward.push_back(d);
  for (uint64_t d = 1; d <= tile_length / 2; d *= 2)
    inverse.push_back(d);
  if (n > kSplit)
    for (uint64_t local = 1; local <= (n / kSplit) / 2; local *= 2)
      inverse.push_back(kSplit * local);
  require(forward == cpu_forward, "forward stages omitted/duplicated/reordered");
  std::reverse(cpu_forward.begin(), cpu_forward.end());
  require(inverse == cpu_forward, "inverse stages omitted/duplicated/reordered");
  // Reuse coverage storage instead of allocating/deallocating at each stage.
  // This also avoids GCC 14's optimized vector-teardown diagnostic here.
  std::vector<uint8_t> seen(n, 0);
  for (const uint64_t d : forward) {
    std::fill(seen.begin(), seen.end(), uint8_t{0});
    const bool far = d >= kSplit;
    const uint64_t length = far ? n / kSplit : tile_length;
    const uint64_t blocks = far ? kSplit : n / tile_length;
    const uint64_t local_distance = far ? d / kSplit : d;
    for (uint64_t block = 0; block < blocks; ++block) {
      for (uint64_t p = 0; p < length / 2; ++p) {
        const uint64_t u = 2 * local_distance * (p / local_distance) + p % local_distance;
        const uint64_t upper = far ? block + kSplit * u : block * tile_length + u;
        const uint64_t lower = upper + d;
        require(lower < n, "index outside polynomial");
        require(upper % (2 * d) < d, "not a CPU upper input");
        require(seen[upper]++ == 0 && seen[lower]++ == 0, "duplicate global endpoint");
      }
    }
    for (const uint8_t count : seen)
      require(count == 1, "missing global endpoint");
  }
  std::cout << "STAGE COVERAGE OK: N=" << n << ", stages=" << forward.size()
            << ", pairs/direction=" << (n / 2) * forward.size() << '\n';
}
} // namespace

int main() {
  try {
    for (const uint64_t n : {65536ULL})
      for (const bool inverse : {false, true})
        check_tile_mapping(n, inverse);
    // Fixed target ring size plus small sizes exercising the min(256,N) clamp
    // and the first size with a column pass.
    for (const uint64_t n : {2ULL, 64ULL, 256ULL, 512ULL, 65536ULL})
      check_stage_coverage(n);
  } catch (const std::exception &error) {
    std::cerr << "FAIL: " << error.what() << '\n';
    return 1;
  }
  return 0;
}
