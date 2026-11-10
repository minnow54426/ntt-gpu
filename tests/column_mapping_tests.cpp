// Exhaustively compare the real device stage body (compiled on CPU) with
// ntt.hpp's global group/offset schedule. Synthetic table entries e+1 expose
// the exponent actually read; these are index probes, not valid NTT roots.
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "ntt/dev_core.hpp"

namespace {

constexpr uint64_t kSplitWidth = 256;
constexpr uint32_t kThreadCount = 256;
constexpr uint64_t kProbeModulus = 1000000007;

void require(bool condition, const std::string &context) {
  if (!condition)
    throw std::runtime_error(context);
}

void check_transform(uint64_t transform_length) {
  const uint64_t column_length = transform_length / kSplitWidth;
  std::vector<uint64_t> tagged_powers(transform_length + 1);
  for (uint64_t exponent = 0; exponent <= transform_length; ++exponent)
    tagged_powers[exponent] = exponent + 1;

  uint64_t checked_pairs = 0;
  for (uint64_t distance = transform_length / 2; distance >= kSplitWidth;
       distance /= 2) {
    const uint64_t local_distance = distance / kSplitWidth;
    std::vector<uint8_t> seen(transform_length, 0);
    std::vector<uint8_t> seen_threads(kSplitWidth * column_length / 2, 0);

    // Independent CPU reference traversal: global groups, then offsets.
    for (uint64_t group_start = 0; group_start < transform_length;
         group_start += 2 * distance) {
      for (uint64_t cpu_offset = 0; cpu_offset < distance; ++cpu_offset) {
        const uint64_t upper_global = group_start + cpu_offset;
        const uint64_t lower_global = upper_global + distance;
        const uint64_t column = upper_global % kSplitWidth;
        const uint64_t upper_local = upper_global / kSplitWidth;
        const uint64_t lower_local = lower_global / kSplitWidth;
        const uint64_t local_group = group_start / (2 * distance);
        const uint32_t thread = static_cast<uint32_t>(
            local_group * local_distance + cpu_offset / kSplitWidth);
        const uint64_t cpu_exponent = (transform_length / distance) * cpu_offset;
        const std::string context = "N=" + std::to_string(transform_length) +
            " d=" + std::to_string(distance) + " column=" +
            std::to_string(column) + " thread=" + std::to_string(thread);
        require(thread < column_length / 2, "thread range: " + context);
        require(seen[upper_global]++ == 0 && seen[lower_global]++ == 0,
                "disjoint CPU pair: " + context);
        require(seen_threads[column * (column_length / 2) + thread]++ == 0,
                "unique GPU owner: " + context);

        for (const bool probe_ladder : {false, true}) {
          std::vector<uint64_t> actual(column_length, 0);
          actual[probe_ladder ? lower_local : upper_local] = 1;
          std::vector<uint64_t> expected = actual;
          const uint64_t scaled_lower = probe_ladder ? tagged_powers[distance] : 0;
          const uint64_t upper_value = probe_ladder ? 0 : 1;
          expected[upper_local] = (upper_value + scaled_lower) % kProbeModulus;
          const uint64_t difference =
              (upper_value + kProbeModulus - scaled_lower) % kProbeModulus;
          expected[lower_local] =
              (difference * tagged_powers[cpu_exponent]) % kProbeModulus;

          // No copied GPU mapping here: execute the production stage helper.
          ntt::dev_dif_col_stage(actual.data(), column_length, local_distance,
                                transform_length, tagged_powers.data(),
                                kProbeModulus, thread, kThreadCount, column);
          require(actual == expected,
                  std::string(probe_ladder ? "ladder/input probe: " :
                                             "twiddle/input probe: ") + context);
        }
        ++checked_pairs;
      }
    }
    for (const uint8_t count : seen)
      require(count == 1, "complete global coverage");
    for (const uint8_t count : seen_threads)
      require(count == 1, "complete active-thread coverage");

    // Check every idle thread in every column at this stage.
    for (uint64_t column = 0; column < kSplitWidth; ++column) {
      for (uint32_t thread = static_cast<uint32_t>(column_length / 2);
           thread < kThreadCount; ++thread) {
        std::vector<uint64_t> values(column_length, 7);
        ntt::dev_dif_col_stage(values.data(), column_length, local_distance,
                              transform_length, tagged_powers.data(),
                              kProbeModulus, thread, kThreadCount, column);
        require(values == std::vector<uint64_t>(column_length, 7),
                "idle thread modified values");
      }
    }
  }
  std::cout << "COLUMN MAPPING OK: N=" << transform_length
            << ", pairs=" << checked_pairs << ", probes=" << checked_pairs * 2
            << " (CPU execution of real stage body)\n";
}

} // namespace

int main() {
  try {
    check_transform(65536);
  } catch (const std::exception &error) {
    std::cerr << "FAIL: " << error.what() << '\n';
    return 1;
  }
  return 0;
}
