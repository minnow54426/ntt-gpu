// Exhaustively compare the real device stage body (compiled on CPU) with
// ntt.hpp's global group/offset schedule. Synthetic table entries e+1 expose
// the exponent actually read; these are index probes, not valid NTT roots.
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "ntt/dev_core.hpp"
#include "ntt/ntt.hpp"
#include "ntt/param_gen.hpp"

namespace {

constexpr uint64_t kSplitWidth = 256;
constexpr uint32_t kThreadCount = 256;
constexpr uint64_t kProbeModulus = 1000000007;

void require(bool condition, const std::string &context) {
  if (!condition)
    throw std::runtime_error(context);
}

void check_transform(uint64_t transform_length, bool inverse) {
  const uint64_t column_length = transform_length / kSplitWidth;
  std::vector<uint64_t> tagged_powers(transform_length + 1);
  for (uint64_t exponent = 0; exponent <= transform_length; ++exponent)
    tagged_powers[exponent] = exponent + 1;

  uint64_t checked_pairs = 0;
  for (uint64_t distance = inverse ? kSplitWidth : transform_length / 2;
       inverse ? distance <= transform_length / 2 : distance >= kSplitWidth;
       distance = inverse ? distance * 2 : distance / 2) {
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

          if (inverse) {
            // CPU DIT: first untwiddle y, then sum/difference, then c^-1.
            const uint64_t x = probe_ladder ? 0 : 1;
            const uint64_t untwiddled_y = probe_ladder ? tagged_powers[cpu_exponent] : 0;
            expected[upper_local] = (x + untwiddled_y) % kProbeModulus;
            expected[lower_local] =
                ((x + kProbeModulus - untwiddled_y) * tagged_powers[distance]) %
                kProbeModulus;
          }

          // No copied GPU mapping here: execute the production stage helper.
          const auto stage = inverse ? ntt::dev_dit_col_stage : ntt::dev_dif_col_stage;
          stage(actual.data(), column_length, local_distance, transform_length,
                tagged_powers.data(), kProbeModulus, thread, kThreadCount, column);
          require(actual == expected,
                  std::string(probe_ladder ? "lower-input probe: " :
                                             "upper-input probe: ") + context);
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
        const auto stage = inverse ? ntt::dev_dit_col_stage : ntt::dev_dif_col_stage;
        stage(values.data(), column_length, local_distance, transform_length,
              tagged_powers.data(), kProbeModulus, thread, kThreadCount, column);
        require(values == std::vector<uint64_t>(column_length, 7),
                "idle thread modified values");
      }
    }
  }
  std::cout << (inverse ? "INVERSE" : "FORWARD") << " COLUMN MAPPING OK: N=" << transform_length
            << ", pairs=" << checked_pairs << ", probes=" << checked_pairs * 2
            << " (CPU execution of real stage body)\n";
}

void check_inverse_correspondence(uint64_t transform_length) {
  const uint64_t column_length = transform_length / kSplitWidth;
  for (const uint64_t modulus : ntt::find_primes(transform_length, 2)) {
    const uint64_t root = ntt::find_psi(modulus, transform_length);
    const uint64_t inverse_root = ntt::inv_mod(root, modulus);
    std::vector<uint64_t> forward_powers(transform_length + 1, 1);
    std::vector<uint64_t> inverse_powers(transform_length + 1, 1);
    for (uint64_t exponent = 1; exponent <= transform_length; ++exponent) {
      forward_powers[exponent] = ntt::mul_mod(forward_powers[exponent - 1], root, modulus);
      inverse_powers[exponent] = ntt::mul_mod(inverse_powers[exponent - 1], inverse_root, modulus);
    }
    // Every column and every local stage: DIT(DIF(values)) == 2*values.
    // Boundary residues plus varied values exercise reduction and negatives.
    for (uint64_t column = 0; column < kSplitWidth; ++column) {
      std::vector<uint64_t> input(column_length);
      for (uint64_t index = 0; index < column_length; ++index) {
        const uint64_t global_index = column + kSplitWidth * index;
        input[index] = global_index % 4 == 0 ? 0 :
                       global_index % 4 == 1 ? modulus - 1 :
                       global_index % 4 == 2 ? 1 :
                       (global_index * 6364136223846793005ULL) % modulus;
      }
      for (uint64_t local_distance = 1; local_distance <= column_length / 2;
           local_distance *= 2) {
        auto values = input;
        for (uint32_t thread = 0; thread < kThreadCount; ++thread)
          ntt::dev_dif_col_stage(values.data(), column_length, local_distance,
                                transform_length, forward_powers.data(), modulus,
                                thread, kThreadCount, column);
        for (uint32_t thread = 0; thread < kThreadCount; ++thread)
          ntt::dev_dit_col_stage(values.data(), column_length, local_distance,
                                transform_length, inverse_powers.data(), modulus,
                                thread, kThreadCount, column);
        for (uint64_t index = 0; index < column_length; ++index)
          require(values[index] == ntt::add_mod(input[index], input[index], modulus),
                  "inverse(forward stage) must be 2*input");
      }
    }

    // Feed the inverse an independently generated CPU forward transform.
    std::vector<uint64_t> original(transform_length);
    for (uint64_t index = 0; index < transform_length; ++index)
      original[index] = index % 2 ? modulus - 1 - index : index;
    const ntt::NegacyclicNtt reference(transform_length, modulus, root);
    auto values = original;
    reference.forward(values);
    auto cpu_inverse = values;
    reference.inverse(cpu_inverse);

    // CUDA launch order, reproduced on CPU: near DIT, far DIT, scale.
    for (uint64_t row = 0; row < transform_length / kSplitWidth; ++row)
      for (uint64_t distance = 1; distance <= 128; distance *= 2)
        for (uint32_t thread = 0; thread < kThreadCount; ++thread)
          ntt::dev_dit_stage(values.data() + row * kSplitWidth, kSplitWidth,
                             distance, transform_length, inverse_powers.data(),
                             modulus, thread, kThreadCount);
    for (uint64_t column = 0; column < kSplitWidth; ++column) {
      std::vector<uint64_t> gathered(column_length);
      for (uint64_t index = 0; index < column_length; ++index)
        gathered[index] = values[column + index * kSplitWidth];
      for (uint64_t distance = 1; distance <= column_length / 2; distance *= 2)
        for (uint32_t thread = 0; thread < kThreadCount; ++thread)
          ntt::dev_dit_col_stage(gathered.data(), column_length, distance,
                                transform_length, inverse_powers.data(), modulus,
                                thread, kThreadCount, column);
      for (uint64_t index = 0; index < column_length; ++index)
        values[column + index * kSplitWidth] = gathered[index];
    }
    const uint64_t inverse_length = ntt::inv_mod(transform_length, modulus);
    for (uint64_t index = 0; index < transform_length; ++index) {
      require(values[index] == ntt::mul_mod(transform_length, original[index], modulus),
              "before normalization: inverse(forward) must be N*input");
      values[index] = ntt::mul_mod(values[index], inverse_length, modulus);
    }
    require(values == original && values == cpu_inverse, "normalized inverse == CPU == original");
  }
  std::cout << "INVERSE CORRESPONDENCE OK: N=" << transform_length
            << ", primes=2, stage factor=2, full factor=N, normalized identity\n";
}

} // namespace

int main(int argc, char **argv) {
  try {
    const bool inverse = argc == 2 && std::string(argv[1]) == "--inverse";
    require(argc == 1 || inverse, "usage: ntt_column_mapping_tests [--inverse]");
    for (const uint64_t length : {65536ULL}) {
      check_transform(length, inverse);
      if (inverse)
        check_inverse_correspondence(length);
    }
  } catch (const std::exception &error) {
    std::cerr << "FAIL: " << error.what() << '\n';
    return 1;
  }
  return 0;
}
