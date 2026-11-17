// ntt_device_sim: portable simulator of the CUDA kernel schedule.
//
// Runs the SAME stage bodies the GPU kernels run (include/ntt/transform_stage.hpp,
// the single source of schedule truth) with std::vector tiles and a serial
// loop over thread ids standing in for thread blocks, and checks the result
// against the CPU reference on identical vectors. Green here means the
// index arithmetic, stage order, two-pass split, and twiddle tables are
// right — proven on this machine, before any GPU is rented.
//
// Catches: index bugs, wrong stage sequence, tile-local-NTT mistakes,
// table-building errors. Cannot catch (it is serial): missing-barrier
// races, nvcc compile errors, launch-config mistakes — the kernels are
// thin wrappers around the shared bodies precisely to keep that surface
// small; the GPU gates cover it on the rented box.
//
// Usage: ntt_device_sim <vector-file> [<vector-file> ...]

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "ntt/transform_stage.hpp"
#include "ntt/ntt.hpp"
#include "ntt/vector_io.hpp"

#ifdef NTT_SIM_COMPARE_OPENFHE
#include "math/hal/intnat/transformnat.h"
#endif
using ntt::transform_stage;
using ntt::transform_twiddles;
using ntt::inv_mod;
using ntt::mul_mod;

namespace {

int failures = 0;

void check(bool condition, const std::string &description) {
  if (!condition) {
    ++failures;
    std::cerr << "FAIL: " << description << "\n";
  }
}

#ifdef NTT_SIM_COMPARE_OPENFHE
void compare_openfhe(const std::vector<uint64_t> &input,
                     const std::vector<uint64_t> &expected, uint64_t modulus,
                     uint64_t root, bool inverse) {
  intnat::NativeVector values(input.size(), intnat::NativeInteger(modulus));
  for (size_t i = 0; i < input.size(); ++i)
    values[i] = intnat::NativeInteger(input[i]);
  intnat::ChineseRemainderTransformFTTNat<intnat::NativeVector> transform;
  const intnat::NativeInteger psi(root);
  const auto order = static_cast<uint32_t>(2 * input.size());
  if (inverse)
    transform.InverseTransformFromBitReverseInPlace(psi, order, &values);
  else
    transform.ForwardTransformToBitReverseInPlace(psi, order, &values);
  for (size_t i = 0; i < expected.size(); ++i) {
    if (values[i].ConvertToInt() != expected[i]) {
      check(false, std::string(inverse ? "inverse" : "forward") +
                   " schedule != OpenFHE at index " + std::to_string(i));
      return;
    }
  }
}
#endif

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

// The launch schedule of CudaNttTransform (src/cuda/ntt_kernels.cu),
// mirrored line for line: strided-group stages followed by contiguous-tile
// stages. Threads are emulated by calling each stage body once per logical
// thread. A stage's butterfly pairs are disjoint, so serial execution is
// equivalent to any GPU thread interleaving.
class ScheduleSim {
public:
  ScheduleSim(uint64_t transform_length, const std::vector<uint64_t> &moduli,
              const std::vector<uint64_t> &primitive_roots)
      : transform_length_(transform_length), moduli_(moduli) {
    for (size_t limb_index = 0; limb_index < moduli_.size(); ++limb_index) {
      // Parameter validation identical to the CUDA path: constructing the
      // reference throws on any bad wheel/root before tables are built.
      cpu_references_.emplace_back(transform_length_, moduli_[limb_index],
                                   primitive_roots[limb_index]);
      const auto modulus = moduli_[limb_index];
      const auto root = primitive_roots[limb_index];
      twiddle_powers_.push_back(transform_twiddles(transform_length_, root, modulus));
      inverse_twiddle_powers_.push_back(
          transform_twiddles(transform_length_, inv_mod(root, modulus), modulus));
      inverse_transform_lengths_.push_back(
          inv_mod(transform_length_, moduli_[limb_index]));
    }
  }

  const ntt::NegacyclicNtt &reference(size_t limb_index) const {
    return cpu_references_[limb_index];
  }

  // Natural coefficients in, values out in bit-reversed order — same
  // contract as NegacyclicNtt::forward, asserted exactly by the gates.
  // The schedule is the two-pass split: pass 1 over 256 strided groups
  // (the far stages, d a multiple of 256), pass 2 over contiguous
  // 256-tiles (the near stages), mirroring the kernels below.
  void forward(std::vector<uint64_t> &values, size_t limb_index) const {
    check_size(values);
    // Pass 1 — each block owns one index modulo 256 and gathers all values
    // at that stride. The far butterfly stages run entirely inside it.
    const uint64_t strided_group_length = transform_length_ / kPassSplitWidth;
    if (strided_group_length >= 2) {
      for (uint64_t strided_group_offset = 0; strided_group_offset < kPassSplitWidth;
           ++strided_group_offset) {
        std::vector<uint64_t> gathered_group_values(strided_group_length);
        for (uint64_t local_index = 0; local_index < strided_group_length;
             ++local_index)
          gathered_group_values[local_index] =
              values[strided_group_offset + local_index * kPassSplitWidth];
        for (uint64_t local_pair_distance = strided_group_length / 2;
             local_pair_distance >= 1; local_pair_distance >>= 1)
          launch_stage(transform_stage<false, kPassSplitWidth>, gathered_group_values.data(),
                                     strided_group_length, local_pair_distance,
                                     twiddle_powers_[limb_index].data(),
                                     moduli_[limb_index], strided_group_offset);
        for (uint64_t local_index = 0; local_index < strided_group_length;
             ++local_index)
          values[strided_group_offset + local_index * kPassSplitWidth] =
              gathered_group_values[local_index];
      }
    }
    // Pass 2 — near butterfly stages stay inside a contiguous 256-value tile.
    const uint64_t tile_length = std::min<uint64_t>(kPassSplitWidth, transform_length_);
    for (uint64_t tile_start_index = 0; tile_start_index < transform_length_;
         tile_start_index += tile_length) {
      std::vector<uint64_t> tile_values(values.begin() + tile_start_index,
                                        values.begin() + tile_start_index +
                                            tile_length);
      for (uint64_t pair_distance = tile_length / 2; pair_distance >= 1;
           pair_distance >>= 1)
        launch_stage(transform_stage<false, 1>, tile_values.data(), tile_length, pair_distance,
                          twiddle_powers_[limb_index].data(),
                          moduli_[limb_index], tile_start_index);
      std::copy(tile_values.begin(), tile_values.end(),
                values.begin() + tile_start_index);
    }
  }

  // Bit-reversed values in, natural coefficients out: the mirror of
  // forward — contiguous stages first, then strided stages, followed by
  // elementwise multiplication by the inverse transform length.
  void inverse(std::vector<uint64_t> &values, size_t limb_index) const {
    check_size(values);
    const uint64_t strided_group_length = transform_length_ / kPassSplitWidth;
    const uint64_t tile_length = std::min<uint64_t>(kPassSplitWidth, transform_length_);
    for (uint64_t tile_start_index = 0; tile_start_index < transform_length_;
         tile_start_index += tile_length) {
      std::vector<uint64_t> tile_values(values.begin() + tile_start_index,
                                        values.begin() + tile_start_index +
                                            tile_length);
      for (uint64_t pair_distance = 1; pair_distance <= tile_length / 2;
           pair_distance <<= 1)
        launch_stage(transform_stage<true, 1>, tile_values.data(), tile_length, pair_distance,
                          inverse_twiddle_powers_[limb_index].data(),
                          moduli_[limb_index], tile_start_index);
      std::copy(tile_values.begin(), tile_values.end(),
                values.begin() + tile_start_index);
    }
    if (strided_group_length >= 2) {
      for (uint64_t strided_group_offset = 0; strided_group_offset < kPassSplitWidth;
           ++strided_group_offset) {
        std::vector<uint64_t> gathered_group_values(strided_group_length);
        for (uint64_t local_index = 0; local_index < strided_group_length;
             ++local_index)
          gathered_group_values[local_index] =
              values[strided_group_offset + local_index * kPassSplitWidth];
        for (uint64_t local_pair_distance = 1;
             local_pair_distance <= strided_group_length / 2; local_pair_distance <<= 1)
          launch_stage(transform_stage<true, kPassSplitWidth>, gathered_group_values.data(),
                                     strided_group_length, local_pair_distance,
                                     inverse_twiddle_powers_[limb_index].data(),
                                     moduli_[limb_index], strided_group_offset);
        for (uint64_t local_index = 0; local_index < strided_group_length;
             ++local_index)
          values[strided_group_offset + local_index * kPassSplitWidth] =
              gathered_group_values[local_index];
      }
    }
    for (uint64_t value_index = 0; value_index < transform_length_; ++value_index)
      values[value_index] =
          mul_mod(values[value_index], inverse_transform_lengths_[limb_index],
                  moduli_[limb_index]);
  }

  // The full ring product on this schedule: forward both, N pointwise
  // multiplies (the entire payoff), inverse — checked against
  // ntt_multiply, which unit_tests.cpp already pins to schoolbook.
  std::vector<uint64_t> multiply(std::vector<uint64_t> left_values,
                                 std::vector<uint64_t> right_values,
                                 size_t limb_index) const {
    check_size(left_values);
    check_size(right_values);
    forward(left_values, limb_index);
    forward(right_values, limb_index);
    for (uint64_t value_index = 0; value_index < transform_length_; ++value_index)
      left_values[value_index] = mul_mod(
          left_values[value_index], right_values[value_index], moduli_[limb_index]);
    inverse(left_values, limb_index);
    return left_values;
  }

private:
  using Stage = void (*)(uint64_t *, uint64_t, uint64_t, uint64_t,
                         const uint64_t *, uint64_t, uint32_t, uint32_t, uint64_t);

  void launch_stage(Stage stage, uint64_t *values, uint64_t window_length,
                    uint64_t local_distance, const uint64_t *twiddles,
                    uint64_t modulus, uint64_t origin) const {
    for (uint32_t thread = 0; thread < kThreadsPerBlock; ++thread)
      stage(values, window_length, local_distance, transform_length_, twiddles,
            modulus, thread, kThreadsPerBlock, origin);
  }

  void check_size(const std::vector<uint64_t> &values) const {
    if (values.size() != transform_length_)
      throw std::length_error("vector size " + std::to_string(values.size()) +
                              " != n " + std::to_string(transform_length_));
  }

  // Must match the kernel constants in src/cuda/ntt_kernels.cu.
  static constexpr uint32_t kThreadsPerBlock = 256;
  static constexpr uint64_t kPassSplitWidth = 256;

  uint64_t transform_length_;
  std::vector<uint64_t> moduli_;
  std::vector<ntt::NegacyclicNtt> cpu_references_;
  std::vector<std::vector<uint64_t>> twiddle_powers_;
  std::vector<std::vector<uint64_t>> inverse_twiddle_powers_;
  std::vector<uint64_t> inverse_transform_lengths_;
};

// Per RNS limb: schedule-forward == reference-forward element for element
// (bit-reversed order included — this is the equality the CUDA gates later
// assert on the GPU), schedule-roundtrip == input, and the full schedule
// multiply == the reference ntt_multiply.
void test_with_vectors(const ntt::VectorFile &vector_file) {
  const ScheduleSim simulator(vector_file.n, vector_file.q, vector_file.psi);
  for (size_t limb_index = 0; limb_index < vector_file.q.size(); ++limb_index) {
    std::vector<uint64_t> original_values(vector_file.n);
    std::vector<uint64_t> scheduled_values(vector_file.n);
    std::vector<uint64_t> reference_values(vector_file.n);
    for (uint64_t value_index = 0; value_index < vector_file.n; ++value_index) {
      const uint64_t reduced_value = ntt::reduce_decimal_mod(
          vector_file.coeffs[value_index], vector_file.q[limb_index]);
      original_values[value_index] = reduced_value;
      scheduled_values[value_index] = reduced_value;
      reference_values[value_index] = reduced_value;
    }

    simulator.forward(scheduled_values, limb_index);
    simulator.reference(limb_index).forward(reference_values);
    check(scheduled_values == reference_values,
          "schedule forward == reference forward, limb " + std::to_string(limb_index));
#ifdef NTT_SIM_COMPARE_OPENFHE
    compare_openfhe(original_values, scheduled_values, vector_file.q[limb_index],
                    vector_file.psi[limb_index], false);
#endif

    simulator.inverse(scheduled_values, limb_index);
    check(scheduled_values == original_values,
          "schedule roundtrip identity, limb " + std::to_string(limb_index));

    Lcg random_generator(0x9E3779B97F4A7C15ULL ^ vector_file.q[limb_index]);
    std::vector<uint64_t> right_operand(vector_file.n);
    for (uint64_t value_index = 0; value_index < vector_file.n; ++value_index)
      right_operand[value_index] = random_generator.next(vector_file.q[limb_index]);
    auto scheduled_inverse = right_operand, reference_inverse = right_operand;
    simulator.inverse(scheduled_inverse, limb_index);
    simulator.reference(limb_index).inverse(reference_inverse);
    check(scheduled_inverse == reference_inverse,
          "independent inverse == reference inverse, limb " + std::to_string(limb_index));
#ifdef NTT_SIM_COMPARE_OPENFHE
    compare_openfhe(right_operand, scheduled_inverse, vector_file.q[limb_index],
                    vector_file.psi[limb_index], true);
#endif
    check(simulator.multiply(original_values, right_operand, limb_index) ==
              ntt::ntt_multiply(simulator.reference(limb_index), original_values,
                                right_operand),
          "schedule multiply == reference multiply, limb " +
              std::to_string(limb_index));
  }
}

} // namespace

int main(int argc, char **argv) {
  if (argc < 2) {
    std::cerr << "usage: ntt_device_sim <vector-file> [<vector-file> ...]\n";
    return 2;
  }
  for (int argument_index = 1; argument_index < argc; ++argument_index) {
    try {
      const ntt::VectorFile vector_file = ntt::read_vector_file(argv[argument_index]);
      test_with_vectors(vector_file);
      std::cout << "DEVICE SCHEDULE OK: n=" << vector_file.n
                << ", limbs=" << vector_file.q.size() << " (" << argv[argument_index]
                << ")\n";
    } catch (const std::exception &error) {
      std::cerr << argv[argument_index] << ": " << error.what() << "\n";
      return 1;
    }
  }
  return failures == 0 ? 0 : 1;
}
