// A deliberately tiny NTT program for learning with a source debugger.
// N=8 has only three butterfly stages, so the complete transform is easy
// to inspect. Set a breakpoint on transform.forward(values), step into it,
// and watch stage_pair_distance, group_start_index, pair_offset, and values.

#include <cstdint>
#include <iostream>
#include <vector>

#include "ntt/ntt.hpp"

namespace {

void print_values(const char *label, const std::vector<uint64_t> &values) {
  std::cout << label << ":";
  for (uint64_t value : values)
    std::cout << ' ' << value;
  std::cout << '\n';
}

} // namespace

int main() {
  constexpr uint64_t kTransformLength = 8;
  constexpr uint64_t kModulus = 17;
  constexpr uint64_t kPrimitiveSixteenthRoot = 3;

  const ntt::NegacyclicNtt transform(kTransformLength, kModulus,
                                     kPrimitiveSixteenthRoot);
  std::vector<uint64_t> values = {1, 2, 3, 4, 5, 6, 7, 8};
  const std::vector<uint64_t> original_values = values;

  print_values("natural-order coefficients", values);

  transform.forward(values); // Put the first breakpoint here.
  print_values("bit-reversed evaluation values", values);

  transform.inverse(values); // Put the second breakpoint here.
  print_values("recovered coefficients", values);

  if (values != original_values) {
    std::cerr << "roundtrip failed\n";
    return 1;
  }

  std::cout << "roundtrip passed\n";
  return 0;
}
