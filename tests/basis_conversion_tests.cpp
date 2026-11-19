#include "ntt/basis_conversion.hpp"
#include <iostream>

namespace {
template <class Operation> void expect_invalid(Operation operation) {
  try {
    operation();
  } catch (const std::invalid_argument &) {
    return;
  }
  throw std::runtime_error("invalid input was accepted");
}
void validate_contract() {
  const ntt::ApproximateBasisPlan plan({17, 97}, {113});
  const std::vector<uint64_t> zeros(16);
  expect_invalid([&] { plan.convert(8, std::span<const uint64_t>(zeros).first(15)); });
  expect_invalid([&] { plan.convert(0, {}); });
  expect_invalid([&] { plan.convert(131072, {}); });
  expect_invalid([] { ntt::shoup_precompute(0, 1); });
  expect_invalid([] { ntt::shoup_precompute(17, 17); });
  expect_invalid([] { ntt::shoup_precompute(0, uint64_t{1} << 63); });
  auto source = std::vector<uint64_t>{17, 97};
  const ntt::ApproximateBasisPlan snapshot(source, {113});
  source[0] = 3;
  if (snapshot.source_moduli()[0] != 17 ||
      snapshot.convert(8, zeros) != std::vector<uint64_t>(8))
    throw std::runtime_error("plan snapshot or zero conversion changed");
  std::cout << "BASIS ADMISSION PASS\n";
}
} // namespace

int main(int argc, char **argv) try {
  if (argc == 2 && std::string(argv[1]) == "--validation") {
    validate_contract();
    return 0;
  }
  if (argc != 1)
    throw std::invalid_argument("usage: basis probe [--validation]");
  uint64_t dimension;
  size_t source_count, target_count;
  while (std::cin >> dimension >> source_count >> target_count) {
    if (source_count > 41 || target_count > 41 || dimension > 65536)
      throw std::invalid_argument("probe allocation bound");
    std::vector<uint64_t> source(source_count), target(target_count),
        values(dimension * source_count);
    for (auto &value : source)
      std::cin >> value;
    for (auto &value : target)
      std::cin >> value;
    for (auto &value : values)
      std::cin >> value;
    if (!std::cin)
      throw std::invalid_argument("truncated basis probe");
    try {
      const ntt::ApproximateBasisPlan plan(std::move(source), std::move(target));
      const auto result = plan.convert(dimension, values);
      for (size_t index = 0; index < result.size(); ++index)
        std::cout << (index ? " " : "") << result[index];
      std::cout << '\n';
    } catch (const std::invalid_argument &) {
      std::cout << "invalid\n";
    }
  }
  if (!std::cin.eof())
    throw std::invalid_argument("malformed basis probe");
  return 0;
} catch (const std::exception &error) {
  std::cerr << error.what() << '\n';
  return 1;
}
