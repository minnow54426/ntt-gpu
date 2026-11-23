#include "ntt/benchmark_profile.hpp"
#include <array>
#include <stdexcept>
#include <string_view>
#include <iostream>

namespace {
struct Sink {
  static inline std::array<std::string_view, 16> events{};
  static inline size_t count = 0;
  static void push(const char *name) noexcept { events[count++] = name; }
  static void pop() noexcept { events[count++] = "pop"; }
};
}
int main() {
  using Range = ntt::profile::BasicRange<Sink>;
  {
    Range setup("setup");
    setup.end();
    setup.end();
    try {
      Range phase("sample");
      Range operation("operation");
      throw std::runtime_error("operation failed");
    } catch (const std::runtime_error &) {}
    { Range verify("verify"); }
  }
  const std::array<std::string_view, 8> expected{
      "setup", "pop", "sample", "operation", "pop", "pop", "verify", "pop"};
  if (Sink::count != expected.size()) return 1;
  for (size_t index = 0; index < expected.size(); ++index)
    if (Sink::events[index] != expected[index]) return 1;
  std::cout << "Profile scopes close once, nest in order and unwind on failure.\n";
}
