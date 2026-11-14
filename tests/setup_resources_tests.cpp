#include "ntt/setup_resources.hpp"
#include <algorithm>
#include <iostream>
#include <set>
#include <vector>
namespace {
void require(bool condition) {
  if (!condition) throw std::runtime_error("setup ownership check failed");
}
struct State {
  std::set<uint64_t *> live;
  std::vector<uint64_t *> allocated, released;
  size_t calls = 0, fail_runtime_at = 0;
  bool cleanup_error = false;
};
struct HostRuntime {
  State *state;
  void before() {
    if (++state->calls == state->fail_runtime_at)
      throw std::runtime_error("runtime failure");
  }
  uint64_t *allocate(size_t elements) {
    before();
    auto *pointer = new uint64_t[elements];
    state->live.insert(pointer);
    state->allocated.push_back(pointer);
    return pointer;
  }
  void upload(uint64_t *destination, const uint64_t *source, size_t elements) {
    before();
    require(state->live.count(destination) == 1);
    std::copy(source, source + elements, destination);
  }
  bool release(uint64_t *pointer) noexcept {
    state->released.push_back(pointer);
    state->live.erase(pointer);
    delete[] pointer;
    return !state->cleanup_error;
  }
};
void setup(ntt::cuda::detail::SetupResources<HostRuntime> &resources) {
  uint64_t *buffers[6];
  for (auto &buffer : buffers) buffer = resources.allocate(4);
  const uint64_t values[] = {0, 1, 2, 3};
  for (size_t index = 2; index < 6; ++index)
    resources.upload(buffers[index], values, 4);
}
} // namespace
int main() {
  try {
    using namespace ntt::cuda::detail;
    for (bool runtime_failure : {false, true}) {
      for (size_t step = 0; step <= 10; ++step) {
        State state;
        state.fail_runtime_at = runtime_failure ? step : 0;
        SetupTrace trace;
        bool failed = false;
        try {
          SetupResources<HostRuntime> resources({&state}, runtime_failure ? 0 : step, &trace);
          setup(resources);
        } catch (const std::runtime_error &) { failed = true; }
        require(failed == (step != 0) && state.live.empty());
        require(trace.allocations == (step ? std::min(step - 1, size_t{6}) : 6));
        require(trace.uploads == (step ? (step > 7 ? step - 7 : 0) : 4));
        require(trace.releases == trace.allocations && trace.release_failures == 0);
        std::reverse(state.allocated.begin(), state.allocated.end());
        require(state.released == state.allocated);
      }
    }
    State state;
    state.cleanup_error = true;
    SetupTrace trace;
    { SetupResources<HostRuntime> resources({&state}, 0, &trace); setup(resources); }
    require(trace.releases == 6 && trace.release_failures == 6 && state.live.empty());
    std::cout << "SETUP HOST PASS: 20 failures, success and cleanup reporting\n";
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n'; return 1;
  }
}
