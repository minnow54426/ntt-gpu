#pragma once
// Shared host ownership logic. Runtime supplies allocate/upload/release;
// release must be noexcept and report success. No CUDA dependency here.
#include <array>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <utility>

namespace ntt::cuda::detail {
struct SetupTrace {
  size_t operations = 0, allocations = 0, uploads = 0;
  size_t releases = 0, release_failures = 0;
};
class InjectedSetupFailure : public std::runtime_error {
public:
  InjectedSetupFailure() : std::runtime_error("injected before setup operation") {}
};
template <class Runtime> class SetupResources {
  Runtime runtime_;
  std::array<uint64_t *, 6> pointers_{};
  size_t count_ = 0, failure_step_;
  SetupTrace local_trace_;
  SetupTrace *trace_;
  void before_operation() {
    if (++trace_->operations == failure_step_)
      throw InjectedSetupFailure();
  }
public:
  explicit SetupResources(Runtime runtime = {}, size_t failure_step = 0,
                          SetupTrace *trace = nullptr)
      : runtime_(std::move(runtime)), failure_step_(failure_step),
        trace_(trace ? trace : &local_trace_) {}
  SetupResources(const SetupResources &) = delete;
  SetupResources &operator=(const SetupResources &) = delete;
  Runtime &runtime() { return runtime_; }
  uint64_t *allocate(size_t elements) {
    if (!elements || count_ == pointers_.size())
      throw std::invalid_argument("invalid setup allocation extent/count");
    before_operation();
    auto *pointer = runtime_.allocate(elements);
    if (!pointer)
      throw std::runtime_error("runtime returned null allocation");
    pointers_[count_++] = pointer;
    ++trace_->allocations;
    return pointer;
  }
  void upload(uint64_t *destination, const uint64_t *source, size_t elements) {
    before_operation();
    runtime_.upload(destination, source, elements);
    ++trace_->uploads;
  }
  ~SetupResources() noexcept {
    while (count_) {
      ++trace_->releases;
      if (!runtime_.release(pointers_[--count_]))
        ++trace_->release_failures;
    }
  }
};
} // namespace ntt::cuda::detail
