#pragma once
// Profiling-only scopes. Ordinary benchmark builds compile the empty backend.
#ifdef NTT_BENCH_PROFILE_NVTX
#include <nvtx3/nvToolsExt.h>
#endif

namespace ntt::profile {
struct Disabled {
  static void push(const char *) noexcept {}
  static void pop() noexcept {}
};
#ifdef NTT_BENCH_PROFILE_NVTX
struct Nvtx {
  static nvtxDomainHandle_t domain() noexcept {
    static const auto value = nvtxDomainCreateA("ntt-gpu");
    return value;
  }
  static void push(const char *name) noexcept {
    nvtxEventAttributes_t attributes{};
    attributes.version = NVTX_VERSION;
    attributes.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
    attributes.messageType = NVTX_MESSAGE_TYPE_ASCII;
    attributes.message.ascii = name;
    nvtxDomainRangePushEx(domain(), &attributes);
  }
  static void pop() noexcept { nvtxDomainRangePop(domain()); }
};
using Backend = Nvtx;
#else
using Backend = Disabled;
#endif

template <class Sink> class BasicRange {
  bool active_ = true;
public:
  explicit BasicRange(const char *name) noexcept { Sink::push(name); }
  BasicRange(const BasicRange &) = delete;
  BasicRange &operator=(const BasicRange &) = delete;
  ~BasicRange() noexcept { end(); }
  void end() noexcept {
    if (active_) {
      Sink::pop();
      active_ = false;
    }
  }
};
using Range = BasicRange<Backend>;
} // namespace ntt::profile
