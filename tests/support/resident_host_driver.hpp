#pragma once
// Deferred CPU transport for runtime tests. Never represents GPU execution.
#include "ntt/resident_driver.hpp"
#include "ntt/ntt.hpp"
#include "ntt/resident_conversion.hpp"
#include <functional>

namespace resident_test {
using namespace ntt::resident;
struct Trace {
  size_t calls = 0, live_buffers = 0, live_events = 0, executed = 0;
  size_t fail_step = 0, partial_step = 0, async_fail_at = 0, drains = 0;
  bool allow_drain = true, corrupt_download = false;
  bool cleanup_error = false;
  std::vector<size_t> enqueue_calls;
};
struct HostBuffer final : detail::Buffer {
  std::shared_ptr<Trace> trace;
  std::vector<uint64_t> values;
  HostBuffer(std::shared_ptr<Trace> t, size_t n) : trace(std::move(t)), values(n) { ++trace->live_buffers; }
  ~HostBuffer() override { --trace->live_buffers; }
};
struct HostEvent final : detail::Event {
  std::shared_ptr<Trace> trace;
  size_t fence = 0;
  bool recorded = false;
  explicit HostEvent(std::shared_ptr<Trace> t) : trace(std::move(t)) { ++trace->live_events; }
  ~HostEvent() override { --trace->live_events; }
};
class HostDriver final : public detail::Driver {
  PolynomialLayout layout_;
  std::vector<std::function<void()>> queue_;
  size_t done_ = 0;
  bool failed_ = false;
  void tick() {
    if (++trace->calls == trace->fail_step) throw std::runtime_error("injected transport failure");
  }
  void enqueue(std::function<void()> work) {
    tick(); queue_.push_back(std::move(work));
    trace->enqueue_calls.push_back(trace->calls);
    if (trace->calls == trace->partial_step) throw std::runtime_error("failure after enqueue");
  }
  void through(size_t fence) {
    if (failed_) throw std::runtime_error("host test queue failed");
    while (done_ < fence) {
      if (done_ + 1 == trace->async_fail_at) {
        failed_ = true; throw std::runtime_error("injected asynchronous failure");
      }
      queue_[done_]();
      queue_[done_] = {};
      ++done_; ++trace->executed;
    }
  }
  static HostBuffer &buffer(detail::Buffer &b) { return dynamic_cast<HostBuffer &>(b); }
  static const HostBuffer &buffer(const detail::Buffer &b) { return dynamic_cast<const HostBuffer &>(b); }
public:
  std::shared_ptr<Trace> trace = std::make_shared<Trace>();
  explicit HostDriver(PolynomialLayout p) : layout_(std::move(p)) { ntt::proposal::validate(layout_); }
  int ordinal() const override { return 0; }
  const PolynomialLayout &layout() const override { return layout_; }
  size_t constant_bytes() const override { return 0; }
  std::unique_ptr<detail::Buffer> allocate(size_t n) override { tick(); return std::make_unique<HostBuffer>(trace, n); }
  std::unique_ptr<detail::Event> make_event() override { tick(); return std::make_unique<HostEvent>(trace); }
  void upload(detail::Buffer &out, std::span<const uint64_t> source) override {
    auto *b = &buffer(out);
    enqueue([b, source] { std::copy(source.begin(), source.end(), b->values.begin()); });
  }
  void copy(detail::Buffer &out, const detail::Buffer &input, size_t offset, size_t n) override {
    auto *a = &buffer(out); auto *b = &buffer(input);
    enqueue([a, b, offset, n] { std::copy_n(b->values.begin() + offset, n, a->values.begin()); });
  }
  void copy_range(detail::Buffer &out, size_t destination, const detail::Buffer &input,
                   size_t source, size_t n) override {
    auto *a = &buffer(out); auto *b = &buffer(input);
    ntt::proposal::require(destination <= a->values.size() && n <= a->values.size() - destination
        && source <= b->values.size() && n <= b->values.size() - source, "host copy range extent");
    enqueue([a, b, destination, source, n] {
      std::copy_n(b->values.begin() + source, n, a->values.begin() + destination);
    });
  }
  void transform(detail::Buffer &out, const PolynomialLayout &p, bool inverse) override {
    auto *b = &buffer(out);
    enqueue([b, p, inverse] {
      const auto n = p.parameters->n();
      for (size_t k = 0; k < p.towers.size(); ++k) {
        const auto t = p.parameters->tower(p.towers[k]);
        std::vector<uint64_t> values(b->values.begin() + k * n, b->values.begin() + (k + 1) * n);
        ntt::NegacyclicNtt ref(n, t.q, t.psi);
        if (inverse) ref.inverse(values); else ref.forward(values);
        std::copy(values.begin(), values.end(), b->values.begin() + k * n);
      }
    });
  }
  void pointwise(detail::Buffer &out, const detail::Buffer &right, size_t offset,
                 const PolynomialLayout &p) override {
    auto *a = &buffer(out); auto *b = &buffer(right);
    enqueue([a, b, offset, p] {
      const auto n = p.parameters->n();
      for (size_t k = 0; k < p.towers.size(); ++k)
        for (size_t i = 0; i < n; ++i)
          a->values[k * n + i] = ntt::mul_mod(a->values[k * n + i], b->values[offset + k * n + i],
                                             p.parameters->tower(p.towers[k]).q);
    });
  }
  void switch_basis(detail::Buffer &out, const detail::Buffer &input, size_t offset,
                     const detail::Buffer &constants, const PolynomialLayout &source,
                     const PolynomialLayout &target) override {
    auto *a = &buffer(out); auto *b = &buffer(input); auto *table = &buffer(constants);
    enqueue([a, b, table, offset, source, target] {
      for (size_t i = 0; i < a->values.size(); ++i)
        a->values[i] = ntt::resident::detail::resident_basis_value(b->values.data() + offset,
            source.parameters->n(), i, source.towers.size(), target.towers.size(), table->values.data());
    });
  }
  void correct(detail::Buffer &out, const detail::Buffer &left, size_t left_offset,
                 const detail::Buffer &right, size_t right_offset, const detail::Buffer &constants,
                 const PolynomialLayout &layout) override {
    auto *a = &buffer(out); auto *b = &buffer(left); auto *c = &buffer(right); auto *table = &buffer(constants);
    enqueue([a, b, c, table, left_offset, right_offset, layout] {
      for (size_t i = 0; i < a->values.size(); ++i)
        a->values[i] = ntt::resident::detail::resident_correction_value(
            b->values[left_offset+i], c->values[right_offset+i], i/layout.parameters->n(),
            layout.towers.size(), table->values.data());
    });
  }
  void download(std::span<uint64_t> destination, const detail::Buffer &input, size_t offset) override {
    auto *b = &buffer(input); auto corrupt = trace->corrupt_download;
    enqueue([b, destination, offset, corrupt] {
      std::copy_n(b->values.begin() + offset, destination.size(), destination.begin());
      if (corrupt) destination[0] = UINT64_MAX;
    });
  }
  void record(detail::Event &event) override {
    tick(); auto &e = dynamic_cast<HostEvent &>(event); e.fence = queue_.size(); e.recorded = true;
  }
  bool ready(detail::Event &event) override {
    tick(); auto &e = dynamic_cast<HostEvent &>(event);
    if (failed_) throw std::runtime_error("host test queue failed");
    return e.recorded && done_ >= e.fence;
  }
  void wait(detail::Event &event) override {
    tick(); auto &e = dynamic_cast<HostEvent &>(event);
    if (!e.recorded) throw std::runtime_error("unrecorded host test event");
    through(e.fence);
  }
  bool quiesce() noexcept override {
    ++trace->drains;
    if (!trace->allow_drain) return false;
    try { through(queue_.size()); } catch (...) {}
    // Unlike a failed CUDA context, this synchronous fake can prove cancellation.
    queue_.clear(); done_ = 0; return true;
  }
  void check_health() const override {
    if (trace->cleanup_error) throw std::runtime_error("injected cleanup error");
  }
};
} // namespace resident_test
