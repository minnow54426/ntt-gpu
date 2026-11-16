#include "ntt/resident_driver.hpp"
#include <atomic>
#include <unordered_map>

namespace ntt::resident {
using namespace proposal;
namespace detail {
struct Budget {
  Limits limits;
  size_t device = 0, host = 0;
};
struct Reservation {
  std::shared_ptr<Budget> budget;
  size_t count;
  bool host;
  Reservation(std::shared_ptr<Budget> b, size_t n, bool h)
      : budget(std::move(b)), count(n), host(h) {
    auto &used = host ? budget->host : budget->device;
    const auto limit = host ? budget->limits.host_bytes : budget->limits.device_bytes;
    require(used <= limit && n <= limit - used, "resident memory budget exhausted");
    used += n;
  }
  ~Reservation() { (host ? budget->host : budget->device) -= count; }
  Reservation(const Reservation &) = delete;
  Reservation &operator=(const Reservation &) = delete;
};
struct Allocation {
  Reservation reservation;
  AllocationRecord record;
  // The device buffer is destroyed before its budget reservation.
  std::unique_ptr<Buffer> buffer;
  Allocation(std::shared_ptr<Budget> b, AllocationRecord r)
      : reservation(std::move(b), r.capacity_bytes, false), record(std::move(r)) {}
};
struct HostStorage {
  Reservation reservation;
  std::vector<uint64_t> values;
  HostStorage(std::shared_ptr<Budget> b, size_t count)
      : reservation(std::move(b), checked_product(count, sizeof(uint64_t)), true), values(count) {}
};
struct Operation {
  CompletionRecord record;
  std::unique_ptr<Event> event;
  std::vector<std::shared_ptr<Allocation>> allocations;
  std::vector<std::shared_ptr<Operation>> dependencies;
  std::shared_ptr<HostStorage> staging;
  explicit Operation(CompletionRecord r) : record(r) {}
  void unpin() noexcept {
    event.reset();
    allocations.clear();
    dependencies.clear();
    staging.reset();
  }
};
uint64_t next_session() {
  static std::atomic<uint64_t> serial{1};
  auto value = serial.load();
  do {
    require(value != std::numeric_limits<uint64_t>::max(), "session identity exhausted");
  } while (!serial.compare_exchange_weak(value, value + 1));
  return value;
}
struct State : std::enable_shared_from_this<State> {
  std::shared_ptr<Driver> driver;
  PolynomialLayout base;
  Device device;
  Stream stream;
  std::shared_ptr<Budget> budget;
  Reservation constants;
  std::unordered_map<uint64_t, std::weak_ptr<Allocation>> registry;
  // Bounded tombstones remain after pins retire, including submissions that
  // throw before a completion handle can be returned.
  std::vector<std::shared_ptr<Operation>> operations;
  std::shared_ptr<State> flight_lease;
  std::exception_ptr error;
  std::exception_ptr drain_error = std::make_exception_ptr(std::runtime_error("resident quiescence unproven"));
  uint64_t allocation_id = 0;
  size_t uploads = 0, downloads = 0, copies = 0;
  bool closed = false, quiescent = true;
  State(std::shared_ptr<Driver> d, Limits limits)
      : driver(std::move(d)), base(driver->layout()), device{driver->ordinal(), next_session()},
        stream{device, {1, 1}}, budget(std::make_shared<Budget>(Budget{limits})),
        constants(budget, driver->constant_bytes(), false) {
    validate(base);
    validate(device);
    require(base.domain == Domain::Coefficient && base.order == Order::Natural,
            "session requires coefficient layout");
    require(base.parameters->n() <= 65536 && base.towers.size() <= 65535,
            "prototype extent unsupported");
    require(limits.operations > 0, "empty resident operation budget");
    driver->attach_once();
  }
  void open() const { require(!closed && !error, "resident session closed or poisoned"); }
  void check(const Polynomial &p) const {
    require(p.state_.get() == this && p.allocation_ && p.producer_, "foreign or empty polynomial");
    const auto it = registry.find(p.view_.allocation.id);
    require(it != registry.end() && it->second.lock() == p.allocation_, "unregistered allocation");
    const auto actual = validate_view(p.view_, p.allocation_->record, device);
    require_layout(actual, p.layout_);
    require(p.producer_->record.outcome != Outcome::Failed, "failed producer");
  }
  std::shared_ptr<Allocation> reserve(const PolynomialLayout &layout) {
    for (auto it = registry.begin(); it != registry.end();)
      if (it->second.expired()) it = registry.erase(it);
      else ++it;
    require(allocation_id != std::numeric_limits<uint64_t>::max(), "allocation identity exhausted");
    auto a = std::make_shared<Allocation>(budget,
        AllocationRecord{{++allocation_id, 1}, device, bytes(layout), layout});
    registry.emplace(a->record.token.id, a);
    return a;
  }
  std::shared_ptr<Operation> begin(std::vector<std::shared_ptr<Allocation>> pins,
                                  std::vector<std::shared_ptr<Operation>> sources,
                                  std::span<const Completion> waits,
                                  std::shared_ptr<HostStorage> staging = {}) {
    open();
    require(operations.size() < budget->limits.operations, "resident operation budget exhausted");
    auto op = std::make_shared<Operation>(CompletionRecord{{operations.size() + 1, 1}, stream, Outcome::Pending});
    Execution execution{stream, op->record.operation.id, {}};
    for (const auto &wait : waits) {
      require(wait.state_.get() == this && wait.operation_, "foreign or empty dependency");
      execution.waits.push_back(wait.operation_->record);
      sources.push_back(wait.operation_);
    }
    for (const auto &source : sources)
      validate_dependency(source->record, execution);
    op->allocations = std::move(pins);
    op->dependencies = std::move(sources);
    op->staging = std::move(staging);
    operations.push_back(op); // All pins/journal storage exist before native work.
    flight_lease = shared_from_this(); // No allocation; retains failed-context quarantine.
    quiescent = false;
    return op;
  }
  void finish(const std::shared_ptr<Operation> &op) {
    driver->record(*op->event);
  }
  bool drain() noexcept {
    if (!driver->quiesce()) return false;
    quiescent = true;
    for (auto &op : operations) {
      if (op->record.outcome == Outcome::Pending)
        op->record.outcome = error ? Outcome::Failed : Outcome::Succeeded;
      op->unpin();
    }
    flight_lease.reset();
    return true;
  }
  void fail(std::exception_ptr reason) noexcept {
    if (!error) error = reason;
    for (auto &op : operations)
      if (op->record.outcome == Outcome::Pending) op->record.outcome = Outcome::Failed;
    (void)drain(); // False keeps journal, staging, buffers, driver and self lease.
  }
  void poll(const std::shared_ptr<Operation> &op, bool wait) {
    if (op->record.outcome == Outcome::Failed) std::rethrow_exception(error);
    if (op->record.outcome == Outcome::Succeeded) return;
    try {
      if (wait) driver->wait(*op->event);
      else if (!driver->ready(*op->event)) return;
      for (auto &prior : operations) {
        if (prior->record.operation.id > op->record.operation.id) break;
        if (prior->record.outcome == Outcome::Pending) prior->record.outcome = Outcome::Succeeded;
        prior->unpin();
      }
      if (op == operations.back()) {
        quiescent = true;
        flight_lease.reset();
      }
    } catch (...) { fail(std::current_exception()); throw; }
  }
  size_t offset(const Polynomial &p) const {
    return byte_offset(p.view_, p.allocation_->record) / sizeof(uint64_t);
  }
  Polynomial output(const std::shared_ptr<Allocation> &a, const std::shared_ptr<Operation> &op) {
    return Polynomial(shared_from_this(), a, op, a->record.layout,
                       {a->record.token, 0, a->record.layout.towers.size(), a->record.layout.active_q});
  }
};
} // namespace detail

Completion::Completion(std::shared_ptr<detail::State> s, std::shared_ptr<detail::Operation> o)
    : state_(std::move(s)), operation_(std::move(o)) {}
CompletionRecord Completion::query() const {
  require(state_ && operation_, "empty completion");
  try { state_->poll(operation_, false); }
  catch (...) { if (operation_->record.outcome != Outcome::Failed) throw; }
  return operation_->record;
}
void Completion::wait() const {
  require(state_ && operation_, "empty completion");
  state_->poll(operation_, true);
}
Polynomial::Polynomial(std::shared_ptr<detail::State> s, std::shared_ptr<detail::Allocation> a,
                       std::shared_ptr<detail::Operation> o, PolynomialLayout p, View v)
    : state_(std::move(s)), allocation_(std::move(a)), producer_(std::move(o)), layout_(std::move(p)), view_(v) {}
const PolynomialLayout &Polynomial::layout() const {
  require(bool(allocation_), "empty polynomial"); return layout_;
}
View Polynomial::view() const { require(bool(allocation_), "empty polynomial"); return view_; }
Completion Polynomial::readiness() const { return Completion(state_, producer_); }
Download::Download(std::shared_ptr<detail::State> s, std::shared_ptr<detail::Operation> o,
                   std::shared_ptr<detail::HostStorage> h, PolynomialLayout p)
    : state_(std::move(s)), operation_(std::move(o)), storage_(std::move(h)), layout_(std::move(p)) {}
Completion Download::completion() const { return Completion(state_, operation_); }
std::span<const uint64_t> Download::wait_values() const {
  require(state_ && operation_ && storage_, "empty download");
  completion().wait();
  if (!validated_) {
    try { validate_host_values(layout_, storage_->values); validated_ = true; }
    catch (...) { operation_->record.outcome = Outcome::Failed; state_->fail(std::current_exception()); throw; }
  }
  return storage_->values;
}
Session::Session(std::shared_ptr<detail::Driver> driver, Limits limits) {
  require(bool(driver), "missing resident driver");
  state_ = std::make_shared<detail::State>(std::move(driver), limits);
}
Session::~Session() noexcept { if (state_) try { close(); } catch (...) {} }
Device Session::device() const { require(bool(state_), "moved session"); return state_->device; }
Stats Session::stats() const {
  require(bool(state_), "moved session");
  size_t pending = 0;
  for (const auto &op : state_->operations) pending += op->record.outcome == Outcome::Pending;
  return {state_->operations.size(), pending, state_->budget->device, state_->budget->host,
          state_->uploads, state_->downloads, state_->copies, state_->closed, bool(state_->error), state_->quiescent};
}
std::vector<CompletionRecord> Session::history() const {
  require(bool(state_), "moved session");
  std::vector<CompletionRecord> out;
  for (const auto &op : state_->operations) out.push_back(op->record);
  return out;
}
Polynomial Session::upload(std::span<const uint64_t> values, std::span<const Completion> waits) {
  require(bool(state_), "moved session"); auto &s = *state_; s.open();
  validate_host_values(s.base, values);
  return upload_checked(s.base, values, waits);
}
Polynomial Session::upload_evaluation(std::span<const uint64_t> values, std::span<const Completion> waits) {
  require(bool(state_), "moved session"); state_->open();
  auto layout = transform_layout(state_->base, false);
  validate_host_values(layout, values);
  return upload_checked(layout, values, waits);
}
Polynomial Session::upload_checked(const PolynomialLayout &layout, std::span<const uint64_t> values,
                                    std::span<const Completion> waits) {
  auto &s = *state_;
  auto staging = std::make_shared<detail::HostStorage>(s.budget, values.size());
  std::copy(values.begin(), values.end(), staging->values.begin());
  auto a = s.reserve(layout);
  auto op = s.begin({a}, {}, waits, staging);
  try {
    op->event = s.driver->make_event();
    a->buffer = s.driver->allocate(values.size());
    require(bool(a->buffer) && bool(op->event), "driver returned null resource");
    s.driver->upload(*a->buffer, staging->values); ++s.uploads;
    s.finish(op);
    return s.output(a, op);
  } catch (...) { s.fail(std::current_exception()); throw; }
}
Polynomial Session::transform(const Polynomial &input, bool inverse, std::span<const Completion> waits) {
  require(bool(state_), "moved session"); auto &s = *state_; s.open(); s.check(input);
  auto a = s.reserve(transform_layout(input.layout_, inverse));
  auto op = s.begin({input.allocation_, a}, {input.producer_}, waits);
  try {
    op->event = s.driver->make_event();
    a->buffer = s.driver->allocate(bytes(a->record.layout) / sizeof(uint64_t));
    require(bool(a->buffer) && bool(op->event), "driver returned null resource");
    s.driver->copy(*a->buffer, *input.allocation_->buffer, s.offset(input), bytes(input.layout_) / sizeof(uint64_t)); ++s.copies;
    s.driver->transform(*a->buffer, a->record.layout, inverse);
    s.finish(op);
    return s.output(a, op);
  } catch (...) { s.fail(std::current_exception()); throw; }
}
Polynomial Session::forward(const Polynomial &p, std::span<const Completion> waits) { return transform(p, false, waits); }
Polynomial Session::inverse(const Polynomial &p, std::span<const Completion> waits) { return transform(p, true, waits); }
Polynomial Session::pointwise(const Polynomial &left, const Polynomial &right, std::span<const Completion> waits) {
  require(bool(state_), "moved session"); auto &s = *state_; s.open(); s.check(left); s.check(right);
  auto a = s.reserve(pointwise_layout(left.layout_, right.layout_));
  auto op = s.begin({left.allocation_, right.allocation_, a}, {left.producer_, right.producer_}, waits);
  try {
    op->event = s.driver->make_event();
    a->buffer = s.driver->allocate(bytes(a->record.layout) / sizeof(uint64_t));
    require(bool(a->buffer) && bool(op->event), "driver returned null resource");
    s.driver->copy(*a->buffer, *left.allocation_->buffer, s.offset(left), bytes(left.layout_) / sizeof(uint64_t)); ++s.copies;
    s.driver->pointwise(*a->buffer, *right.allocation_->buffer, s.offset(right), a->record.layout);
    s.finish(op);
    return s.output(a, op);
  } catch (...) { s.fail(std::current_exception()); throw; }
}
Polynomial Session::subview(const Polynomial &p, size_t first, size_t count, size_t active_q) {
  require(bool(state_), "moved session"); auto &s = *state_; s.open(); s.check(p);
  require(first <= p.view_.tower_count && count <= p.view_.tower_count - first,
          "subview exceeds parent extent");
  require(active_q <= p.view_.active_q, "subview cannot add Q towers");
  View v{p.view_.allocation, p.view_.first_tower + first, count, active_q};
  auto layout = validate_view(v, p.allocation_->record, s.device);
  return Polynomial(state_, p.allocation_, p.producer_, std::move(layout), v);
}
Download Session::download(const Polynomial &p, std::span<const Completion> waits) {
  require(bool(state_), "moved session"); auto &s = *state_; s.open(); s.check(p);
  auto staging = std::make_shared<detail::HostStorage>(s.budget, bytes(p.layout_) / sizeof(uint64_t));
  auto op = s.begin({p.allocation_}, {p.producer_}, waits, staging);
  try {
    op->event = s.driver->make_event();
    require(bool(op->event), "driver returned null event");
    s.driver->download(staging->values, *p.allocation_->buffer, s.offset(p)); ++s.downloads;
    s.finish(op);
    return Download(state_, op, staging, p.layout_);
  } catch (...) { s.fail(std::current_exception()); throw; }
}
Download Session::multiply(std::span<const uint64_t> a, std::span<const uint64_t> b) {
  // Validate both external arrays before the first enqueue.
  require(bool(state_), "moved session"); state_->open();
  validate_host_values(state_->base, a); validate_host_values(state_->base, b);
  auto da = upload_checked(state_->base, a, {}), db = upload_checked(state_->base, b, {});
  auto fa = forward(da), fb = forward(db);
  auto product = pointwise(fa, fb);
  return download(inverse(product));
}
void Session::close() {
  require(bool(state_), "moved session"); auto &s = *state_;
  s.closed = true;
  if (!s.quiescent && !s.drain()) s.fail(s.drain_error);
  for (auto &[id, weak] : s.registry) {
    (void)id;
    if (auto a = weak.lock()) a->record.live = false;
  }
  try { s.driver->check_health(); }
  catch (...) { s.fail(std::current_exception()); }
  if (s.error) std::rethrow_exception(s.error);
}
} // namespace ntt::resident
