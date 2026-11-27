#include "ntt/resident_driver.hpp"
#include "ntt/resident_conversion.hpp"
#include <array>
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
struct DeviceStorage {
  Reservation reservation;
  std::unique_ptr<Buffer> buffer;
  DeviceStorage(std::shared_ptr<Budget> b, size_t count)
      : reservation(std::move(b), checked_product(count, sizeof(uint64_t)), false) {}
};
struct Operation {
  CompletionRecord record;
  std::unique_ptr<Event> event;
  std::vector<std::shared_ptr<Allocation>> allocations;
  std::vector<std::shared_ptr<Operation>> dependencies;
  std::shared_ptr<HostStorage> staging;
  std::shared_ptr<DeviceStorage> constants;
  explicit Operation(CompletionRecord r) : record(r) {}
  void unpin() noexcept {
    event.reset();
    allocations.clear();
    dependencies.clear();
    staging.reset();
    constants.reset();
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
  size_t uploads = 0, downloads = 0, copies = 0, parameter_uploads = 0;
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
  void compatible(const PolynomialLayout &p) const {
    validate(p);
    require(*p.parameters == *base.parameters && p.active_q <= base.active_q,
            "foreign resident conversion parameters");
    const auto it = std::find(base.towers.begin(), base.towers.end(), p.towers.front());
    require(it != base.towers.end() && p.towers.size() <= static_cast<size_t>(base.towers.end() - it)
              && std::equal(p.towers.begin(), p.towers.end(), it),
            "layout is not a contiguous subset of the session basis");
  }
  void conversion_input(const PolynomialLayout &p, bool up) const {
    compatible(p);
    require(base.towers == qp_basis(*base.parameters, base.active_q)
              && p.active_q == base.active_q && p.active_q <= 40
              && base.parameters->p().size() <= 40,
            "conversion requires a session with active Q then 1..40 P towers");
    require(p.towers == (up ? q_prefix(*p.parameters, p.active_q)
                            : qp_basis(*p.parameters, p.active_q)),
            "conversion requires complete active Q or QP in original order");
    require(up || p.domain == Domain::Evaluation, "ModDown requires evaluation input");
  }
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
  // Inputs, staging and output reservation are already pinned by begin().
  // Keep allocation, enqueue, event recording and result construction under
  // the same failure guard, including a throw after a partial submission.
  template <class Enqueue>
  Polynomial submit_output(const std::shared_ptr<Allocation> &a,
                           const std::shared_ptr<Operation> &op, Enqueue enqueue) {
    try {
      op->event = driver->make_event();
      a->buffer = driver->allocate(a->record.capacity_bytes / sizeof(uint64_t));
      require(bool(a->buffer) && bool(op->event), "driver returned null resource");
      enqueue(*a->buffer);
      finish(op);
      return output(a, op);
    } catch (...) {
      fail(std::current_exception());
      throw;
    }
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
detail::State &Session::open_state() {
  require(bool(state_), "moved session");
  state_->open();
  return *state_;
}
Device Session::device() const { require(bool(state_), "moved session"); return state_->device; }
Stats Session::stats() const {
  require(bool(state_), "moved session");
  size_t pending = 0;
  for (const auto &op : state_->operations) pending += op->record.outcome == Outcome::Pending;
  return {state_->operations.size(), pending, state_->budget->device, state_->budget->host,
          state_->uploads, state_->downloads, state_->copies, state_->parameter_uploads,
          state_->closed, bool(state_->error), state_->quiescent};
}
std::vector<CompletionRecord> Session::history() const {
  require(bool(state_), "moved session");
  std::vector<CompletionRecord> out;
  for (const auto &op : state_->operations) out.push_back(op->record);
  return out;
}
Polynomial Session::upload(std::span<const uint64_t> values, std::span<const Completion> waits) {
  auto &s = open_state();
  validate_host_values(s.base, values);
  return upload_checked(s.base, values, waits);
}
Polynomial Session::upload_evaluation(std::span<const uint64_t> values, std::span<const Completion> waits) {
  auto &s = open_state();
  auto layout = transform_layout(s.base, false);
  validate_host_values(layout, values);
  return upload_checked(layout, values, waits);
}
Polynomial Session::upload(const PolynomialLayout &p, std::span<const uint64_t> values,
                            std::span<const Completion> waits) {
  auto &s = open_state();
  s.compatible(p);
  validate_host_values(p, values);
  return upload_checked(p, values, waits);
}
Polynomial Session::upload_checked(const PolynomialLayout &layout, std::span<const uint64_t> values,
                                    std::span<const Completion> waits) {
  auto &s = *state_;
  auto staging = std::make_shared<detail::HostStorage>(s.budget, values.size());
  std::copy(values.begin(), values.end(), staging->values.begin());
  auto a = s.reserve(layout);
  auto op = s.begin({a}, {}, waits, staging);
  return s.submit_output(a, op, [&](detail::Buffer &output) {
    s.driver->upload(output, staging->values);
    ++s.uploads;
  });
}
Polynomial Session::transform(const Polynomial &input, bool inverse, std::span<const Completion> waits) {
  auto &s = open_state();
  s.check(input);
  auto a = s.reserve(transform_layout(input.layout_, inverse));
  auto op = s.begin({input.allocation_, a}, {input.producer_}, waits);
  return s.submit_output(a, op, [&](detail::Buffer &output) {
    s.driver->copy(output, *input.allocation_->buffer, s.offset(input), bytes(input.layout_) / sizeof(uint64_t));
    ++s.copies;
    s.driver->transform(output, a->record.layout, inverse);
  });
}
Polynomial Session::forward(const Polynomial &p, std::span<const Completion> waits) { return transform(p, false, waits); }
Polynomial Session::inverse(const Polynomial &p, std::span<const Completion> waits) { return transform(p, true, waits); }
Polynomial Session::pointwise(const Polynomial &left, const Polynomial &right, std::span<const Completion> waits) {
  auto &s = open_state();
  s.check(left); s.check(right);
  auto a = s.reserve(pointwise_layout(left.layout_, right.layout_));
  auto op = s.begin({left.allocation_, right.allocation_, a}, {left.producer_, right.producer_}, waits);
  return s.submit_output(a, op, [&](detail::Buffer &output) {
    s.driver->copy(output, *left.allocation_->buffer, s.offset(left), bytes(left.layout_) / sizeof(uint64_t));
    ++s.copies;
    s.driver->pointwise(output, *right.allocation_->buffer, s.offset(right), a->record.layout);
  });
}
Polynomial Session::switch_basis(const Polynomial &input, const PolynomialLayout &target,
                                 std::span<const Completion> waits) {
  auto &s = open_state();
  s.check(input);
  s.compatible(input.layout_); s.compatible(target);
  require(input.layout_.domain == Domain::Coefficient && target.domain == Domain::Coefficient
            && input.layout_.active_q == target.active_q,
          "basis switching requires coefficient layouts at the same active level");
  std::vector<uint64_t> from, to;
  for (auto t : input.layout_.towers) from.push_back(target.parameters->tower(t).q);
  for (auto t : target.towers) to.push_back(target.parameters->tower(t).q);
  const auto packed = detail::pack_basis(ApproximateBasisPlan(std::move(from), std::move(to)));
  auto staging = std::make_shared<detail::HostStorage>(s.budget, packed.size());
  std::copy(packed.begin(), packed.end(), staging->values.begin());
  auto constants = std::make_shared<detail::DeviceStorage>(s.budget, packed.size());
  auto a = s.reserve(target);
  auto op = s.begin({input.allocation_, a}, {input.producer_}, waits, staging);
  op->constants = constants; // Retained before the first possible enqueue.
  try {
    op->event = s.driver->make_event();
    a->buffer = s.driver->allocate(bytes(target) / sizeof(uint64_t));
    constants->buffer = s.driver->allocate(packed.size());
    require(bool(a->buffer) && bool(constants->buffer) && bool(op->event), "driver returned null resource");
    s.driver->upload(*constants->buffer, staging->values); ++s.parameter_uploads;
    s.driver->switch_basis(*a->buffer, *input.allocation_->buffer, s.offset(input),
                            *constants->buffer, input.layout_, target);
    s.finish(op);
    return s.output(a, op);
  } catch (...) { s.fail(std::current_exception()); throw; }
}
Polynomial Session::join(std::span<const Polynomial> parts) {
  auto &s = open_state();
  require(!parts.empty(), "empty join");
  auto layout = parts.front().layout_;
  layout.towers.clear();
  std::vector<std::shared_ptr<detail::Allocation>> allocations;
  std::vector<std::shared_ptr<detail::Operation>> producers;
  for (const auto &part : parts) {
    s.check(part);
    require(*layout.parameters == *part.layout_.parameters
                && layout.active_q == part.layout_.active_q
                && layout.domain == part.layout_.domain
                && layout.order == part.layout_.order,
            "incompatible join layouts");
    layout.towers.insert(layout.towers.end(), part.layout_.towers.begin(), part.layout_.towers.end());
    allocations.push_back(part.allocation_);
    producers.push_back(part.producer_);
  }
  s.compatible(layout);
  auto a = s.reserve(layout);
  allocations.push_back(a);
  auto op = s.begin(std::move(allocations), std::move(producers), {});
  return s.submit_output(a, op, [&](detail::Buffer &output) {
    size_t destination = 0;
    for (const auto &part : parts) {
      const auto count = bytes(part.layout_) / sizeof(uint64_t);
      s.driver->copy_range(output, destination, *part.allocation_->buffer, s.offset(part), count);
      destination += count;
      ++s.copies;
    }
  });
}
Polynomial Session::correct(const Polynomial &left, const Polynomial &right) {
  auto &s = open_state();
  s.check(left); s.check(right);
  const auto layout = pointwise_layout(left.layout_, right.layout_);
  std::vector<uint64_t> packed;
  for (auto t : layout.towers) packed.push_back(layout.parameters->tower(t).q);
  const auto moduli = packed;
  for (auto q : moduli)
    packed.push_back(inv_mod(detail::special_prime_product(*layout.parameters, q), q));
  auto staging = std::make_shared<detail::HostStorage>(s.budget, packed.size());
  std::copy(packed.begin(), packed.end(), staging->values.begin());
  auto constants = std::make_shared<detail::DeviceStorage>(s.budget, packed.size());
  auto a = s.reserve(layout);
  auto op = s.begin({left.allocation_, right.allocation_, a}, {left.producer_, right.producer_}, {}, staging);
  op->constants = constants;
  try {
    op->event = s.driver->make_event();
    a->buffer = s.driver->allocate(bytes(layout) / sizeof(uint64_t));
    constants->buffer = s.driver->allocate(packed.size());
    require(bool(a->buffer) && bool(constants->buffer) && bool(op->event), "driver returned null resource");
    s.driver->upload(*constants->buffer, staging->values); ++s.parameter_uploads;
    s.driver->correct(*a->buffer, *left.allocation_->buffer, s.offset(left),
                      *right.allocation_->buffer, s.offset(right), *constants->buffer, layout);
    s.finish(op);
    return s.output(a, op);
  } catch (...) { s.fail(std::current_exception()); throw; }
}
Polynomial Session::mod_up(const Polynomial &input, std::span<const Completion> waits) {
  auto &s = open_state();
  s.check(input);
  s.conversion_input(input.layout_, true);
  auto p_layout = s.base;
  p_layout.towers.erase(p_layout.towers.begin(), p_layout.towers.begin() + s.base.active_q);
  Polynomial coefficients, evaluations;
  if (input.layout_.domain == Domain::Coefficient) {
    coefficients = input;
    evaluations = forward(input, waits);
  } else {
    evaluations = input;
    coefficients = inverse(input, waits);
  }
  auto p_coefficients = switch_basis(coefficients, p_layout);
  return join(std::array{evaluations, forward(p_coefficients)});
}
Polynomial Session::mod_down(const Polynomial &input, std::span<const Completion> waits) {
  auto &s = open_state();
  s.check(input);
  s.conversion_input(input.layout_, false);
  auto q = subview(input, 0, input.layout_.active_q, input.layout_.active_q);
  auto p = subview(input, input.layout_.active_q, input.layout_.parameters->p().size(), input.layout_.active_q);
  const auto q_coefficient = transform_layout(q.layout_, true);
  auto correction = forward(switch_basis(inverse(p, waits), q_coefficient));
  return correct(q, correction);
}
std::vector<Polynomial> Session::hybrid_digits(const Polynomial &input,
                                                std::span<const Completion> waits) {
  auto &s = open_state();
  s.check(input);
  s.conversion_input(input.layout_, true);
  require(input.layout_.domain == Domain::Evaluation, "Hybrid digits require evaluation Q input");

  const size_t active_q = input.layout_.active_q;
  const size_t alpha = s.base.parameters->alpha();
  std::vector<Polynomial> digits;
  digits.reserve(1 + (active_q - 1) / alpha);
  for (size_t first = 0; first < active_q; first += alpha) {
    const size_t count = std::min(alpha, active_q - first);
    const auto evaluations = subview(input, first, count, active_q);
    const auto coefficients = inverse(evaluations, waits);
    std::vector<Polynomial> parts;
    auto append_complement = [&](size_t begin, size_t end) {
      // Each distinct target is independent. Keep a conversion within the
      // shared plan's 40-tower limit and the driver's contiguous layout.
      while (begin < end) {
        const size_t next = begin + std::min<size_t>(40, end - begin);
        auto target = s.base;
        target.towers.assign(s.base.towers.begin() + begin, s.base.towers.begin() + next);
        parts.push_back(forward(switch_basis(coefficients, target)));
        begin = next;
      }
    };
    append_complement(0, first);
    parts.push_back(evaluations);
    append_complement(first + count, s.base.towers.size());
    digits.push_back(join(parts));
  }
  return digits;
}
HybridKey Session::upload_hybrid_key(const PolynomialLayout &full_key_layout,
                                     std::span<const HybridKeyDigit> digits) {
  auto &s = open_state();
  const auto active_layout = transform_layout(s.base, false);
  s.conversion_input(active_layout, false);
  validate(full_key_layout);
  const auto &parameters = *s.base.parameters;
  const auto full_q = parameters.q().size();
  const auto alpha = parameters.alpha();
  require(*full_key_layout.parameters == parameters &&
              full_key_layout.active_q == full_q &&
              full_key_layout.towers == qp_basis(parameters, full_q) &&
              full_key_layout.domain == Domain::Evaluation,
          "Hybrid keys require full-context evaluation QP");
  require(digits.size() == 1 + (full_q - 1) / alpha, "wrong full-context Hybrid key digit count");
  for (const auto &digit : digits) {
    validate_host_values(full_key_layout, digit.b);
    validate_host_values(full_key_layout, digit.a);
  }

  const auto offsets = key_indices(parameters, active_layout.active_q);
  const size_t active_digits = 1 + (active_layout.active_q - 1) / alpha;
  const size_t n = parameters.n();
  std::vector<uint64_t> packed(offsets.size() * n);
  auto upload_channel = [&](std::span<const uint64_t> values) {
    for (size_t tower = 0; tower < offsets.size(); ++tower)
      std::copy_n(values.begin() + offsets[tower] * n, n, packed.begin() + tower * n);
    return upload_checked(active_layout, packed, {});
  };
  HybridKey key;
  key.b_.reserve(active_digits);
  key.a_.reserve(active_digits);
  for (const auto &digit : digits.first(active_digits)) {
    key.b_.push_back(upload_channel(digit.b));
    key.a_.push_back(upload_channel(digit.a));
  }
  return key;
}
HybridContributions Session::hybrid_accumulate(std::span<const Polynomial> digits,
                                               const HybridKey &key,
                                               std::span<const Completion> waits) {
  auto &s = open_state();
  const auto layout = transform_layout(s.base, false);
  s.conversion_input(layout, false);
  const size_t active_digits = 1 + (layout.active_q - 1) / layout.parameters->alpha();
  require(digits.size() == active_digits && key.b_.size() == active_digits &&
              key.a_.size() == active_digits, "wrong active Hybrid digit count");

  std::vector<std::shared_ptr<detail::Allocation>> allocations;
  std::vector<std::shared_ptr<detail::Operation>> producers;
  for (size_t index = 0; index < active_digits; ++index) {
    for (const Polynomial *part : {&digits[index], &key.b_[index], &key.a_[index]}) {
      s.check(*part);
      require_layout(part->layout_, layout);
      require(s.offset(*part) == 0, "Hybrid MAC requires full active QP buffers");
      allocations.push_back(part->allocation_);
      producers.push_back(part->producer_);
    }
  }
  auto b = s.reserve(layout), a = s.reserve(layout);
  allocations.insert(allocations.end(), {b, a});
  auto op = s.begin(std::move(allocations), std::move(producers), waits);
  try {
    op->event = s.driver->make_event();
    const size_t elements = bytes(layout) / sizeof(uint64_t);
    b->buffer = s.driver->allocate(elements);
    a->buffer = s.driver->allocate(elements);
    require(bool(b->buffer) && bool(a->buffer) && bool(op->event), "driver returned null resource");
    for (size_t index = 0; index < active_digits; ++index)
      s.driver->hybrid_mac(*b->buffer, *a->buffer, *digits[index].allocation_->buffer,
                          *key.b_[index].allocation_->buffer, *key.a_[index].allocation_->buffer,
                          layout, index == 0);
    s.finish(op);
    return {s.output(b, op), s.output(a, op)};
  } catch (...) {
    s.fail(std::current_exception());
    throw;
  }
}
HybridContributions Session::hybrid_rotate_extended(const HybridContributions &extended,
                                                    const Polynomial &c0,
                                                    uint32_t automorphism) {
  auto &s = open_state();
  s.check(extended.b);
  s.check(extended.a);
  s.check(c0);
  const auto &layout = extended.b.layout_;
  s.conversion_input(layout, false);
  require_layout(extended.a.layout_, layout);
  s.conversion_input(c0.layout_, true);
  require(c0.layout_.domain == Domain::Evaluation && s.offset(extended.b) == 0 &&
              s.offset(extended.a) == 0,
          "Hybrid rotation requires evaluation Q and full evaluation QP buffers");
  require((automorphism & 1) && automorphism < 2 * layout.parameters->n(),
          "invalid negacyclic automorphism index");

  auto staging = std::make_shared<detail::HostStorage>(s.budget, layout.active_q);
  for (size_t tower = 0; tower < layout.active_q; ++tower) {
    const auto modulus = layout.parameters->q()[tower].q;
    staging->values[tower] = detail::special_prime_product(*layout.parameters, modulus);
  }
  auto constants = std::make_shared<detail::DeviceStorage>(s.budget, layout.active_q);
  auto b = s.reserve(layout), a = s.reserve(layout);
  auto op = s.begin({extended.b.allocation_, extended.a.allocation_, c0.allocation_, b, a},
                    {extended.b.producer_, extended.a.producer_, c0.producer_}, {}, staging);
  op->constants = constants;
  try {
    op->event = s.driver->make_event();
    const size_t elements = bytes(layout) / sizeof(uint64_t);
    b->buffer = s.driver->allocate(elements);
    a->buffer = s.driver->allocate(elements);
    constants->buffer = s.driver->allocate(layout.active_q);
    require(bool(b->buffer) && bool(a->buffer) && bool(constants->buffer) && bool(op->event),
            "driver returned null resource");
    s.driver->upload(*constants->buffer, staging->values);
    ++s.parameter_uploads;
    s.driver->hybrid_rotate(*b->buffer, *a->buffer, *extended.b.allocation_->buffer,
                            *extended.a.allocation_->buffer, *c0.allocation_->buffer, s.offset(c0),
                            *constants->buffer, layout, automorphism);
    s.finish(op);
    return {s.output(b, op), s.output(a, op)};
  } catch (...) {
    s.fail(std::current_exception());
    throw;
  }
}
Polynomial Session::subview(const Polynomial &p, size_t first, size_t count, size_t active_q) {
  auto &s = open_state();
  s.check(p);
  require(first <= p.view_.tower_count && count <= p.view_.tower_count - first,
          "subview exceeds parent extent");
  require(active_q <= p.view_.active_q, "subview cannot add Q towers");
  View v{p.view_.allocation, p.view_.first_tower + first, count, active_q};
  auto layout = validate_view(v, p.allocation_->record, s.device);
  return Polynomial(state_, p.allocation_, p.producer_, std::move(layout), v);
}
Download Session::download(const Polynomial &p, std::span<const Completion> waits) {
  auto &s = open_state();
  s.check(p);
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
  auto &s = open_state();
  validate_host_values(s.base, a); validate_host_values(s.base, b);
  auto da = upload_checked(s.base, a, {}), db = upload_checked(s.base, b, {});
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
