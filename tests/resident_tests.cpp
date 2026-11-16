#include "support/resident_host_driver.hpp"
#include <iostream>
#include <type_traits>

using namespace ntt::resident;
using namespace ntt::proposal;
using resident_test::HostDriver;
namespace {
size_t checks = 0;
void check(bool ok, const char *why) { ++checks; if (!ok) throw std::runtime_error(why); }
template<class F> void rejects(F f, const char *why) {
  ++checks; try { f(); } catch (const std::exception &) { return; }
  throw std::runtime_error(why);
}
PolynomialLayout layout(size_t q_count = 2, size_t p_count = 0) {
  auto primes = ntt::find_primes(8, q_count + p_count, 20);
  std::vector<Tower> q, p;
  for (size_t i = 0; i < primes.size(); ++i)
    (i < q_count ? q : p).push_back({primes[i], ntt::find_psi(primes[i], 8)});
  auto params = Parameters::create(8, q, p, 1);
  return {params, q_count, p_count ? qp_basis(*params, q_count) : q_prefix(*params, q_count),
          Domain::Coefficient, Order::Natural};
}
std::vector<uint64_t> input(const PolynomialLayout &p, size_t seed) {
  std::vector<uint64_t> out(bytes(p) / sizeof(uint64_t));
  for (size_t k = 0; k < p.towers.size(); ++k)
    for (size_t i = 0; i < p.parameters->n(); ++i)
      out[k * p.parameters->n() + i] = (seed + i * 17 + k * 31) % p.parameters->tower(p.towers[k]).q;
  return out;
}
std::vector<uint64_t> naive(const PolynomialLayout &p, std::span<const uint64_t> a,
                             std::span<const uint64_t> b) {
  const auto n = p.parameters->n(); std::vector<uint64_t> out(a.size());
  for (size_t k = 0; k < p.towers.size(); ++k) {
    const auto q = p.parameters->tower(p.towers[k]).q;
    for (size_t i = 0; i < n; ++i) for (size_t j = 0; j < n; ++j) {
      const auto value = ntt::mul_mod(a[k*n+i], b[k*n+j], q);
      auto &target = out[k*n+(i+j)%n];
      target = i+j < n ? ntt::add_mod(target, value, q) : ntt::sub_mod(target, value, q);
    }
  }
  return out;
}
void equal(std::span<const uint64_t> actual, const std::vector<uint64_t> &expected, const char *why) {
  check(actual.size() == expected.size() && std::equal(actual.begin(), actual.end(), expected.begin()), why);
}
void chain_and_leases() {
  static_assert(!std::is_aggregate_v<Polynomial> && !std::is_aggregate_v<Completion>);
  auto p = layout(); auto driver = std::make_shared<HostDriver>(p); Session s(driver);
  rejects([&] { Session duplicate(driver); }, "a transport cannot be shared across sessions");
  auto a = input(p, 5), b = input(p, 9), original = a;
  auto da = s.upload(a), db = s.upload(b);
  std::fill(a.begin(), a.end(), 0); // Caller memory is no longer borrowed.
  check(da.readiness().query().outcome == Outcome::Pending, "upload must remain pending in deferred transport");
  check(driver->trace->executed == 0, "submission must not synchronously execute host fake queue");
  auto fa = s.forward(da), fb = s.forward(db);
  auto product = s.pointwise(fa, fb); auto coefficient = s.inverse(product);
  auto result = s.download(coefficient);
  check(s.stats().submitted == 7 && s.stats().pending == 7, "full pipeline operation journal");
  check(s.stats().uploads == 2 && s.stats().downloads == 1 && s.stats().device_copies == 4,
        "exactly two uploads, four explicit D2D copies and one final download");
  check(da.view().allocation != fa.view().allocation && fa.view().allocation != product.view().allocation,
        "out-of-place operations issue fresh allocations");
  equal(result.wait_values(), naive(p, original, b), "resident chain equals independent negacyclic convolution");
  check(s.stats().pending == 0 && s.stats().quiescent, "completion retires all prior work");
  equal(s.download(da).wait_values(), original, "forward must not modify uploaded input");
  equal(s.download(db).wait_values(), b, "right input must remain immutable");
  auto source_completion = da.readiness();
  auto expected_eval = s.download(fa);
  auto expected_values = expected_eval.wait_values();
  auto with_dependency = s.download(s.forward(da, std::span<const Completion>(&source_completion, 1)));
  equal(with_dependency.wait_values(), {expected_values.begin(), expected_values.end()}, "explicit same-stream dependency");
  expected_eval = {}; with_dependency = {};
  auto again = s.multiply(original, b);
  equal(again.wait_values(), naive(p, original, b), "convenience chain");
  auto square = s.download(s.inverse(s.pointwise(fa, fa)));
  equal(square.wait_values(), naive(p, original, original), "read-only input alias is safe with fresh output");
  auto copied_view = da.view(); ++copied_view.allocation.generation;
  check(da.view().allocation != copied_view.allocation, "descriptor copies cannot alter trusted registry");
  da = {}; db = {}; fa = {}; fb = {}; product = {}; coefficient = {};
  result = {}; again = {}; square = {};
  check(driver->trace->live_buffers == 0 && driver->trace->live_events == 0, "completed pins do not retain allocations/events");
  check(s.stats().device_bytes == 0 && s.stats().host_bytes == 0, "budget leases retire with resources");
}
void rejection_and_views() {
  auto p = layout(3, 2); auto driver = std::make_shared<HostDriver>(p); Session s(driver);
  auto a = input(p, 3), b = a; b[0] = p.parameters->tower(p.towers[0]).q;
  rejects([&] { s.multiply(a, b); }, "bad second input must reject entire chain before enqueue");
  check(driver->trace->calls == 0, "invalid external inputs submit nothing");
  rejects([&] { s.upload(std::span<const uint64_t>(a).first(1)); }, "short input");
  auto da = s.upload(a); const auto calls = driver->trace->calls;
  rejects([&] { s.inverse(da); }, "inverse domain rejection");
  rejects([&] { s.pointwise(da, da); }, "pointwise domain rejection");
  rejects([&] { s.forward(Polynomial{}); }, "empty polynomial rejection");
  auto second_driver = std::make_shared<HostDriver>(p); Session other(second_driver);
  rejects([&] { other.forward(da); }, "same-ordinal foreign session rejected");
  auto completion = da.readiness();
  rejects([&] { other.upload(a, std::span<const Completion>(&completion, 1)); }, "foreign explicit dependency rejected");
  rejects([&] { s.subview(da, 0, 4, 2); }, "reduced QP cannot be fabricated as contiguous prefix");
  rejects([&] { s.subview(da, 4, 2, 3); }, "subview beyond extent");
  check(driver->trace->calls == calls && second_driver->trace->calls == 0, "preflight failures call no transport");
  auto view = s.subview(da, 1, 2, 3);
  auto nested = s.subview(view, 1, 1, 3);
  auto result = s.download(s.inverse(s.forward(nested)));
  equal(result.wait_values(), {a.begin() + 16, a.begin() + 24}, "nested tower view uses allocation-relative offset");
  check(nested.view().first_tower == 2, "nested view offset");
  auto e = s.forward(view), e2 = s.forward(nested);
  rejects([&] { s.pointwise(e, e2); }, "incompatible tower layouts");
  s.close();
  rejects([&] { s.forward(da); }, "stale handles after close");
  rejects([&] { s.upload(a); }, "closed session refuses admission");
  check(da.readiness().query().outcome == Outcome::Succeeded, "completion metadata survives close");
  rejects([] { Completion{}.wait(); }, "empty completion");
  rejects([] { Download{}.wait_values(); }, "empty host result");
}
void budgets() {
  auto p = layout(); auto a = input(p, 4);
  auto driver = std::make_shared<HostDriver>(p);
  Session s(driver, {bytes(p), bytes(p), 1});
  auto da = s.upload(a); const auto calls = driver->trace->calls;
  rejects([&] { s.forward(da); }, "device budget exhausted before native admission");
  rejects([&] { s.download(da); }, "staging budget exhausted before native admission");
  check(driver->trace->calls == calls && !s.stats().poisoned, "budget refusal is preflight");
  da.readiness().wait();
  rejects([&] { s.download(da); }, "operation journal budget remains bounded");
  check(driver->trace->calls == calls + 1, "operation budget submits nothing");
}
void failure_matrix() {
  auto p = layout(); auto a = input(p, 1), b = input(p, 2);
  size_t steps;
  {
    auto d = std::make_shared<HostDriver>(p); Session s(d);
    auto result = s.multiply(a, b); steps = d->trace->calls; result.wait_values();
  }
  for (size_t step = 1; step <= steps + 1; ++step) {
    auto d = std::make_shared<HostDriver>(p); auto trace = d->trace;
    {
      Session s(d); trace->fail_step = step;
      rejects([&] { auto result = s.multiply(a, b); result.wait_values(); }, "injected synchronous failure must propagate");
      check(s.stats().poisoned && s.stats().quiescent, "failure drains and poisons");
      check(!s.history().empty() && s.history().back().outcome == Outcome::Failed, "orphaned failed operation remains in history");
      rejects([&] { s.upload(a); }, "no fallback or resubmission after transport failure");
      rejects([&] { s.close(); }, "close propagates retained failure");
    }
    check(trace->live_buffers == 0 && trace->live_events == 0, "failure unwinds all quiescent buffers/events");
  }
  for (size_t sequence : {size_t{1}, size_t{4}, size_t{11}}) {
    auto d = std::make_shared<HostDriver>(p); Session s(d); d->trace->async_fail_at = sequence;
    auto result = s.multiply(a, b);
    rejects([&] { result.wait_values(); }, "asynchronous failure cannot expose host output");
    check(s.stats().poisoned, "asynchronous failure poisons session");
    check(result.completion().query().outcome != Outcome::Succeeded, "failed output not successful");
  }
}
void partial_and_quarantine() {
  auto p = layout(); auto a = input(p, 1), b = input(p, 2);
  std::vector<size_t> enqueue_calls;
  { auto d = std::make_shared<HostDriver>(p); Session s(d); auto r = s.multiply(a,b);
    enqueue_calls = d->trace->enqueue_calls; r.wait_values(); }
  for (auto step : enqueue_calls) {
    auto d = std::make_shared<HostDriver>(p); Session s(d); d->trace->partial_step = step;
    rejects([&] { s.multiply(a,b); }, "failure after partial submission");
    check(d->trace->executed > 0 && d->trace->live_buffers == 0 && d->trace->live_events == 0,
          "partial submission pins every resource until drain");
    check(s.history().back().outcome == Outcome::Failed, "partial failure retains its journal record");
  }
  auto d = std::make_shared<HostDriver>(p); auto trace = d->trace;
  {
    Session s(d); auto result = s.multiply(a, b);
    trace->async_fail_at = 3; trace->allow_drain = false;
    rejects([&] { result.wait_values(); }, "failed wait with unproven quiescence");
    result = {};
    check(s.stats().poisoned && !s.stats().quiescent && trace->live_buffers == 6,
          "abandoned external handles must not free quarantined device buffers");
    check(s.stats().host_bytes == 3 * bytes(p), "quarantine retains upload and download staging");
    rejects([&] { s.close(); }, "failed close retains pins");
    check(trace->live_buffers == 6, "unproven drain cannot release");
    trace->allow_drain = true;
    rejects([&] { s.close(); }, "recovered cleanup still reports original failure");
    check(s.stats().quiescent && trace->live_buffers == 0 && trace->live_events == 0, "proven recovery reclaims quarantine");
  }
  check(trace->live_buffers == 0 && trace->live_events == 0, "no leaked recoverable quarantine");
}
void abandonment_and_bad_output() {
  auto p = layout(); auto a = input(p, 1), b = input(p, 2);
  auto d = std::make_shared<HostDriver>(p); Download output;
  { Session s(d); output = s.multiply(a, b); }
  equal(output.wait_values(), naive(p, a, b), "close with live output drains before destruction");
  output = {};
  check(d->trace->live_buffers == 0 && d->trace->live_events == 0, "abandonment cleanup");
  auto corrupt = std::make_shared<HostDriver>(p); Session s(corrupt);
  corrupt->trace->corrupt_download = true;
  auto result = s.multiply(a, b);
  rejects([&] { result.wait_values(); }, "noncanonical driver output must stay hidden");
  check(s.stats().poisoned && s.history().back().outcome == Outcome::Failed, "bad output fails its completion");
}
void query_failure_and_evaluation_input() {
  auto p = layout(); auto a = input(p, 8);
  auto d = std::make_shared<HostDriver>(p); Session s(d);
  auto eval = s.upload_evaluation(a);
  check(eval.layout().domain == Domain::Evaluation && eval.layout().order == Order::BitReversed,
        "evaluation upload declares exact representation");
  auto result = s.download(s.inverse(eval));
  auto expected = a;
  for (size_t k = 0; k < p.towers.size(); ++k) {
    const auto t = p.parameters->tower(p.towers[k]);
    std::vector<uint64_t> plane(a.begin() + k * 8, a.begin() + (k + 1) * 8);
    ntt::NegacyclicNtt(8, t.q, t.psi).inverse(plane);
    std::copy(plane.begin(), plane.end(), expected.begin() + k * 8);
  }
  equal(result.wait_values(), expected, "independent evaluation input inverse");
  auto other = std::make_shared<HostDriver>(p); Session failed(other);
  auto value = failed.upload(a); other->trace->fail_step = other->trace->calls + 1;
  check(value.readiness().query().outcome == Outcome::Failed, "query reports driver failure as failed outcome");
  rejects([&] { value.readiness().wait(); }, "wait rethrows failed-query error");
  check(failed.stats().poisoned, "query errors poison admission");
  auto cleanup = std::make_shared<HostDriver>(p); Session close_error(cleanup);
  cleanup->trace->cleanup_error = true;
  rejects([&] { close_error.close(); }, "close reports deferred cleanup errors");
  check(close_error.stats().poisoned && close_error.stats().closed, "cleanup error stops admission");
}
} // namespace
int main() {
  try {
    chain_and_leases(); rejection_and_views(); budgets(); failure_matrix();
    partial_and_quarantine(); abandonment_and_bad_output(); query_failure_and_evaluation_input();
    std::cout << checks << " RESIDENT HOST CONTROL CHECKS PASSED; NO GPU EXECUTION\n";
  } catch (const std::exception &e) { std::cerr << e.what() << '\n'; return 1; }
}
