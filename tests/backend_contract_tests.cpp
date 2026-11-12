#include "ntt/backend_api.hpp"
#include <iostream>
#include <type_traits>

using namespace ntt::proposal;
namespace {
size_t checks = 0;
void check(bool ok, const char *label) {
  ++checks;
  if (!ok)
    throw std::runtime_error(label);
}
template <class F> void rejects(F f, const char *label) {
  ++checks;
  try {
    f();
  } catch (const ContractError &) {
    return;
  }
  throw std::runtime_error(label);
}
const std::vector<Tower> Q = {{1152921504606584833ULL, 18043022392882ULL},
                              {1125899902124033ULL, 15995550378ULL},
                              {1125899911168001ULL, 21670117260ULL},
                              {1125899903827969ULL, 938640682ULL},
                              {1125899908022273ULL, 6511788946ULL}};
const std::vector<Tower> P = {{1152921504598720513ULL, 800790938143ULL},
                              {1152921504597016577ULL, 17749908910371ULL},
                              {1152921504595968001ULL, 11469071954203ULL}};
void run() {
  static_assert(std::is_abstract_v<Backend> && std::is_abstract_v<Completion>);
  auto q = Q;
  auto p = Parameters::create(65536, q, P, 3);
  q[0].psi = 1;
  check(p->q() == Q, "parameters must own a deep snapshot");
  check(!std::is_copy_assignable_v<Parameters>, "parameters must be immutable");
  rejects([&] { Parameters::create(65535, Q, P, 3); }, "non-power-of-two ring");
  rejects([&] { Parameters::create(65536, Q, P, 0); }, "zero digit width");
  rejects([&] { Parameters::create(65536, Q, P, 6); }, "oversized digit width");
  rejects([&] { Parameters::create(65536, q, P, 3); }, "wrong primitive root");
  auto duplicate = P;
  duplicate[0] = Q[0];
  rejects([&] { Parameters::create(65536, Q, duplicate, 3); }, "duplicate Q/P prime");
  auto composite = Q;
  composite[0] = {65537ULL * 65537ULL, 2};
  rejects([&] { Parameters::create(65536, composite, P, 3); }, "composite modulus");
  rejects([] { checked_product(std::numeric_limits<size_t>::max(), 2); },
          "byte overflow");
  PolynomialLayout coeff{p, 5, q_prefix(*p, 5), Domain::Coefficient, Order::Natural};
  auto eval = coeff;
  eval.domain = Domain::Evaluation;
  eval.order = Order::BitReversed;
  validate(coeff);
  validate(eval);
  require_layout(transform_layout(coeff, false), eval);
  require_layout(transform_layout(eval, true), coeff);
  require_layout(pointwise_layout(eval, eval), eval);
  rejects([&] { transform_layout(eval, false); }, "forward requires coefficient input");
  rejects([&] { transform_layout(coeff, true); }, "inverse requires evaluation input");
  rejects([&] { pointwise_layout(coeff, coeff); },
          "pointwise requires evaluation input");
  check(bytes(coeff) == 5 * 65536 * 8, "tower-major extent");
  check(initial_ntt_shape_supported(coeff), "initial 50-60 bit capability");
  const auto small_q = ::ntt::find_primes(65536, 1, 30).at(0);
  const auto small =
      Parameters::create(65536, {{small_q, ::ntt::find_psi(small_q, 65536)}}, {}, 1);
  check(!initial_ntt_shape_supported(
            {small, 1, q_prefix(*small, 1), Domain::Coefficient, Order::Natural}),
        "valid parameters can be outside initial device capability");
  auto wrong = coeff;
  wrong.order = Order::BitReversed;
  rejects([&] { validate(wrong); }, "coefficient order");
  wrong = eval;
  wrong.order = Order::Natural;
  rejects([&] { validate(wrong); }, "evaluation order");
  wrong = coeff;
  wrong.encoding = Encoding::LazyU64;
  rejects([&] { validate(wrong); }, "lazy residues");
  wrong.encoding = Encoding::MontgomeryU64;
  rejects([&] { validate(wrong); }, "Montgomery residues");
  wrong = coeff;
  wrong.towers[1] = wrong.towers[0];
  rejects([&] { validate(wrong); }, "duplicate tower");
  wrong = coeff;
  wrong.active_q = 4;
  rejects([&] { validate(wrong); }, "inactive Q tower");
  std::vector<uint64_t> host(bytes(coeff) / 8, 0);
  host[65536] = Q[1].q - 1;
  validate_host_values(coeff, host);
  host[65536] = Q[1].q;
  rejects([&] { validate_host_values(coeff, host); }, "noncanonical value");
  rejects(
      [&] { validate_host_values(coeff, std::span<const uint64_t>(host).first(1)); },
      "short host span");

  Device device{0, 42};
  Token token{7, 1};
  AllocationRecord allocation{token, device, bytes(eval), eval};
  View prefix{token, 0, 4, 4};
  const auto prefix_layout = validate_view(prefix, allocation, device);
  check(prefix_layout.towers == q_prefix(*p, 4), "active Q prefix");
  require_layout(prefix_layout,
                 {p, 4, q_prefix(*p, 4), Domain::Evaluation, Order::BitReversed});
  auto qp = eval;
  qp.towers = qp_basis(*p, 5);
  AllocationRecord full{token, device, bytes(qp), qp};
  rejects([&] { validate_view({token, 0, 7, 4}, full, device); },
          "shortened QP is not QlP");
  auto reduced = eval;
  reduced.active_q = 4;
  reduced.towers = qp_basis(*p, 4);
  AllocationRecord packed{{8, 1}, device, bytes(reduced), reduced};
  validate_view({packed.token, 0, 7, 4}, packed, device);
  check(key_indices(*p, 4) == std::vector<size_t>({0, 1, 2, 3, 5, 6, 7}),
        "P key offset at reduced level");
  check(byte_offset({token, 5, 3, 5}, full) == 5 * 65536 * 8, "P suffix byte offset");
  rejects([&] { validate_view({{7, 2}, 0, 4, 4}, allocation, device); },
          "stale generation");
  rejects([&] { validate_view(prefix, allocation, {0, 43}); }, "wrong session");
  rejects([&] { validate_view(prefix, allocation, {1, 42}); }, "wrong device");
  auto retired = allocation;
  retired.live = false;
  rejects([&] { validate_view(prefix, retired, device); }, "retired allocation");
  auto short_record = allocation;
  --short_record.capacity_bytes;
  rejects([&] { validate_view(prefix, short_record, device); },
          "undersized allocation");
  rejects(
      [&] {
        validate_view({token, std::numeric_limits<size_t>::max(), 4, 4}, allocation,
                      device);
      },
      "view range overflow");
  require_disjoint({token, 0, 2, 5}, allocation, {token, 2, 3, 5}, allocation);
  rejects(
      [&] {
        require_disjoint({token, 0, 3, 5}, allocation, {token, 2, 3, 5}, allocation);
      },
      "overlapping output");
  rejects([&] { require_layout(coeff, eval); }, "format mismatch");

  Stream s1{device, {1, 1}}, s2{device, {2, 1}};
  CompletionRecord producer{{10, 1}, s1, Outcome::Pending};
  validate_dependency(producer, {s1, 11, {}});
  rejects([&] { validate_dependency(producer, {s2, 11, {}}); },
          "missing cross-stream wait");
  validate_dependency(producer, {s2, 11, {producer}});
  auto stale = producer;
  stale.operation.generation = 2;
  rejects([&] { validate_dependency(producer, {s2, 11, {stale}}); },
          "wrong event generation");
  rejects([&] { validate_dependency(producer, {s1, 10, {}}); },
          "future or cyclic dependency");
  auto complete = producer;
  complete.outcome = Outcome::Succeeded;
  validate_dependency(complete, {s2, 11, {}});
  auto failed = producer;
  failed.outcome = Outcome::Failed;
  rejects([&] { validate_dependency(failed, {s1, 11, {}}); }, "failed producer");
  rejects([&] { validate_dependency(complete, {s2, 11, {failed}}); },
          "failed explicit wait");
  auto foreign = producer;
  foreign.stream.device.session = 43;
  rejects([&] { validate_dependency(producer, {s2, 11, {foreign}}); },
          "foreign-session event");
  check(!can_retire(1, 0, Outcome::Succeeded, true), "external lease pin");
  check(!can_retire(0, 1, Outcome::Succeeded, true), "in-flight operation pin");
  check(!can_retire(0, 0, Outcome::Pending, false), "pending upload pin");
  check(!can_retire(0, 0, Outcome::Failed, false), "failure does not prove quiescence");
  check(can_retire(0, 0, Outcome::Succeeded, false), "completed unpinned eviction");
  check(can_retire(0, 0, Outcome::Failed, true), "drained failed allocation");

  auto key = ntt_plan_key(eval, device, 1);
  auto changed = key;
  check(key == ntt_plan_key(coeff, device, 1), "shared forward/inverse plan");
  changed.device.session++;
  check(key != changed, "plan session key");
  changed = key;
  changed.arithmetic_revision++;
  check(key != changed, "plan arithmetic key");
  changed = key;
  changed.tables = TableConvention::OpenFheBitReversed;
  check(key != changed, "plan table convention");
  auto roots = Q;
  roots[0].psi = ::ntt::pow_mod(roots[0].psi, 3, roots[0].q);
  auto other = Parameters::create(65536, roots, P, 3);
  auto root_layout = eval;
  root_layout.parameters = other;
  check(key != ntt_plan_key(root_layout, device, 1), "exact root identity");
  auto reordered = eval;
  std::swap(reordered.towers[0], reordered.towers[1]);
  check(key != ntt_plan_key(reordered, device, 1), "ordered basis key");
  EvaluationKeyCacheKey ek{device, p, {31, 1}, "key-A", KeyKind::Rotation, 5, 1};
  validate(ek);
  auto ek2 = ek;
  ek2.material.generation++;
  check(!(ek == ek2), "immutable key material generation");
  ek2 = ek;
  ek2.parameters = Parameters::create(65536, Q, P, 2);
  check(!(ek == ek2), "digit width in key context");
  ek2 = ek;
  ek2.key_index = 7;
  check(!(ek == ek2), "rotation key identity");
  ek2 = ek;
  ek2.material.id = 0;
  rejects([&] { validate(ek2); }, "invalid key snapshot token");
  ConversionPlanKey cv{device,
                       p,
                       4,
                       1,
                       {{Family::Q, 3}},
                       {{Family::Q, 0},
                        {Family::Q, 1},
                        {Family::Q, 2},
                        {Family::P, 0},
                        {Family::P, 1},
                        {Family::P, 2}},
                       1};
  validate(cv);
  auto cv2 = cv;
  cv2.active_q = 5;
  check(!(cv == cv2), "conversion level key");
  cv2 = cv;
  cv2.rule_revision++;
  check(!(cv == cv2), "conversion arithmetic convention");
  cv2 = cv;
  cv2.digit = 2;
  rejects([&] { validate(cv2); }, "inactive conversion digit");
  cv2 = cv;
  std::swap(cv2.to[0], cv2.to[1]);
  rejects([&] { validate(cv2); }, "reordered complement");
  HoistProvenance hoist{
      p, device, {token, 0, 5, 5}, Domain::Evaluation, Order::BitReversed, 1};
  require_same_hoist(hoist, hoist);
  auto h2 = hoist;
  h2.source.allocation.generation++;
  rejects([&] { require_same_hoist(hoist, h2); }, "hoist source version");
  h2 = hoist;
  h2.parameters = other;
  rejects([&] { require_same_hoist(hoist, h2); }, "hoist exact context");
  h2 = hoist;
  h2.rule_revision = 0;
  rejects([&] { require_same_hoist(h2, h2); }, "invalid hoist cannot match itself");

  for (auto policy :
       {FallbackPolicy::RequireDevice, FallbackPolicy::AllowCpuBeforeSubmit}) {
    check(choose_route(policy, Phase::Preflight, true, false, {true, true, true}) ==
              Route::Device,
          "verified device route");
    check(choose_route(policy, Phase::Preflight, true, true, {false, false, false}) ==
              Route::NoOp,
          "same format no-op");
    rejects(
        [&] {
          choose_route(policy, Phase::Preflight, false, true, {true, true, true});
        },
        "invalid input no-op/fallback");
    rejects(
        [&] {
          choose_route(policy, Phase::Enqueued, true, false, {false, false, false});
        },
        "post-submit fallback forbidden");
  }
  for (auto cap : {Capability{false, true, true}, Capability{true, false, true},
                   Capability{true, true, false}}) {
    check(choose_route(FallbackPolicy::AllowCpuBeforeSubmit, Phase::Preflight, true,
                       false, cap) == Route::Cpu,
          "preflight CPU route");
    rejects(
        [&] {
          choose_route(FallbackPolicy::RequireDevice, Phase::Preflight, true, false,
                       cap);
        },
        "required unavailable device");
  }
}
} // namespace
int main() {
  try {
    run();
    std::cout << "D9 metadata contract: " << checks
              << " checks passed (CPU only; no CUDA runtime)\n";
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
