#pragma once
// D9 host-side metadata prototype. This header performs no CUDA allocation,
// synchronization, pointer inspection, or computation. Runtime provenance must
// come from the future backend's private registry; descriptors are not proofs
// that a physical device allocation or event exists.
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "ntt/param_gen.hpp"

namespace ntt::proposal {
struct ContractError : std::invalid_argument {
  using std::invalid_argument::invalid_argument;
};
inline void require(bool ok, const char *why) {
  if (!ok)
    throw ContractError(why);
}
inline size_t checked_product(size_t a, size_t b) {
  require(a == 0 || b <= std::numeric_limits<size_t>::max() / a, "size overflow");
  return a * b;
}
struct Tower {
  uint64_t q, psi;
  bool operator==(const Tower &) const = default;
};
enum class Family { Q, P };
struct TowerRef {
  Family family;
  size_t index;
  bool operator==(const TowerRef &) const = default;
};

// Immutable deep snapshot. Contexts may describe semantically valid parameters
// outside the first GPU capability; dispatch policy is a separate decision.
class Parameters final {
  uint64_t n_;
  std::vector<Tower> q_, p_;
  size_t alpha_;
  Parameters(uint64_t n, std::vector<Tower> q, std::vector<Tower> p, size_t alpha)
      : n_(n), q_(std::move(q)), p_(std::move(p)), alpha_(alpha) {
    require(n >= 2 && (n & (n - 1)) == 0 &&
                n <= std::numeric_limits<uint32_t>::max() / 2,
            "invalid ring dimension");
    require(!q_.empty() && alpha > 0 && alpha <= q_.size(), "invalid Q/digit geometry");
    std::vector<uint64_t> seen;
    for (const auto *basis : {&q_, &p_})
      for (const auto &t : *basis) {
        require(t.q > n && t.q < (uint64_t{1} << 63) && t.q % (2 * n) == 1,
                "invalid modulus shape");
        require(::ntt::detail::is_prime_mr(t.q), "composite modulus");
        require(t.psi > 0 && t.psi < t.q && ::ntt::pow_mod(t.psi, n, t.q) == t.q - 1,
                "invalid primitive root");
        require(std::find(seen.begin(), seen.end(), t.q) == seen.end(),
                "duplicate Q/P modulus");
        seen.push_back(t.q);
      }
  }

public:
  Parameters &operator=(const Parameters &) = delete;
  static std::shared_ptr<const Parameters> create(uint64_t n, std::vector<Tower> q,
                                                  std::vector<Tower> p, size_t alpha) {
    return std::shared_ptr<const Parameters>(
        new Parameters(n, std::move(q), std::move(p), alpha));
  }
  uint64_t n() const {
    return n_;
  }
  const std::vector<Tower> &q() const {
    return q_;
  }
  const std::vector<Tower> &p() const {
    return p_;
  }
  size_t alpha() const {
    return alpha_;
  }
  bool operator==(const Parameters &other) const {
    return n_ == other.n_ && q_ == other.q_ && p_ == other.p_ && alpha_ == other.alpha_;
  }
  const Tower &tower(TowerRef ref) const {
    require(ref.family == Family::Q || ref.family == Family::P, "invalid basis family");
    const auto &basis = ref.family == Family::Q ? q_ : p_;
    require(ref.index < basis.size(), "tower index out of range");
    return basis[ref.index];
  }
};
using ParameterHandle = std::shared_ptr<const Parameters>;
inline std::vector<TowerRef> q_prefix(const Parameters &p, size_t L) {
  require(L > 0 && L <= p.q().size(), "invalid active Q count");
  std::vector<TowerRef> out;
  for (size_t i = 0; i < L; ++i)
    out.push_back({Family::Q, i});
  return out;
}
inline std::vector<TowerRef> qp_basis(const Parameters &p, size_t L) {
  auto out = q_prefix(p, L);
  require(!p.p().empty(), "QP requires P towers");
  for (size_t i = 0; i < p.p().size(); ++i)
    out.push_back({Family::P, i});
  return out;
}
inline std::vector<size_t> key_indices(const Parameters &p, size_t L) {
  const auto active = qp_basis(p, L), full = qp_basis(p, p.q().size());
  std::vector<size_t> out;
  for (auto ref : active)
    out.push_back(
        static_cast<size_t>(std::find(full.begin(), full.end(), ref) - full.begin()));
  return out;
}

enum class Domain { Coefficient, Evaluation };
enum class Order { Natural, BitReversed };
enum class Encoding { CanonicalU64, MontgomeryU64, LazyU64 };
struct PolynomialLayout {
  ParameterHandle parameters;
  size_t active_q;
  std::vector<TowerRef> towers;
  Domain domain;
  Order order;
  Encoding encoding = Encoding::CanonicalU64;
};
inline void validate(const PolynomialLayout &p) {
  require(bool(p.parameters), "missing parameters");
  require(p.active_q > 0 && p.active_q <= p.parameters->q().size(),
          "invalid active Q count");
  require(!p.towers.empty(), "empty basis");
  require(p.encoding == Encoding::CanonicalU64, "unsupported residue encoding");
  require((p.domain == Domain::Coefficient && p.order == Order::Natural) ||
              (p.domain == Domain::Evaluation && p.order == Order::BitReversed),
          "domain/order mismatch");
  std::vector<TowerRef> seen;
  for (auto ref : p.towers) {
    p.parameters->tower(ref);
    require(ref.family != Family::Q || ref.index < p.active_q, "inactive Q tower");
    require(std::find(seen.begin(), seen.end(), ref) == seen.end(),
            "duplicate view tower");
    seen.push_back(ref);
  }
  checked_product(
      checked_product(static_cast<size_t>(p.parameters->n()), p.towers.size()),
      sizeof(uint64_t));
}
inline size_t bytes(const PolynomialLayout &p) {
  validate(p);
  return checked_product(
      checked_product(static_cast<size_t>(p.parameters->n()), p.towers.size()),
      sizeof(uint64_t));
}
inline void validate_host_values(const PolynomialLayout &p,
                                 std::span<const uint64_t> values) {
  const size_t count = bytes(p) / sizeof(uint64_t);
  require(values.size() == count, "host length mismatch");
  const size_t n = p.parameters->n();
  for (size_t t = 0; t < p.towers.size(); ++t) {
    const auto q = p.parameters->tower(p.towers[t]).q;
    for (size_t i = 0; i < n; ++i)
      require(values[t * n + i] < q, "noncanonical host residue");
  }
}
struct Device {
  int ordinal;
  uint64_t session;
  bool operator==(const Device &) const = default;
};
struct Token {
  uint64_t id, generation;
  bool operator==(const Token &) const = default;
};
struct Stream {
  Device device;
  Token token;
  bool operator==(const Stream &) const = default;
};
inline void validate(Device d) {
  require(d.ordinal >= 0 && d.session != 0, "invalid device/session");
}
inline void validate(Token t) {
  require(t.id != 0 && t.generation != 0, "invalid token");
}
inline void validate(Stream s) {
  validate(s.device);
  validate(s.token);
}
struct AllocationRecord {
  Token token;
  Device device;
  size_t capacity_bytes;
  PolynomialLayout layout;
  bool live = true;
};
struct View {
  Token allocation;
  size_t first_tower, tower_count, active_q;
};
// Views select whole contiguous towers. QlP cannot be fabricated by shortening
// a full-QP span across the gap left by removed Q towers.
inline PolynomialLayout validate_view(const View &v, const AllocationRecord &record,
                                      Device expected) {
  validate(record.token);
  validate(record.device);
  validate(expected);
  validate(v.allocation);
  require(record.live && v.allocation == record.token, "stale allocation/generation");
  require(record.device == expected, "device/session mismatch");
  require(record.capacity_bytes >= bytes(record.layout), "allocation too short");
  require(v.active_q > 0 && v.active_q <= record.layout.active_q,
          "view cannot add active Q towers");
  require(v.tower_count > 0 && v.first_tower <= record.layout.towers.size() &&
              v.tower_count <= record.layout.towers.size() - v.first_tower,
          "view extent out of range");
  auto out = record.layout;
  out.active_q = v.active_q;
  out.towers.assign(record.layout.towers.begin() + v.first_tower,
                    record.layout.towers.begin() + v.first_tower + v.tower_count);
  validate(out);
  return out;
}
inline size_t byte_offset(const View &v, const AllocationRecord &record) {
  validate_view(v, record, record.device);
  return checked_product(
      checked_product(v.first_tower,
                      static_cast<size_t>(record.layout.parameters->n())),
      sizeof(uint64_t));
}
inline void require_layout(const PolynomialLayout &actual,
                           const PolynomialLayout &expected) {
  validate(actual);
  validate(expected);
  require(*actual.parameters == *expected.parameters &&
              actual.active_q == expected.active_q &&
              actual.towers == expected.towers && actual.domain == expected.domain &&
              actual.order == expected.order && actual.encoding == expected.encoding,
          "operation layout mismatch");
}
inline PolynomialLayout transform_layout(const PolynomialLayout &input, bool inverse) {
  validate(input);
  require(input.domain == (inverse ? Domain::Evaluation : Domain::Coefficient),
          "wrong transform input domain");
  auto output = input;
  output.domain = inverse ? Domain::Coefficient : Domain::Evaluation;
  output.order = inverse ? Order::Natural : Order::BitReversed;
  return output;
}
inline PolynomialLayout pointwise_layout(const PolynomialLayout &a,
                                         const PolynomialLayout &b) {
  require_layout(a, b);
  require(a.domain == Domain::Evaluation, "pointwise input must be evaluation format");
  return a;
}
inline void require_disjoint(const View &a, const AllocationRecord &ar, const View &b,
                             const AllocationRecord &br) {
  validate_view(a, ar, ar.device);
  validate_view(b, br, br.device);
  if (ar.device != br.device || ar.token != br.token)
    return;
  require(a.first_tower + a.tower_count <= b.first_tower ||
              b.first_tower + b.tower_count <= a.first_tower,
          "overlapping out-of-place views");
}

enum class Outcome { Pending, Succeeded, Failed };
struct CompletionRecord {
  Token operation;
  Stream stream;
  Outcome outcome;
};
struct Execution {
  Stream stream;
  uint64_t operation_id;
  std::vector<CompletionRecord> waits;
};
inline void validate_dependency(const CompletionRecord &source,
                                const Execution &consumer) {
  validate(consumer.stream);
  validate(source.stream);
  validate(source.operation);
  require(source.stream.device == consumer.stream.device, "cross-device dependency");
  require(consumer.operation_id > source.operation.id,
          "dependency must already be submitted");
  require(source.outcome == Outcome::Pending || source.outcome == Outcome::Succeeded ||
              source.outcome == Outcome::Failed,
          "invalid completion outcome");
  require(source.outcome != Outcome::Failed, "failed producer");
  // Every declared dependency must be an earlier, nonfailed operation in this session.
  for (const auto &wait : consumer.waits) {
    validate(wait.stream);
    validate(wait.operation);
    require(wait.stream.device == consumer.stream.device &&
                wait.operation.id < consumer.operation_id,
            "invalid dependency session/order");
    require(wait.outcome == Outcome::Pending || wait.outcome == Outcome::Succeeded,
            "failed/invalid explicit dependency");
  }
  if (source.outcome == Outcome::Succeeded || source.stream == consumer.stream)
    return;
  const auto found =
      std::find_if(consumer.waits.begin(), consumer.waits.end(), [&](const auto &w) {
        return w.operation == source.operation && w.stream == source.stream;
      });
  require(found != consumer.waits.end(), "missing cross-stream event wait");
}
// Pin counts are trusted runtime registry facts, not shared_ptr::use_count guesses.
inline bool can_retire(size_t external_leases, size_t in_flight_pins, Outcome readiness,
                       bool quiescence_proven) {
  require(readiness == Outcome::Pending || readiness == Outcome::Succeeded ||
              readiness == Outcome::Failed,
          "invalid readiness state");
  return external_leases == 0 && in_flight_pins == 0 &&
         (readiness == Outcome::Succeeded || quiescence_proven);
}

enum class TableConvention { NaturalPsiPowersInclusive, OpenFheBitReversed };
struct NttPlanKey {
  Device device;
  uint64_t n;
  std::vector<Tower> ordered_towers;
  Encoding encoding;
  TableConvention tables;
  uint32_t arithmetic_revision;
  bool operator==(const NttPlanKey &) const = default;
};
inline NttPlanKey ntt_plan_key(const PolynomialLayout &p, Device d, uint32_t revision) {
  validate(p);
  validate(d);
  require(revision != 0, "missing arithmetic revision");
  std::vector<Tower> towers;
  for (auto ref : p.towers)
    towers.push_back(p.parameters->tower(ref));
  return {d,
          p.parameters->n(),
          std::move(towers),
          p.encoding,
          TableConvention::NaturalPsiPowersInclusive,
          revision};
}
enum class KeyKind { KeySwitch, Relinearization, Rotation };
struct EvaluationKeyCacheKey {
  Device device;
  ParameterHandle parameters;
  Token material; // assigned to a deep snapshot, independent of an OpenFHE object
                  // address/tag
  std::string key_tag;
  KeyKind kind;
  uint32_t key_index, revision;
  bool operator==(const EvaluationKeyCacheKey &b) const {
    return device == b.device && parameters && b.parameters &&
           *parameters == *b.parameters && material == b.material &&
           key_tag == b.key_tag && kind == b.kind && key_index == b.key_index &&
           revision == b.revision;
  }
};
inline void validate(const EvaluationKeyCacheKey &k) {
  validate(k.device);
  validate(k.material);
  require(bool(k.parameters), "missing evaluation-key parameters");
  require(k.revision != 0, "missing key packing revision");
  require(k.kind == KeyKind::KeySwitch || k.kind == KeyKind::Relinearization ||
              k.kind == KeyKind::Rotation,
          "invalid evaluation-key kind");
  require(!k.parameters->p().empty(), "Hybrid key requires P basis");
}
struct ConversionPlanKey {
  Device device;
  ParameterHandle parameters;
  size_t active_q, digit;
  std::vector<TowerRef> from, to;
  uint32_t rule_revision; // includes the uncentered, unreduced-CRT convention
  bool operator==(const ConversionPlanKey &b) const {
    return device == b.device && parameters && b.parameters &&
           *parameters == *b.parameters && active_q == b.active_q && digit == b.digit &&
           from == b.from && to == b.to && rule_revision == b.rule_revision;
  }
};
inline void validate(const ConversionPlanKey &k) {
  validate(k.device);
  require(bool(k.parameters), "missing conversion parameters");
  require(k.rule_revision != 0, "missing conversion rule revision");
  const auto basis = qp_basis(*k.parameters, k.active_q);
  const size_t alpha = k.parameters->alpha();
  require(k.digit < 1 + (k.active_q - 1) / alpha, "conversion digit out of range");
  const size_t first = k.digit * alpha, count = std::min(alpha, k.active_q - first);
  std::vector<TowerRef> from, to;
  for (auto ref : basis) {
    if (ref.family == Family::Q && ref.index >= first && ref.index - first < count)
      from.push_back(ref);
    else
      to.push_back(ref);
  }
  require(k.from == from && k.to == to, "conversion digit/complement order mismatch");
}
struct HoistProvenance {
  ParameterHandle parameters;
  Device device;
  View source;
  Domain domain;
  Order order;
  uint32_t rule_revision;
};
inline void validate(const HoistProvenance &p) {
  require(bool(p.parameters), "missing hoist parameters");
  validate(p.device);
  validate(p.source.allocation);
  require(p.rule_revision != 0, "missing hoist rule revision");
  require(p.domain == Domain::Evaluation && p.order == Order::BitReversed,
          "hoist source must be evaluation ordered");
  q_prefix(*p.parameters, p.source.active_q);
  require(p.source.first_tower == 0 && p.source.tower_count == p.source.active_q,
          "hoist requires the complete active Q input");
}
inline void require_same_hoist(const HoistProvenance &a, const HoistProvenance &b) {
  validate(a);
  validate(b);
  require(a.parameters && b.parameters && *a.parameters == *b.parameters &&
              a.device == b.device && a.source.allocation == b.source.allocation &&
              a.source.first_tower == b.source.first_tower &&
              a.source.tower_count == b.source.tower_count &&
              a.source.active_q == b.source.active_q && a.domain == b.domain &&
              a.order == b.order && a.rule_revision == b.rule_revision,
          "stale/incompatible hoisted digits");
}

enum class FallbackPolicy { RequireDevice, AllowCpuBeforeSubmit };
enum class Phase { Preflight, Enqueued };
enum class Route { NoOp, Device, Cpu };
struct Capability {
  bool available, operation_verified, layout_supported;
};
inline Route choose_route(FallbackPolicy policy, Phase phase, bool valid_input,
                          bool same_format, Capability c) {
  require(policy == FallbackPolicy::RequireDevice ||
              policy == FallbackPolicy::AllowCpuBeforeSubmit,
          "invalid fallback policy");
  require(phase == Phase::Preflight, "runtime failure is not a CPU fallback route");
  require(valid_input, "invalid input cannot fall back");
  if (same_format)
    return Route::NoOp;
  if (c.available && c.operation_verified && c.layout_supported)
    return Route::Device;
  require(policy == FallbackPolicy::AllowCpuBeforeSubmit,
          "required device capability unavailable");
  return Route::Cpu;
}
inline bool initial_ntt_shape_supported(const PolynomialLayout &p) {
  validate(p);
  if (p.parameters->n() != 65536)
    return false;
  for (auto ref : p.towers) {
    const auto q = p.parameters->tower(ref).q;
    if (q < (uint64_t{1} << 49) || q >= (uint64_t{1} << 60))
      return false;
  }
  return true;
}
} // namespace ntt::proposal
