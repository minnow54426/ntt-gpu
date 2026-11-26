#pragma once
// B06 preparation: bounded single-stream prototype, not a production Backend.
// Host test execution and compiled-only CUDA paths are not native GPU evidence.
#include "ntt/backend_contract.hpp"
#include <exception>

namespace ntt::resident {
using proposal::CompletionRecord;
using proposal::Device;
using proposal::Outcome;
using proposal::PolynomialLayout;
using proposal::View;
namespace detail {
struct State;
struct Allocation;
struct Operation;
struct HostStorage;
class Driver;
}
struct Limits {
  size_t device_bytes = 256 * 1024 * 1024;
  size_t host_bytes = 128 * 1024 * 1024;
  size_t operations = 1024;
};
struct Stats {
  size_t submitted, pending, device_bytes, host_bytes;
  size_t uploads, downloads, device_copies;
  size_t parameter_uploads;
  bool closed, poisoned, quiescent;
};
class Completion {
  std::shared_ptr<detail::State> state_;
  std::shared_ptr<detail::Operation> operation_;
  Completion(std::shared_ptr<detail::State>, std::shared_ptr<detail::Operation>);
  friend class Session;
  friend class Polynomial;
  friend class Download;
  friend struct detail::State;
public:
  Completion() = default;
  CompletionRecord query() const;
  void wait() const;
};
class Polynomial {
  std::shared_ptr<detail::State> state_;
  std::shared_ptr<detail::Allocation> allocation_;
  std::shared_ptr<detail::Operation> producer_;
  PolynomialLayout layout_{};
  View view_{};
  Polynomial(std::shared_ptr<detail::State>, std::shared_ptr<detail::Allocation>,
             std::shared_ptr<detail::Operation>, PolynomialLayout, View);
  friend class Session;
  friend struct detail::State;
public:
  Polynomial() = default;
  const PolynomialLayout &layout() const;
  View view() const;
  Completion readiness() const;
};
class Download {
  std::shared_ptr<detail::State> state_;
  std::shared_ptr<detail::Operation> operation_;
  std::shared_ptr<detail::HostStorage> storage_;
  PolynomialLayout layout_{};
  mutable bool validated_ = false;
  Download(std::shared_ptr<detail::State>, std::shared_ptr<detail::Operation>,
           std::shared_ptr<detail::HostStorage>, PolynomialLayout);
  friend class Session;
public:
  Download() = default;
  Completion completion() const;
  // Owned storage is exposed only after successful completion and validation.
  std::span<const uint64_t> wait_values() const;
};
struct HybridKeyDigit {
  std::span<const uint64_t> b, a;
};
class HybridKey {
  std::vector<Polynomial> b_, a_;
  friend class Session;
};
struct HybridContributions {
  Polynomial b, a;
};
class Session {
  std::shared_ptr<detail::State> state_;
  Polynomial upload_checked(const PolynomialLayout &, std::span<const uint64_t>, std::span<const Completion>);
  Polynomial transform(const Polynomial &, bool inverse, std::span<const Completion>);
  Polynomial join(std::span<const Polynomial>);
  Polynomial correct(const Polynomial &, const Polynomial &);
public:
  // Driver injection is a test/prototype seam. Only make_cuda_session creates
  // the real CUDA transport; no automatic host fallback exists.
  explicit Session(std::shared_ptr<detail::Driver>, Limits = {});
  ~Session() noexcept;
  Session(const Session &) = delete;
  Session &operator=(const Session &) = delete;
  Session(Session &&) noexcept = default;
  Session &operator=(Session &&) = delete;
  Device device() const;
  Stats stats() const;
  std::vector<CompletionRecord> history() const;
  Polynomial upload(std::span<const uint64_t>, std::span<const Completion> waits = {});
  // Explicit whole-tower subset of the session's precomputed basis.
  Polynomial upload(const PolynomialLayout &, std::span<const uint64_t>,
                    std::span<const Completion> waits = {});
  Polynomial upload_evaluation(std::span<const uint64_t>, std::span<const Completion> waits = {});
  Polynomial forward(const Polynomial &, std::span<const Completion> waits = {});
  Polynomial inverse(const Polynomial &, std::span<const Completion> waits = {});
  Polynomial pointwise(const Polynomial &, const Polynomial &,
                       std::span<const Completion> waits = {});
  // Coefficient-domain unreduced approximate CRT; source/target are disjoint.
  Polynomial switch_basis(const Polynomial &, const PolynomialLayout &target,
                          std::span<const Completion> waits = {});
  // CKKS / reduced-noise OFF. The session basis must be active Q followed by P.
  // ModUp accepts coefficient or evaluation Q and returns evaluation QP.
  // ModDown requires evaluation QP and returns evaluation Q.
  Polynomial mod_up(const Polynomial &, std::span<const Completion> waits = {});
  Polynomial mod_down(const Polynomial &, std::span<const Completion> waits = {});
  // Precompute Hybrid digits from complete evaluation Q in an active QP session.
  // Each output is evaluation QP and retains the digit's original Q towers.
  std::vector<Polynomial> hybrid_digits(const Polynomial &,
                                       std::span<const Completion> waits = {});
  // Snapshot full-context QP key rows into this session's active QP layout.
  HybridKey upload_hybrid_key(const PolynomialLayout &full_key_layout,
                              std::span<const HybridKeyDigit>);
  // Accumulate both channels in evaluation QP before calling ModDown.
  HybridContributions hybrid_accumulate(std::span<const Polynomial> digits,
                                         const HybridKey &,
                                         std::span<const Completion> waits = {});
  // Add P*c0 in the Q towers, then permute both evaluation QP channels.
  HybridContributions hybrid_rotate_extended(const HybridContributions &,
                                              const Polynomial &c0, uint32_t automorphism);
  Polynomial subview(const Polynomial &, size_t first_tower, size_t count, size_t active_q);
  Download download(const Polynomial &, std::span<const Completion> waits = {});
  Download multiply(std::span<const uint64_t>, std::span<const uint64_t>);
  // Stops admission and drains. After failure, retrying close may reclaim a
  // previously quarantined journal, but it still reports the original error.
  void close();
};
Session make_cuda_session(const PolynomialLayout &coefficient_layout, int device = 0,
                          Limits = {});
} // namespace ntt::resident
