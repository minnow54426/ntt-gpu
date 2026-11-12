#pragma once
// D9 proposed interface: compile-checked, with no runtime implementation or
// factory. All records must be issued/validated by a future private registry.
#include "ntt/backend_contract.hpp"

namespace ntt::proposal {
class Completion {
public:
  virtual ~Completion() = default;
  virtual CompletionRecord query() const = 0; // refresh from actual event status
  virtual void wait() const = 0;              // propagate asynchronous errors
};
using CompletionHandle = std::shared_ptr<const Completion>;

class StreamLease {
public:
  virtual ~StreamLease() = default;
  virtual Stream identity() const = 0;
};
using StreamHandle = std::shared_ptr<const StreamLease>;

class DevicePolynomial {
public:
  virtual ~DevicePolynomial() = default;
  virtual const PolynomialLayout &layout() const = 0;
  virtual View view() const = 0;
  virtual CompletionHandle readiness() const = 0;
};
using PolynomialHandle = std::shared_ptr<const DevicePolynomial>;

struct Submit {
  StreamHandle stream;
  std::vector<CompletionHandle> dependencies;
};
struct PendingPolynomial {
  PolynomialHandle value; // immutable; consuming it must respect readiness
  CompletionHandle completion;
};
class HostResult {
public:
  virtual ~HostResult() = default;
  virtual CompletionHandle completion() const = 0;
  // Waits, propagates failure, then exposes owned canonical host storage.
  virtual std::span<const uint64_t> wait_values() const = 0;
};
using HostResultHandle = std::shared_ptr<const HostResult>;

enum class Operation { Forward, Inverse, Pointwise };
struct CacheStats {
  size_t budget_bytes;
  size_t allocated_bytes; // includes pending uploads; excludes operand/scratch pools
  size_t pinned_bytes;    // subset of allocated_bytes, counted once per entry
};
class Backend {
public:
  virtual ~Backend() = default; // no throw; outstanding work stays pinned
  virtual Device device() const = 0;
  virtual Capability capability(Operation, const PolynomialLayout &) const = 0;
  virtual StreamHandle create_stream() = 0;
  // Copy the host span to owned staging before returning. Never retain a
  // borrowed span across asynchronous work. Validate canonical residues first.
  virtual PendingPolynomial upload(const PolynomialLayout &, std::span<const uint64_t>,
                                   const Submit &) = 0;
  virtual PendingPolynomial forward(PolynomialHandle, const Submit &) = 0;
  virtual PendingPolynomial inverse(PolynomialHandle, const Submit &) = 0;
  virtual PendingPolynomial pointwise(PolynomialHandle, PolynomialHandle,
                                      const Submit &) = 0;
  // Whole-tower immutable alias, retaining allocation and readiness leases.
  virtual PolynomialHandle subview(PolynomialHandle, size_t first_tower, size_t count,
                                   size_t active_q) = 0;
  virtual HostResultHandle download(PolynomialHandle, const Submit &) = 0;
  virtual CacheStats cache_stats() const = 0;
  // Evict eligible entries first; reject a lower budget if pins prevent it.
  // Reserve budget before allocation; no silent cache-budget oversubscription.
  virtual void set_cache_budget(size_t bytes) = 0;
  virtual size_t trim_cache() = 0; // actual bytes freed from eligible entries
  virtual void close() = 0; // drain, report errors, invalidate further submission
};
// Hybrid operations and key/hoist handles are deliberately not executable
// interfaces yet. D8 fixes their graph; D9 fixes metadata/cache/lifetime rules.
// Exact conversion/MAC implementations and acceptance belong to later phases.
} // namespace ntt::proposal
