#pragma once
// Internal transport seam. No raw allocation or event can become a public handle.
#include "ntt/resident.hpp"
#include <atomic>

namespace ntt::resident::detail {
struct Buffer {
  virtual ~Buffer() = default;
};
struct Event {
  virtual ~Event() = default;
};
class Driver {
  std::atomic<bool> attached_{false};
public:
  Driver() = default;
  Driver(const Driver &) = delete;
  Driver &operator=(const Driver &) = delete;
  virtual ~Driver() = default;
  void attach_once() {
    proposal::require(!attached_.exchange(true), "transport already belongs to a session");
  }
  virtual int ordinal() const = 0;
  virtual const PolynomialLayout &layout() const = 0;
  virtual size_t constant_bytes() const = 0;
  virtual std::unique_ptr<Buffer> allocate(size_t elements) = 0;
  virtual std::unique_ptr<Event> make_event() = 0;
  virtual void upload(Buffer &, std::span<const uint64_t>) = 0;
  virtual void copy(Buffer &, const Buffer &, size_t offset, size_t elements) = 0;
  virtual void copy_range(Buffer &, size_t destination_offset, const Buffer &,
                          size_t source_offset, size_t elements) = 0;
  virtual void transform(Buffer &, const PolynomialLayout &, bool inverse) = 0;
  virtual void pointwise(Buffer &, const Buffer &, size_t offset,
                         const PolynomialLayout &) = 0;
  virtual void hybrid_mac(Buffer &result_b, Buffer &result_a,
                          const Buffer &digit, const Buffer &key_b,
                          const Buffer &key_a, const PolynomialLayout &,
                          bool first_digit) = 0;
  virtual void hybrid_rotate(Buffer &out_b, Buffer &out_a, const Buffer &extended_b,
                             const Buffer &extended_a, const Buffer &c0, size_t c0_offset,
                             const Buffer &scales, const PolynomialLayout &,
                             uint32_t automorphism) = 0;
  virtual void switch_basis(Buffer &, const Buffer &, size_t source_offset,
                            const Buffer &constants, const PolynomialLayout &source,
                            const PolynomialLayout &target) = 0;
  virtual void correct(Buffer &, const Buffer &, size_t left_offset,
                       const Buffer &, size_t right_offset, const Buffer &constants,
                       const PolynomialLayout &) = 0;
  virtual void download(std::span<uint64_t>, const Buffer &, size_t offset) = 0;
  virtual void record(Event &) = 0;
  virtual bool ready(Event &) = 0;
  virtual void wait(Event &) = 0;
  // True proves no queued command can still touch a pinned resource. An error
  // code alone is not proof; false must preserve every operation pin.
  virtual bool quiesce() noexcept = 0;
  virtual void check_health() const {}
};
} // namespace ntt::resident::detail
