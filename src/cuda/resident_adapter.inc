// Included in ntt_kernels.cu to reuse the exact existing kernels/launch schedule.
// All queued work uses the owned stream; intermediate arrays stay on device.
// Boundary uploads/downloads use owned host staging.
#include <array>
#include "ntt/resident_conversion.hpp"
namespace ntt::cuda {
namespace resident_impl {
using namespace ::ntt::proposal;
namespace transport = ::ntt::resident::detail;
struct Plan {
  uint64_t transform_length, limb_count, contiguous_tile_length, strided_group_length;
  uint32_t threads_per_block = kThreadsPerBlock;
  uint64_t *device_twiddle_powers, *device_inverse_twiddle_powers;
  uint64_t *device_moduli, *device_inverse_transform_lengths;
  uint32_t contiguous_tile_block_count, elementwise_block_count;
};
__global__ void resident_basis_kernel(uint64_t *out, const uint64_t *in, uint64_t n,
                                      uint64_t source_count, uint64_t target_count,
                                      const uint64_t *table) {
  const uint64_t i = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n * target_count)
    out[i] = ::ntt::resident::detail::resident_basis_value(in, n, i, source_count, target_count, table);
}
__global__ void resident_correction_kernel(uint64_t *out, const uint64_t *left,
                                           const uint64_t *right, uint64_t n,
                                           uint64_t count, const uint64_t *table) {
  const uint64_t i = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n * count)
    out[i] = ::ntt::resident::detail::resident_correction_value(left[i], right[i], i/n, count, table);
}
struct Context : std::enable_shared_from_this<Context> {
  PolynomialLayout base;
  int device;
  cudaStream_t stream = nullptr;
  std::array<uint64_t *, 4> tables{};
  std::array<std::vector<uint64_t>, 4> host;
  std::shared_ptr<Context> setup_lease;
  bool cleanup_failed = false;
  explicit Context(PolynomialLayout p, int ordinal) : base(std::move(p)), device(ordinal) {
    const auto n = base.parameters->n();
    host[0].reserve(checked_product(base.towers.size(), n + 1));
    host[1].reserve(host[0].capacity());
    host[2].reserve(base.towers.size()); host[3].reserve(base.towers.size());
    for (auto ref : base.towers) {
      const auto t = base.parameters->tower(ref);
      const auto forward_table = transform_twiddles(n, t.psi, t.q);
      const auto inverse_table = transform_twiddles(n, inv_mod(t.psi, t.q), t.q);
      host[0].insert(host[0].end(), forward_table.begin(), forward_table.end());
      host[1].insert(host[1].end(), inverse_table.begin(), inverse_table.end());
      host[2].push_back(t.q); host[3].push_back(inv_mod(n, t.q));
    }
  }
  bool quiesce() noexcept {
    if (!stream) return true;
    try { DeviceGuard guard(device); return cudaStreamSynchronize(stream) == cudaSuccess; }
    catch (...) { return false; }
  }
  void initialize() {
    setup_lease = shared_from_this();
    try {
      DeviceGuard guard(device);
      CHECK_CUDA(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
      for (size_t i = 0; i < tables.size(); ++i)
        CHECK_CUDA(cudaMalloc(reinterpret_cast<void **>(&tables[i]), host[i].size() * sizeof(uint64_t)));
      for (size_t i = 0; i < tables.size(); ++i)
        CHECK_CUDA(cudaMemcpyAsync(tables[i], host[i].data(), host[i].size() * sizeof(uint64_t), cudaMemcpyHostToDevice, stream));
      CHECK_CUDA(cudaStreamSynchronize(stream));
      for (auto &values : host) std::vector<uint64_t>().swap(values);
      setup_lease.reset();
    } catch (...) {
      if (quiesce()) setup_lease.reset(); // Otherwise retain tables AND setup host storage.
      throw;
    }
  }
  void healthy() const {
    if (cleanup_failed) throw std::runtime_error("resident CUDA cleanup failed");
  }
  ~Context() noexcept {
    // Reachable only after proven setup/operation quiescence. Unproven contexts
    // retain setup_lease or the runtime's flight lease instead of destructing.
    try {
      DeviceGuard guard(device);
      for (auto *p : tables) if (p && cudaFree(p) != cudaSuccess) cleanup_failed = true;
      if (stream && cudaStreamDestroy(stream) != cudaSuccess) cleanup_failed = true;
    } catch (...) {}
  }
  Plan plan(const PolynomialLayout &p) const {
    validate(p);
    require(*p.parameters == *base.parameters && p.active_q <= base.active_q,
            "foreign resident CUDA parameters");
    const auto it = std::find(base.towers.begin(), base.towers.end(), p.towers.front());
    const size_t first = static_cast<size_t>(it - base.towers.begin());
    require(first < base.towers.size() && p.towers.size() <= base.towers.size() - first &&
              std::equal(p.towers.begin(), p.towers.end(), it), "noncontiguous CUDA plan view");
    const uint64_t n = base.parameters->n(), k = p.towers.size(), tile = std::min<uint64_t>(256, n);
    return {n, k, tile, n / 256, kThreadsPerBlock,
            tables[0] + first * (n + 1), tables[1] + first * (n + 1),
            tables[2] + first, tables[3] + first,
            static_cast<uint32_t>(n / tile), static_cast<uint32_t>((n * k + 255) / 256)};
  }
};
struct Buffer final : transport::Buffer {
  std::shared_ptr<Context> context;
  uint64_t *data = nullptr;
  size_t elements;
  Buffer(std::shared_ptr<Context> c, size_t n) : context(std::move(c)), elements(n) {
    DeviceGuard guard(context->device); context->healthy();
    CHECK_CUDA(cudaMalloc(reinterpret_cast<void **>(&data), n * sizeof(uint64_t)));
  }
  ~Buffer() noexcept override {
    try { DeviceGuard guard(context->device); if (cudaFree(data) != cudaSuccess) context->cleanup_failed = true; }
    catch (...) { context->cleanup_failed = true; }
  }
};
struct Event final : transport::Event {
  std::shared_ptr<Context> context;
  cudaEvent_t event = nullptr;
  bool recorded = false;
  explicit Event(std::shared_ptr<Context> c) : context(std::move(c)) {
    DeviceGuard guard(context->device); context->healthy();
    CHECK_CUDA(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));
  }
  ~Event() noexcept override {
    try { DeviceGuard guard(context->device); if (cudaEventDestroy(event) != cudaSuccess) context->cleanup_failed = true; }
    catch (...) { context->cleanup_failed = true; }
  }
};
class Driver final : public transport::Driver {
  std::shared_ptr<Context> context_;
  const Buffer &buffer(const transport::Buffer &b) const {
    const auto &out = dynamic_cast<const Buffer &>(b);
    require(out.context == context_, "foreign CUDA buffer"); return out;
  }
  Event &event(transport::Event &e) const {
    auto &out = dynamic_cast<Event &>(e);
    require(out.context == context_, "foreign CUDA event"); return out;
  }
public:
  Driver(PolynomialLayout p, int ordinal) : context_(std::make_shared<Context>(std::move(p), ordinal)) { context_->initialize(); }
  int ordinal() const override { return context_->device; }
  const PolynomialLayout &layout() const override { return context_->base; }
  size_t constant_bytes() const override {
    return checked_product(checked_product(context_->base.towers.size(),
           2 * (context_->base.parameters->n() + 1) + 2), sizeof(uint64_t));
  }
  void check_health() const override { context_->healthy(); }
  std::unique_ptr<transport::Buffer> allocate(size_t n) override { return std::make_unique<Buffer>(context_, n); }
  std::unique_ptr<transport::Event> make_event() override { return std::make_unique<Event>(context_); }
  void upload(transport::Buffer &out, std::span<const uint64_t> source) override {
    const auto &a = buffer(out); require(a.elements == source.size(), "CUDA upload extent");
    DeviceGuard guard(ordinal()); check_health();
    CHECK_CUDA(cudaMemcpyAsync(a.data, source.data(), source.size_bytes(), cudaMemcpyHostToDevice, context_->stream));
  }
  void copy(transport::Buffer &out, const transport::Buffer &input, size_t offset, size_t n) override {
    const auto &a = buffer(out), &b = buffer(input);
    require(a.elements == n && offset <= b.elements && n <= b.elements - offset, "CUDA copy extent");
    DeviceGuard guard(ordinal()); check_health();
    CHECK_CUDA(cudaMemcpyAsync(a.data, b.data + offset, n * sizeof(uint64_t), cudaMemcpyDeviceToDevice, context_->stream));
  }
  void copy_range(transport::Buffer &out, size_t destination, const transport::Buffer &input,
                   size_t source, size_t n) override {
    const auto &a = buffer(out), &b = buffer(input);
    require(destination <= a.elements && n <= a.elements-destination
              && source <= b.elements && n <= b.elements-source, "CUDA range copy extent");
    DeviceGuard guard(ordinal()); check_health();
    CHECK_CUDA(cudaMemcpyAsync(a.data+destination, b.data+source, n*sizeof(uint64_t),
                               cudaMemcpyDeviceToDevice, context_->stream));
  }
  void transform(transport::Buffer &out, const PolynomialLayout &p, bool inverse) override {
    const auto &a = buffer(out); require(a.elements == bytes(p) / sizeof(uint64_t), "CUDA transform extent");
    auto plan = context_->plan(p); DeviceGuard guard(ordinal()); check_health();
    if (inverse) launch_inverse_transform(plan, a.data, context_->stream);
    else launch_forward_transform(plan, a.data, context_->stream);
  }
  void pointwise(transport::Buffer &out, const transport::Buffer &right, size_t offset,
                 const PolynomialLayout &p) override {
    const auto &a = buffer(out), &b = buffer(right); auto plan = context_->plan(p);
    require(a.elements == bytes(p) / sizeof(uint64_t) && offset <= b.elements &&
              a.elements <= b.elements - offset, "CUDA pointwise extent");
    DeviceGuard guard(ordinal()); check_health();
    pointwise_multiply_kernel<<<plan.elementwise_block_count, kThreadsPerBlock, 0, context_->stream>>>(
        a.data, b.data + offset, plan.device_moduli, plan.transform_length, plan.limb_count);
    CHECK_CUDA(cudaGetLastError());
  }
  void switch_basis(transport::Buffer &out, const transport::Buffer &input, size_t offset,
                     const transport::Buffer &constants, const PolynomialLayout &source,
                     const PolynomialLayout &target) override {
    const auto &a = buffer(out), &b = buffer(input), &table = buffer(constants);
    (void)context_->plan(source); (void)context_->plan(target);
    const auto n = target.parameters->n(), s = source.towers.size(), t = target.towers.size();
    require(a.elements == n*t && offset <= b.elements && n*s <= b.elements-offset
              && table.elements == 3*s+t+s*t, "CUDA basis extent");
    DeviceGuard guard(ordinal()); check_health();
    resident_basis_kernel<<<static_cast<uint32_t>((a.elements+255)/256), 256, 0, context_->stream>>>(
        a.data, b.data+offset, n, s, t, table.data);
    CHECK_CUDA(cudaGetLastError());
  }
  void correct(transport::Buffer &out, const transport::Buffer &left, size_t left_offset,
                 const transport::Buffer &right, size_t right_offset, const transport::Buffer &constants,
                 const PolynomialLayout &p) override {
    const auto &a = buffer(out), &b = buffer(left), &c = buffer(right), &table = buffer(constants);
    auto plan = context_->plan(p);
    require(a.elements == p.parameters->n()*p.towers.size()
              && left_offset <= b.elements && a.elements <= b.elements-left_offset
              && right_offset <= c.elements && a.elements <= c.elements-right_offset
              && table.elements == 2*p.towers.size(), "CUDA correction extent");
    DeviceGuard guard(ordinal()); check_health();
    resident_correction_kernel<<<plan.elementwise_block_count, 256, 0, context_->stream>>>(
        a.data, b.data+left_offset, c.data+right_offset, plan.transform_length, plan.limb_count, table.data);
    CHECK_CUDA(cudaGetLastError());
  }
  void download(std::span<uint64_t> target, const transport::Buffer &input, size_t offset) override {
    const auto &a = buffer(input);
    require(offset <= a.elements && target.size() <= a.elements - offset, "CUDA download extent");
    DeviceGuard guard(ordinal()); check_health();
    CHECK_CUDA(cudaMemcpyAsync(target.data(), a.data + offset, target.size_bytes(), cudaMemcpyDeviceToHost, context_->stream));
  }
  void record(transport::Event &e) override {
    auto &out = event(e); DeviceGuard guard(ordinal()); check_health();
    CHECK_CUDA(cudaEventRecord(out.event, context_->stream)); out.recorded = true;
  }
  bool ready(transport::Event &e) override {
    auto &out = event(e); require(out.recorded, "unrecorded CUDA event");
    DeviceGuard guard(ordinal()); check_health(); const auto status = cudaEventQuery(out.event);
    if (status == cudaErrorNotReady) return false;
    CHECK_CUDA(status); return true;
  }
  void wait(transport::Event &e) override {
    auto &out = event(e); require(out.recorded, "unrecorded CUDA event");
    DeviceGuard guard(ordinal()); check_health(); CHECK_CUDA(cudaEventSynchronize(out.event));
  }
  bool quiesce() noexcept override { return context_->quiesce(); }
};
} // namespace resident_impl
} // namespace ntt::cuda

namespace ntt::resident {
Session make_cuda_session(const PolynomialLayout &p, int device, Limits limits) {
  proposal::validate(p);
  proposal::require(p.domain == proposal::Domain::Coefficient && p.order == proposal::Order::Natural &&
                    p.parameters->n() <= 65536 && p.towers.size() <= 65535 && device >= 0 && limits.operations,
                    "unsupported resident CUDA session");
  const auto constant_bytes = proposal::checked_product(proposal::checked_product(p.towers.size(),
      2 * (p.parameters->n() + 1) + 2), sizeof(uint64_t));
  proposal::require(constant_bytes <= limits.device_bytes && constant_bytes <= limits.host_bytes,
                    "CUDA plan exceeds resident setup budget");
  return Session(std::make_shared<cuda::resident_impl::Driver>(p, device), limits);
}
} // namespace ntt::resident
