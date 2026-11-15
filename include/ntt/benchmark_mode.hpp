#pragma once
// One definition of the three timing protocols; existing CLI/JSONL names stay stable.
#include <cstddef>
#include <optional>
#include <stdexcept>
#include <string_view>

namespace ntt::measurement {
enum class BenchmarkMode { PublicWithCpu, PublicGpuOnly, ResidentEvents };

struct BenchmarkProtocol {
  BenchmarkMode mode;
  std::string_view command;
  std::string_view timing;
  size_t operation_count;
};

inline constexpr BenchmarkProtocol kBenchmarkProtocols[] = {
    {BenchmarkMode::PublicWithCpu, "bench-jsonl", "host_wall_public_call", 6},
    {BenchmarkMode::PublicGpuOnly, "gpu-bench-jsonl", "host_wall_gpu_public_call", 5},
    {BenchmarkMode::ResidentEvents, "events-jsonl", "cuda_event_resident_and_pinned_copy", 6},
};

inline const BenchmarkProtocol &benchmark_protocol(BenchmarkMode mode) {
  for (const auto &protocol : kBenchmarkProtocols)
    if (protocol.mode == mode) return protocol;
  throw std::invalid_argument("unknown benchmark mode");
}

inline std::optional<BenchmarkMode> benchmark_mode(std::string_view command) {
  for (const auto &protocol : kBenchmarkProtocols)
    if (protocol.command == command) return protocol.mode;
  return std::nullopt;
}
} // namespace ntt::measurement
