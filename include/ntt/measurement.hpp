#pragma once
// Host-wall timing protocol: reset before every warmup/sample, outside timing.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <numeric>
#include <ostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace ntt::measurement {
inline void validate_counts(size_t warmups, size_t samples) {
  if (warmups < 1 || warmups > 10000 || samples < 2 || samples > 100000)
    throw std::invalid_argument("require 1..10000 warmups and 2..100000 samples");
}
template <class Clock = std::chrono::steady_clock, class Prepare, class Operation>
std::vector<double> collect(size_t warmups, size_t samples, Prepare prepare,
                            Operation operation) {
  validate_counts(warmups, samples);
  for (size_t index = 0; index < warmups; ++index) {
    prepare(); operation();
  }
  std::vector<double> result;
  result.reserve(samples);
  for (size_t index = 0; index < samples; ++index) {
    prepare();
    const auto start = Clock::now();
    operation();
    const auto finish = Clock::now();
    const double elapsed = std::chrono::duration<double, std::milli>(finish - start).count();
    if (!std::isfinite(elapsed) || elapsed < 0)
      throw std::runtime_error("invalid wall-clock sample");
    result.push_back(elapsed);
  }
  return result;
}
inline double percentile(std::vector<double> values, double fraction) {
  if (values.empty() || !std::isfinite(fraction) || fraction < 0 || fraction > 1)
    throw std::invalid_argument("invalid percentile input");
  for (auto value : values)
    if (!std::isfinite(value) || value < 0)
      throw std::invalid_argument("invalid sample");
  std::sort(values.begin(), values.end());
  const double position = (values.size() - 1) * fraction;
  const auto lower = static_cast<size_t>(position);
  const auto upper = std::min(lower + 1, values.size() - 1);
  return values[lower] + (values[upper] - values[lower]) * (position - lower);
}
inline void json_string(std::ostream &output, const std::string &value) {
  const char *hex = "0123456789abcdef";
  output << '"';
  for (unsigned char character : value) {
    if (character == '"' || character == '\\') output << '\\' << character;
    else if (character < 32)
      output << "\\u00" << hex[character >> 4] << hex[character & 15];
    else output << character;
  }
  output << '"';
}
inline void write_samples(std::ostream &output, const std::string &operation,
                           const std::vector<double> &samples) {
  const auto median = percentile(samples, .5);
  output << std::setprecision(17);
  for (size_t index = 0; index < samples.size(); ++index) {
    output << "{\"type\":\"sample\",\"operation\":";
    json_string(output, operation);
    output << ",\"iteration\":" << index << ",\"milliseconds\":" << samples[index] << "}\n";
  }
  output << "{\"type\":\"summary\",\"operation\":";
  json_string(output, operation);
  output << ",\"count\":" << samples.size()
         << ",\"mean_ms\":" << std::accumulate(samples.begin(), samples.end(), 0.0) / samples.size()
         << ",\"median_ms\":" << median << ",\"p5_ms\":" << percentile(samples, .05)
         << ",\"p95_ms\":" << percentile(samples, .95) << "}\n";
}
} // namespace ntt::measurement
