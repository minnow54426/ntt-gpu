#pragma once
// Vector-file I/O shared by the CLI and the unit tests. The same files are
// produced and consumed by tests/oracle/*.py, which parse an identical
// format — one file, three consumers, one convention.
//
// Format (whitespace-separated; lines starting with '#' are comments):
//   # anything
//   <N> <num_primes P>
//   <q_1> <psi_1>
//   ...
//   <q_P> <psi_P>
//   COEFFS
//   <coeff 0 as decimal, arbitrary precision, in [0, Q)>   (N lines)

#include <cstdint>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace ntt {

struct VectorFile {
  uint64_t n = 0;
  std::vector<uint64_t> q;      // RNS primes
  std::vector<uint64_t> psi;    // per-prime primitive 2N-th roots of unity
  std::vector<std::string> coeffs;  // decimal big-int coefficients
};

namespace detail {

inline std::vector<std::string> tokenize(std::istream& in) {
  std::vector<std::string> tokens;
  std::string line;
  while (std::getline(in, line)) {
    if (!line.empty() && line[0] == '#') continue;
    std::istringstream ls(line);
    std::string tok;
    while (ls >> tok) tokens.push_back(tok);
  }
  return tokens;
}

}  // namespace detail

inline VectorFile read_vector_file(const std::string& path) {
  std::ifstream in(path);
  if (!in) throw std::runtime_error("cannot open vector file: " + path);
  const std::vector<std::string> t = detail::tokenize(in);

  size_t p = 0;
  auto next = [&t, &p]() -> std::string {
    if (p >= t.size()) throw std::runtime_error("vector file: truncated");
    return t[p++];
  };

  VectorFile vf;
  vf.n = std::stoull(next());
  const uint64_t num_primes = std::stoull(next());
  if (vf.n < 2 || (vf.n & (vf.n - 1)) != 0)
    throw std::runtime_error("vector file: N must be a power of two >= 2");
  if (num_primes == 0) throw std::runtime_error("vector file: need >= 1 prime");

  for (uint64_t k = 0; k < num_primes; ++k) {
    vf.q.push_back(std::stoull(next()));
    vf.psi.push_back(std::stoull(next()));
  }
  if (next() != "COEFFS") throw std::runtime_error("vector file: expected COEFFS");
  for (uint64_t i = 0; i < vf.n; ++i) {
    std::string c = next();
    for (char ch : c)
      if (ch < '0' || ch > '9') throw std::runtime_error("vector file: bad coefficient");
    vf.coeffs.push_back(std::move(c));
  }
  return vf;
}

// Reduce an arbitrary-precision decimal string mod q, Horner over digits.
// Requires q < 2^60 so that r*10 + 9 cannot overflow uint64_t.
inline uint64_t reduce_decimal_mod(const std::string& decimal, uint64_t q) {
  if (q >= (uint64_t{1} << 60)) throw std::invalid_argument("q too large for decimal reduce");
  uint64_t r = 0;
  for (char ch : decimal) r = (r * 10 + static_cast<uint64_t>(ch - '0')) % q;
  return r;
}

}  // namespace ntt
