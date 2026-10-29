// ntt_cli: consumes a vector file (see include/ntt/vector_io.hpp) and either
// prints the forward RNS negacyclic NTT per limb (for the Python oracle to
// verify) or checks forward/inverse roundtrip identity.

#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#include "ntt/ntt.hpp"
#include "ntt/vector_io.hpp"

namespace {

int cmd_transform(const ntt::VectorFile& vf) {
  for (size_t k = 0; k < vf.q.size(); ++k) {
    const ntt::NegacyclicNtt ntt(vf.n, vf.q[k], vf.psi[k]);
    std::vector<uint64_t> a(vf.n);
    for (uint64_t i = 0; i < vf.n; ++i)
      a[i] = ntt::reduce_decimal_mod(vf.coeffs[i], vf.q[k]);
    ntt.forward(a);
    std::cout << "LIMB " << k << "\n";
    for (uint64_t i = 0; i < vf.n; ++i) std::cout << a[i] << "\n";
  }
  return 0;
}

int cmd_identity(const ntt::VectorFile& vf) {
  for (size_t k = 0; k < vf.q.size(); ++k) {
    const ntt::NegacyclicNtt ntt(vf.n, vf.q[k], vf.psi[k]);
    std::vector<uint64_t> a(vf.n), orig(vf.n);
    for (uint64_t i = 0; i < vf.n; ++i) {
      orig[i] = a[i] = ntt::reduce_decimal_mod(vf.coeffs[i], vf.q[k]);
    }
    ntt.forward(a);
    ntt.inverse(a);
    for (uint64_t i = 0; i < vf.n; ++i)
      if (a[i] != orig[i]) {
        std::cerr << "IDENTITY FAIL limb " << k << " coeff " << i << ": "
                  << a[i] << " != " << orig[i] << "\n";
        return 1;
      }
  }
  std::cout << "IDENTITY OK\n";
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 3) {
    std::cerr << "usage: ntt_cli <vector-file> <transform|identity>\n";
    return 2;
  }
  try {
    const ntt::VectorFile vf = ntt::read_vector_file(argv[1]);
    const std::string cmd = argv[2];
    if (cmd == "transform") return cmd_transform(vf);
    if (cmd == "identity") return cmd_identity(vf);
    std::cerr << "unknown command: " << cmd << "\n";
    return 2;
  } catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << "\n";
    return 1;
  }
}
