// Host arithmetic probe for an independent Python big-integer oracle, plus
// full-size algebraic tests. No CUDA execution is implied by this target.
#include <algorithm>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "ntt/ntt.hpp"
#include "ntt/param_gen.hpp"

namespace {
void require(bool condition, const std::string &message) {
  if (!condition)
    throw std::runtime_error(message);
}

void check_transform_contract() {
  constexpr uint64_t dimension = 65536;
  for (unsigned modulus_bits : {50U, 55U, 60U}) {
    const uint64_t modulus = ntt::find_primes(dimension, 1, modulus_bits).front();
    const uint64_t root = ntt::find_psi(modulus, dimension);
    const ntt::NegacyclicNtt transform(dimension, modulus, root);
    uint64_t random_state = 0xD6;
    for (unsigned pattern = 0; pattern < 6; ++pattern) {
      std::vector<uint64_t> input(dimension, 0);
      for (uint64_t index = 0; index < dimension; ++index) {
        random_state = random_state * 6364136223846793005ULL + 1;
        switch (pattern) {
        case 0: break; // all zero
        case 1: input[index] = modulus - 1; break;
        case 2: input[index] = index % 2 ? modulus - 1 : 0; break;
        case 3: input[index] = index == dimension - 1 ? 1 : 0; break;
        case 4: input[index] = index; break;
        case 5: input[index] = random_state % modulus; break;
        }
      }
      auto values = input;
      transform.forward(values);
      require(std::all_of(values.begin(), values.end(),
                          [modulus](uint64_t value) { return value < modulus; }),
              "forward produced noncanonical residue");
      if (pattern == 0)
        require(values == input, "zero forward differs");
      transform.inverse(values);
      require(values == input, "roundtrip differs at width " +
                                   std::to_string(modulus_bits));
      // Treat the same pattern as an arbitrary evaluation array, rather
      // than testing inverse only on this forward implementation's output.
      values = input;
      transform.inverse(values);
      require(std::all_of(values.begin(), values.end(),
                          [modulus](uint64_t value) { return value < modulus; }),
              "inverse produced noncanonical residue");
      transform.forward(values);
      require(values == input, "evaluation roundtrip differs");
    }
    // Closed-form negacyclic product: x^(N-1) * x = -1, NOT +1.
    std::vector<uint64_t> left(dimension, 0), right(dimension, 0);
    left.back() = 1;
    right[1] = 1;
    const auto product = ntt::ntt_multiply(transform, left, right);
    require(product[0] == modulus - 1 &&
                std::all_of(product.begin() + 1, product.end(),
                            [](uint64_t value) { return value == 0; }),
            "negacyclic wrap sign differs");
    std::cout << "N=" << dimension << " bits=" << modulus_bits
              << " six patterns and wrap product passed\n";
  }
}
} // namespace

int main(int argc, char **argv) {
  try {
    if (argc == 2 && std::string(argv[1]) == "--transforms") {
      check_transform_contract();
      return 0;
    }
    // Protocol: operation operand exponent-or-second-operand modulus.
    std::string operation;
    uint64_t first, second, modulus;
    while (std::cin >> operation >> first >> second >> modulus) {
      if (modulus == 0)
        throw std::invalid_argument("probe refuses modulus zero");
      if (operation == "add")
        std::cout << ntt::add_mod(first, second, modulus);
      else if (operation == "sub")
        std::cout << ntt::sub_mod(first, second, modulus);
      else if (operation == "mul")
        std::cout << ntt::mul_mod(first, second, modulus);
      else if (operation == "pow")
        std::cout << ntt::pow_mod(first, second, modulus);
      else if (operation == "inv") {
        try {
          std::cout << ntt::inv_mod(first, modulus);
        } catch (const std::invalid_argument &) {
          std::cout << "invalid";
        }
      } else
        throw std::invalid_argument("unknown probe operation");
      std::cout << '\n';
    }
    if (!std::cin.eof())
      throw std::invalid_argument("malformed probe input");
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
  return 0;
}
