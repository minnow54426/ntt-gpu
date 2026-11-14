#include "ntt/cuda_ntt.hpp"
#include <iostream>
#include <stdexcept>
int main() {
  try {
    ntt::cuda::describe_device(std::cout);
    ntt::cuda::check_native_setup_failures(std::cout);
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n'; return 1;
  }
}
