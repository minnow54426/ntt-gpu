#include "support/basis_cuda.hpp"
#include <iostream>
int main() try {
  check_barrett_cuda();
  std::cout << "NATIVE CUDA BARRETT PASS\n";
  return 0;
} catch (const std::exception &error) {
  std::cerr << error.what() << '\n';
  return 1;
}
