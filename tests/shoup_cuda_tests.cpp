#include "support/basis_cuda.hpp"
#include <iostream>
int main() try {
  check_shoup_cuda();
  std::cout << "NATIVE CUDA SHOUP PASS\n";
  return 0;
} catch (const std::exception &error) {
  std::cerr << error.what() << '\n';
  return 1;
}
