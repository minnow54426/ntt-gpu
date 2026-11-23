#pragma once
#include "ntt/basis_conversion.hpp"

// Native test transport. Allocations and transfers belong to each probe.
std::vector<uint64_t> convert_basis_cuda(const ntt::ApproximateBasisPlan &plan,
                                         uint64_t dimension,
                                         std::span<const uint64_t> input);
void check_shoup_cuda();
void check_barrett_cuda();
