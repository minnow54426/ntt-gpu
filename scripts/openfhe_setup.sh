#!/usr/bin/env bash
set -euo pipefail

# Fetch the exact OpenFHE release used by this project, install its core/PKE
# libraries under build/, then configure/build/run the compatibility gate.
# Everything downloaded or generated stays under ignored build/ paths.

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
deps_root="${repo_root}/build/openfhe-compat"
src_dir="${deps_root}/src"
openfhe_build="${deps_root}/openfhe-build"
install_dir="${deps_root}/install"
ntt_build="${deps_root}/ntt-build"
openfhe_repo="https://github.com/openfheorg/openfhe-development.git"
openfhe_tag="v1.6.1"
openfhe_commit="b59e1f291965f14fbee5172cd83a20f5b856204f"

mkdir -p "${deps_root}"
if [[ ! -d "${src_dir}/.git" ]]; then
  git clone --depth 1 --branch "${openfhe_tag}" "${openfhe_repo}" "${src_dir}"
fi

actual_commit="$(git -C "${src_dir}" rev-parse HEAD)"
if [[ "${actual_commit}" != "${openfhe_commit}" ]]; then
  echo "OpenFHE ${openfhe_tag} resolved to ${actual_commit}, expected ${openfhe_commit}" >&2
  exit 1
fi

cmake -S "${src_dir}" -B "${openfhe_build}" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="${install_dir}" \
  -DBUILD_SHARED=OFF \
  -DBUILD_STATIC=ON \
  -DBUILD_UNITTESTS=OFF \
  -DBUILD_EXAMPLES=OFF \
  -DBUILD_BENCHMARKS=OFF \
  -DBUILD_EXTRAS=OFF \
  -DWITH_OPENMP=OFF \
  -DNATIVE_SIZE=64 \
  -DWITH_REDUCED_NOISE=OFF \
  -DWITH_NATIVEOPT=OFF
# OpenFHE's install manifest includes core, PKE, and BinFHE archives even when
# this first compatibility executable links only core. Build the complete
# library set now: the D8 Hybrid graph gate links PKE and its dependencies.
cmake --build "${openfhe_build}" --target allmodules -j"${NTT_BUILD_JOBS:-2}"
cmake --install "${openfhe_build}"

cmake -S "${repo_root}" -B "${ntt_build}" \
  -DCMAKE_BUILD_TYPE=Release \
  -DNTT_ENABLE_OPENFHE_COMPAT=ON \
  -DNTT_ENABLE_CUDA=OFF \
  -DCMAKE_PREFIX_PATH="${install_dir}"
cmake --build "${ntt_build}" -j"${NTT_BUILD_JOBS:-2}"
ctest --test-dir "${ntt_build}" --output-on-failure
