#!/usr/bin/env bash
# Run from the intended checkout. No implicit git pull, package installs or
# driver changes: the recorded commit is the one the caller selected.
# Requires an existing pinned OpenFHE installation and CUDA GPU image.
set -euo pipefail
repo_root="${NTT_REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cuda_root="${NTT_CUDA_ROOT:-/usr/local/cuda}"
openfhe_prefix="${NTT_OPENFHE_PREFIX:-$repo_root/build/openfhe-compat/install}"
extra_args=()
if [[ "${NTT_SKIP_SANITIZERS:-0}" == "1" ]]; then extra_args+=(--skip-sanitizers); fi
exec python3 "$repo_root/scripts/gpu_session.py" --mode gpu \
  --cuda-root "$cuda_root" --openfhe-prefix "$openfhe_prefix" \
  --architectures "${NTT_CUDA_ARCH:-native}" --benchmark "${extra_args[@]}" "$@"
