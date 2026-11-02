#!/usr/bin/env bash
# AutoDL one-shot: environment check -> source -> build -> every gate ->
# bench sweep -> VERDICT.txt. Idempotent: safe to re-run after fixes; each
# stage skips what is already satisfied. Exits nonzero if any gate fails.
#
# Usage (on the rented box):
#   bash scripts/autodl_setup.sh              # clone-or-pull, then everything
#   NTT_REPO_DIR=/path/to/ntt-gpu bash scripts/autodl_setup.sh
#   NTT_CUDA_ARCH="89" bash scripts/autodl_setup.sh    # force one arch
#   NTT_BENCH_ARGS="12 16 3" bash scripts/autodl_setup.sh
#
# What it reports (VERDICT.txt + stdout): date, GPU name/driver, toolkit
# version, the compile-arch sanity check, per-gate ctest results, and the
# bench table. That file is the record the README's benchmark rule wants:
# numbers only ever attached to named hardware.

set -euo pipefail

REPO_URL="https://github.com/minnow54426/ntt-gpu.git"
ROOT="${NTT_REPO_DIR:-$HOME/ntt-gpu}"
BUILD="$ROOT/build"
VERDICT="$ROOT/VERDICT.txt"
BENCH_OUT="$ROOT/bench_latest.txt"
BENCH_ARGS="${NTT_BENCH_ARGS:-12 18 3}"

log() { printf '\n== %s ==\n' "$*"; }

log "environment"
# AutoDL CUDA images often keep nvcc off PATH, in /usr/local/cuda/bin.
if ! command -v nvcc >/dev/null 2>&1; then
  if [ -x /usr/local/cuda/bin/nvcc ]; then
    export PATH="/usr/local/cuda/bin:$PATH"
  fi
fi
command -v nvcc >/dev/null 2>&1 || { echo "FATAL: no nvcc (need a CUDA image)"; exit 1; }
NVCC_LINE="$(nvcc --version | tail -1)"
echo "$NVCC_LINE"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader || true

# Missing basics get installed (AutoDL Ubuntu containers are usually root).
need() { command -v "$1" >/dev/null 2>&1; }
if ! need cmake || ! need python3 || ! need git; then
  echo "installing missing build tools"
  SUDO=""
  [ "$(id -u)" -ne 0 ] && SUDO="sudo"
  $SUDO apt-get update -qq
  $SUDO apt-get install -y -qq cmake python3 git
fi

log "source"
if [ -d "$ROOT/.git" ]; then
  git -C "$ROOT" pull --ff-only
else
  git clone "$REPO_URL" "$ROOT"
fi
cd "$ROOT"
REPO_HEAD="$(git rev-parse --short HEAD)"

log "configure + build"
# Arch default matches CMakeLists.txt: 89 (4090) + 120 (5090, CUDA >= 12.8);
# the plain values also embed compute_89 PTX, so one build runs on either
# card. NTT_CUDA_ARCH overrides for a single known box.
ARCH="${NTT_CUDA_ARCH:-}"
if [ -z "$ARCH" ]; then
  case "$NVCC_LINE" in
    *"release 12.8"*|*"release 12.9"*|*"release 13"*) ARCH="89;120" ;;
    *) ARCH="89" ;;
  esac
fi
echo "architectures: $ARCH"
cmake -S . -B "$BUILD" -DCMAKE_CUDA_ARCHITECTURES="$ARCH"
cmake --build "$BUILD" -j"$(nproc)"

# Sanity: confirm the embedded arch/PTX (README: don't trust, check).
ARCH_OK="$(grep -rho 'compute_[0-9]*' "$BUILD/CMakeFiles" --include='*.build' 2>/dev/null | sort -u | tr '\n' ' ' || true)"
echo "compiled arch targets: ${ARCH_OK:-not found (check build.ninja/Makefiles manually)}"

log "gates"
cd "$BUILD"
GATES=FAIL
if ctest --output-on-failure; then GATES=PASS; fi

log "bench"
set +e
"$BUILD/ntt_cuda_cli" bench $BENCH_ARGS | tee "$BENCH_OUT"
set -e

log "verdict"
{
  echo "date (UTC): $(date -u '+%Y-%m-%d %H:%M:%S')"
  echo "repo head: $REPO_HEAD"
  nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader || true
  echo "toolkit: $NVCC_LINE"
  echo "compiled arch targets: ${ARCH_OK:-unknown}"
  echo "gates: $GATES"
  echo
  cat "$BENCH_OUT"
} > "$VERDICT"
cat "$VERDICT"

[ "$GATES" = "PASS" ] || exit 1
echo "all green — VERDICT.txt is the record to keep"
