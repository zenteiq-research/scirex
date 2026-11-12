#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 ZenteiQ AiTech Innovations Pvt. Ltd.
# SPDX-License-Identifier: LGPL-3.0-only
#
# Check that ScireX works on this machine (Linux, macOS, or Windows through
# Git Bash or WSL). It prints what it is doing at every step:
#
#   1. detects the operating system, CPU and Python, and says whether they are supported
#   2. imports every module, class and function of the library
#   3. for Transolver, FNO, GINO and WNO: builds dummy data, runs training steps, saves a
#      checkpoint, loads it into a new model and runs inference
#
#   scripts/dev/check_platform.sh                    # everything, on the default JAX device
#   scripts/dev/check_platform.sh --backend gpu      # test a GPU (or cpu, tpu, auto); fails if it is not available
#   scripts/dev/check_platform.sh --cpu              # same as --backend cpu
#   scripts/dev/check_platform.sh --only fno,wno     # some models only
#   scripts/dev/check_platform.sh --steps 5          # more training steps (default 2)
#   scripts/dev/check_platform.sh --python PATH      # interpreter that has scirex installed
#   scripts/dev/check_platform.sh --python-version 3.11   # use python3.11 and fail if it is not that version
#   scripts/dev/check_platform.sh --keep DIR         # keep the checkpoints in DIR
#   scripts/dev/check_platform.sh --light            # fewer CPU threads, for a busy machine
#   scripts/dev/check_platform.sh --help
#
# Exit status: 0 when nothing failed, 1 when a check failed, 2 when it could not start.
# It writes nothing to the repository; checkpoints go to a temporary folder.
set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="${PYTHON:-}"
ARGS=()
LIGHT=0
PYVER=""

while [ $# -gt 0 ]; do
  case "$1" in
    --python) PY="${2:?--python needs a path}"; shift ;;
    --light) LIGHT=1 ;;
    --python-version) PYVER="${2:?--python-version needs a value such as 3.11}"; ARGS+=("$1" "$2"); shift ;;
    --cpu|--skip-imports) ARGS+=("$1") ;;
    --only|--steps|--keep|--backend) ARGS+=("$1" "${2:?$1 needs a value}"); shift ;;
    -h|--help) sed -n '2,26p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown option: $1 (see --help)" >&2; exit 2 ;;
  esac
  shift
done

say() { printf '%s\n' "$*"; }

say "ScireX platform check"
say "------------------------------------------------------------------"
say "step 1: operating system"
UNAME_S="$(uname -s 2>/dev/null || echo unknown)"
UNAME_M="$(uname -m 2>/dev/null || echo unknown)"
case "$UNAME_S" in
  Linux*)
    if grep -qi microsoft /proc/version 2>/dev/null; then
      OS="Windows (WSL2)"
      say "  detected: Linux running inside Windows (WSL2), $UNAME_M"
      say "  note: an NVIDIA GPU works here only with the Windows driver for WSL; see doc/install/troubleshooting.md"
    else
      OS="Linux"
      say "  detected: Linux, $UNAME_M"
      if command -v nvidia-smi >/dev/null 2>&1; then
        say "  NVIDIA GPU: $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>/dev/null | head -1)"
      else
        say "  NVIDIA GPU: none found (nvidia-smi is missing); the CPU will be used"
      fi
    fi ;;
  Darwin*)
    OS="macOS"
    say "  detected: macOS $(sw_vers -productVersion 2>/dev/null), $UNAME_M"
    [ "$UNAME_M" = "arm64" ] && say "  Apple Silicon: use an arm64 Python, not one running under Rosetta"
    say "  note: macOS runs on the CPU only; there is no supported GPU backend" ;;
  MINGW*|MSYS*|CYGWIN*)
    OS="Windows"
    say "  detected: Windows (Git Bash / MSYS), $UNAME_M"
    say "  note: native Windows runs on the CPU only; use WSL2 for an NVIDIA GPU" ;;
  *)
    OS="$UNAME_S"
    say "  detected: $UNAME_S $UNAME_M (not a tested system)" ;;
esac
say "  cores: $( (nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo "${NUMBER_OF_PROCESSORS:-unknown}") | head -1 )"
if [ -r /proc/meminfo ]; then
  say "  memory: $(awk '/MemTotal/ {printf "%.1f GB total", $2/1048576}' /proc/meminfo), $(awk '/MemAvailable/ {printf "%.1f GB available", $2/1048576}' /proc/meminfo)"
fi

say
say "step 2: python interpreter"
if [ -z "$PY" ] && [ -n "$PYVER" ]; then
  # Look for an interpreter of the requested version: python3.11, then the Windows launcher.
  if command -v "python$PYVER" >/dev/null 2>&1; then
    PY="python$PYVER"
  elif command -v py >/dev/null 2>&1 && py "-$PYVER" -c 'pass' >/dev/null 2>&1; then
    PY="py -$PYVER"
  else
    say "  no Python $PYVER found on the path. Install it, or pass --python PATH." >&2
    exit 2
  fi
fi
if [ -z "$PY" ]; then
  for candidate in python3 python py; do
    if command -v "$candidate" >/dev/null 2>&1; then
      if [ "$candidate" = "py" ]; then PY="py -3"; else PY="$candidate"; fi
      break
    fi
  done
fi
if [ -z "$PY" ]; then
  say "  no Python found. Install Python 3.10 to 3.12 and rerun, or pass --python PATH." >&2
  exit 2
fi
# shellcheck disable=SC2086
if ! $PY -c 'import sys; print("  using:", sys.executable, sys.version.split()[0])' 2>/dev/null; then
  say "  '$PY' does not run. Pass a working interpreter with --python PATH." >&2
  exit 2
fi
# shellcheck disable=SC2086
if ! $PY -c 'import jax, flax, optax' 2>/dev/null; then
  say "  JAX, Flax or Optax is missing in this interpreter."
  say "  Install the library first, from the repository root:  $PY -m pip install -e ."
  exit 2
fi

export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export PYTHONWARNINGS="${PYTHONWARNINGS:-ignore}"
export MLFLOW_DISABLE_AGENT_HINT=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false   # do not reserve a whole GPU for the tiny dummy models
if [ "$LIGHT" -eq 1 ]; then
  say "  light mode: 2 CPU threads, lowest priority"
  export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
  export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=2 ${XLA_FLAGS:-}"
fi

say
say "step 3: library checks (imports, public API, train / save / load / infer)"
say "------------------------------------------------------------------"
cd "$ROOT" || exit 2
# shellcheck disable=SC2086
if [ "$LIGHT" -eq 1 ] && command -v nice >/dev/null 2>&1; then
  nice -n 10 $PY scripts/dev/check_platform.py "${ARGS[@]+"${ARGS[@]}"}"
else
  $PY scripts/dev/check_platform.py "${ARGS[@]+"${ARGS[@]}"}"
fi
STATUS=$?
say
if [ $STATUS -eq 0 ]; then
  say "Platform check finished on $OS $UNAME_M: OK"
else
  say "Platform check finished on $OS $UNAME_M: FAILED (status $STATUS). Read the FAIL lines above."
fi
exit $STATUS
