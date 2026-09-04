#!/usr/bin/env bash
# Builds the vendored Stockfish submodule in place (external/stockfish/src/stockfish).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="$ROOT/external/stockfish/src"
JOBS="$(command -v nproc >/dev/null 2>&1 && nproc || echo 4)"

if [ ! -f "$SRC/Makefile" ]; then
  echo "external/stockfish is empty. Run: git submodule update --init --recursive" >&2
  exit 1
fi

# ARCH=native is the Stockfish default and picks the best ISA for this CPU.
# 'build' also downloads the default NNUE nets, so it needs network on first run.
make -C "$SRC" -j"$JOBS" build ARCH="${ARCH:-native}"

echo "built: $SRC/stockfish"
