#!/usr/bin/env bash
# One command to watch the engines fight: builds whatever is missing, starts
# the UCI bridge and the Vite dev server, and shuts both down on Ctrl-C.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [ ! -f external/stockfish/src/Makefile ]; then
  echo "==> fetching the Stockfish submodule"
  git submodule update --init --recursive
fi

if [ ! -x external/stockfish/src/stockfish ]; then
  echo "==> building Stockfish (first run only, a few minutes)"
  ./scripts/build-stockfish.sh
fi

echo "==> building Bee"
./scripts/build-bee.sh

if [ ! -d frontend/node_modules ]; then
  echo "==> installing frontend dependencies"
  npm --prefix frontend install
fi

# The bridge needs `websockets`; uv gives it an ephemeral env so nothing has
# to be installed system-wide.
if command -v uv >/dev/null 2>&1; then
  BRIDGE=(uv run --quiet --with websockets python bridge/server.py)
else
  BRIDGE=(python3 bridge/server.py)
fi

echo "==> starting the UCI bridge"
"${BRIDGE[@]}" &
BRIDGE_PID=$!

cleanup() { kill "$BRIDGE_PID" 2>/dev/null || true; }
trap cleanup EXIT INT TERM

echo "==> starting the UI on http://localhost:5173"
npm --prefix frontend run dev
