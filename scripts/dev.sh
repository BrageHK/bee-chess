#!/usr/bin/env bash
# One command to watch the engines fight: builds whatever is missing, starts
# Bee Lab (the Rust orchestration server -- see #67/lab/) and the Vite dev
# server, and shuts both down on Ctrl-C.
#
# Not bridge/server.py (the old Python bridge) -- as of #69's frontend
# migration, the frontend (frontend/src/labClient.ts) only ever talks to
# Bee Lab's HTTP+WebSocket API, on :8080 by default. Running the Python
# bridge here instead would start the wrong backend entirely: the UI would
# load, but every request would go to a port nothing is listening on.
#
# Override the port with LAB_PORT if :8080 is already taken on this
# machine (Docker Desktop commonly claims it) -- this passes the same
# value to both Bee Lab (its own PORT env var) and the frontend
# (VITE_LAB_PORT), which otherwise aren't linked automatically:
#   LAB_PORT=8081 ./scripts/dev.sh
set -euo pipefail

LAB_PORT="${LAB_PORT:-8080}"

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

echo "==> building Bee Lab"
cargo build -p bee-lab

if [ ! -d frontend/node_modules ]; then
  echo "==> installing frontend dependencies"
  npm --prefix frontend install
fi

# Bee Lab also serves frontend/dist/ as a static fallback (useful for
# running it standalone, with no Vite dev server at all -- see
# lab/README.md), so it insists that directory exist even though this
# script's own UI comes from Vite's dev server on :5173 instead. A stale
# build there is harmless (Vite is what's actually served here); a
# *missing* one would make Bee Lab refuse to start at all.
if [ ! -d frontend/dist ]; then
  echo "==> building the frontend once (Bee Lab requires frontend/dist/ to exist)"
  npm --prefix frontend run build
fi

echo "==> starting Bee Lab on http://localhost:$LAB_PORT"
PORT="$LAB_PORT" cargo run -p bee-lab &
LAB_PID=$!

cleanup() { kill "$LAB_PID" 2>/dev/null || true; }
trap cleanup EXIT INT TERM

echo "==> starting the UI on http://localhost:5173"
VITE_LAB_PORT="$LAB_PORT" npm --prefix frontend run dev
