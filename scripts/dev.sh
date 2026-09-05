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
# :8080 is commonly already taken on a dev machine (Docker Desktop claims
# it by default) -- rather than failing outright (a raw port-bind panic
# out of Bee Lab, with no indication *why* the UI then shows "Bee Lab
# doesn't seem to be running"), this script itself checks :8080 first and
# picks the next free port automatically if it's busy, no LAB_PORT needed.
# Set LAB_PORT explicitly to force a specific port instead (still passed
# to both Bee Lab's own PORT env var and the frontend's VITE_LAB_PORT,
# which otherwise aren't linked automatically):
#   LAB_PORT=8081 ./scripts/dev.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# Whether `port` looks free: nothing accepts a TCP connection on it right
# now. Not perfectly race-free (something could grab it between this
# check and Bee Lab actually binding a moment later), but good enough to
# avoid the common case (a long-running unrelated service already using
# :8080) without needing a human to notice and pick a port themselves.
port_is_free() {
  ! (exec 3<>"/dev/tcp/127.0.0.1/$1") 2>/dev/null
}

if [ -n "${LAB_PORT:-}" ]; then
  echo "==> using LAB_PORT=$LAB_PORT (explicitly set)"
elif port_is_free 8080; then
  LAB_PORT=8080
else
  echo "==> :8080 is already in use on this machine -- looking for a free port instead"
  LAB_PORT=""
  for candidate in 8081 8082 8083 8084 8085 8086 8087 8088 8089 8090; do
    if port_is_free "$candidate"; then
      LAB_PORT="$candidate"
      break
    fi
  done
  if [ -z "$LAB_PORT" ]; then
    echo "no free port found in 8080-8090 -- set LAB_PORT explicitly, e.g. LAB_PORT=9000 ./scripts/dev.sh" >&2
    exit 1
  fi
  echo "==> using :$LAB_PORT instead"
fi

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
