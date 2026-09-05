#!/usr/bin/env bash
# One-time (and re-runnable) setup: installs/creates every subproject's
# environment so `./scripts/dev.sh` and `./scripts/check.sh` don't have to.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

echo "==> engine: checking cargo is available"
if ! command -v cargo >/dev/null 2>&1; then
  echo "cargo not found. Install Rust: https://rustup.rs" >&2
  exit 1
fi

if ! command -v uv >/dev/null 2>&1; then
  echo "uv not found. Install it: https://docs.astral.sh/uv/getting-started/installation/" >&2
  exit 1
fi

echo "==> training: syncing Python environment (training/.venv)"
(cd training && uv sync)

# bridge/server.py (Python) is the pre-#67 UCI bridge, superseded by
# lab/ (Rust) as of #69's frontend migration -- the frontend no longer
# talks to it at all. Still synced here since the code hasn't been
# deleted yet (see #68's explicit "not deleted in this PR" decision),
# but ./scripts/dev.sh no longer starts it.
echo "==> bridge: syncing Python environment (bridge/.venv, legacy -- see lab/)"
(cd bridge && uv sync)

echo "==> frontend: installing npm dependencies"
npm --prefix frontend install

if [ ! -f external/stockfish/src/Makefile ]; then
  echo "==> fetching the Stockfish submodule"
  git submodule update --init --recursive
fi

echo
echo "Setup complete. Open bee-chess.code-workspace in VS Code, then:"
echo "  ./scripts/dev.sh    # build engines, start Bee Lab and the UI"
echo "  ./scripts/check.sh  # run every subproject's lint/format/test checks"
