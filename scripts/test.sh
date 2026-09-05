#!/usr/bin/env bash
# Runs every subproject's test suite (no lint/format/build). For the full
# set of checks CI runs, use ./scripts/check.sh instead.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

echo "==> engine: cargo test"
(cd engine && cargo test)

echo "==> training: pytest"
(cd training && uv run pytest)

echo
echo "All tests passed."
