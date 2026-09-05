#!/usr/bin/env bash
# Runs every subproject's format/lint/test checks locally, mirroring what
# CI runs per area (see .github/workflows/ci-*.yml). Run ./scripts/setup.sh
# first if any environment hasn't been created yet.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

echo "==> engine: cargo fmt --check"
cargo fmt --check --manifest-path engine/Cargo.toml

echo "==> engine: cargo clippy"
(cd engine && cargo clippy --all-targets -- -D warnings)

echo "==> engine: cargo test"
(cd engine && cargo test)

echo "==> lab: cargo fmt --check"
cargo fmt --check --manifest-path lab/Cargo.toml

echo "==> lab: cargo clippy"
(cd lab && cargo clippy --all-targets -- -D warnings)

echo "==> lab: cargo test"
(cd lab && cargo test)

echo "==> training: ruff check"
(cd training && uv run ruff check .)

echo "==> training: pytest"
(cd training && uv run pytest)

echo "==> bridge: ruff check"
(cd bridge && uv run ruff check .)

echo "==> frontend: lint"
npm --prefix frontend run lint

echo "==> frontend: test"
npm --prefix frontend test

echo "==> frontend: build"
npm --prefix frontend run build

echo
echo "All checks passed."
