#!/usr/bin/env bash
# Builds the Bee engine in release mode (target/release/bee).
#
# Output lives at the repo root's target/, not engine/target/, since
# engine/ is a member of the root Cargo workspace (see /Cargo.toml,
# introduced by #68/lab/) -- Cargo shares one target/ directory across
# all workspace members by default.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if ! command -v cargo >/dev/null 2>&1; then
  echo "cargo not found. Install Rust: https://rustup.rs" >&2
  exit 1
fi

# Cargo.lock is version 4, which needs cargo 1.78+. Distro packages are often
# older, and cargo's own error for this ("requires -Znext-lockfile-bump") does
# not say so.
CARGO_VERSION="$(cargo --version | cut -d' ' -f2)"
CARGO_MINOR="$(echo "$CARGO_VERSION" | cut -d. -f2)"
if [ "$(echo "$CARGO_VERSION" | cut -d. -f1)" -eq 1 ] && [ "$CARGO_MINOR" -lt 78 ]; then
  echo "cargo $CARGO_VERSION is too old (need 1.78+ for the v4 lockfile)." >&2
  echo "Found: $(command -v cargo)" >&2
  if [ -x "$HOME/.cargo/bin/cargo" ]; then
    echo "rustup is installed but not on PATH here. Run:" >&2
    echo "  source \"\$HOME/.cargo/env\"" >&2
  else
    echo "Install a current toolchain: https://rustup.rs" >&2
  fi
  exit 1
fi

cargo build --release --manifest-path "$ROOT/Cargo.toml" -p bee-engine --bin bee

echo "built: $ROOT/target/release/bee"
