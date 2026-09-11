#!/usr/bin/env bash
set -euo pipefail
CRATE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="$CRATE_DIR/.venv"

if [[ "$(uname -s)" == "Darwin" ]]; then
    BACKEND="mac (mps)"
    TORCH_SPEC="torch==2.9.0"
    INDEX_ARGS=()
elif command -v rocminfo >/dev/null 2>&1 || [[ -d /opt/rocm ]]; then
    BACKEND="rocm"
    TORCH_SPEC="torch==2.9.0+rocm6.4"
    INDEX_ARGS=(--index-url https://download.pytorch.org/whl/rocm6.4)
elif command -v nvidia-smi >/dev/null 2>&1; then
    BACKEND="cuda"
    TORCH_SPEC="torch==2.9.0"
    INDEX_ARGS=()
else
    BACKEND="cpu"
    TORCH_SPEC="torch==2.9.0+cpu"
    INDEX_ARGS=(--index-url https://download.pytorch.org/whl/cpu)
fi

echo "mamba_mcts_batched_uci setup: detected backend = $BACKEND"

if ! command -v uv >/dev/null 2>&1; then
    echo "uv not found -- install it from https://docs.astral.sh/uv/ first" >&2
    exit 1
fi

if [[ ! -d "$VENV" ]]; then
    uv venv "$VENV" --python 3.14
fi
uv pip install --python "$VENV/bin/python" "$TORCH_SPEC" "${INDEX_ARGS[@]}"

export MAMBA_VENV_PYTHON="$VENV/bin/python"
export LIBTORCH_USE_PYTORCH=1
export LD_LIBRARY_PATH="$("$VENV/bin/python" -c 'import torch, os; print(os.path.join(os.path.dirname(torch.__file__), "lib"))'):${LD_LIBRARY_PATH:-}"
export PATH="$VENV/bin:$PATH"

cd "$CRATE_DIR"
cargo build --release

echo "build complete -- run the engine with: $CRATE_DIR/run_uci.sh"
