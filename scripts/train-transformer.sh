#!/usr/bin/env bash
# Trains the pure-attention (n_ssm=0) baseline ChessMamba config, for
# comparison against the hybrid/pure-SSM runs from train-mamba.sh.
#
# Data-generation only, not training, is what train-mamba.sh's `generate`
# call owns -- don't run two `generate` processes against the same
# run-id/manifest concurrently, they'll race each other's shard writes.
# Run train-mamba.sh (or `bee-training generate` directly) separately if
# you need more data; this script just trains on whatever's already on
# disk under data/<run-id>/.
#
# Ctrl-C stops cleanly; re-running resumes from the last checkpoint.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT/training"
mkdir -p logs

RUN_ID="${RUN_ID:-main-dawg-fast}"
TOTAL_STEPS="${TOTAL_STEPS:-200000}"

echo "==> starting transformer (pure-attention) training (checkpoint dir: checkpoints/${RUN_ID}-transformer)"
uv run python3 -m bee_training.chess_mamba.train \
  --data-glob "data/${RUN_ID}/shards/*.positions.jsonl" \
  --n-ssm 0 \
  --total-steps "$TOTAL_STEPS" \
  --checkpoint-dir "checkpoints/${RUN_ID}-transformer" \
  > "logs/train-transformer.log" 2>&1 &
TRAIN_PID=$!

cleanup() { kill "$TRAIN_PID" 2>/dev/null || true; }
trap cleanup EXIT INT TERM

echo "==> log: training/logs/train-transformer.log"
echo "==> tail -f training/logs/train-transformer.log"
wait "$TRAIN_PID"
