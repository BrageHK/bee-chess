#!/usr/bin/env bash
# Runs self-play data generation and ChessMamba training in parallel.
# Ctrl-C stops both cleanly. Re-running this script resumes both: the
# generator picks up from its manifest, train.py resumes from its last
# checkpoint -- neither restarts from scratch. train.py also waits for
# the generator to produce its first game before it starts training, so
# launching both at once (as this script does) is safe.
#
# Defaults target main-dawg-fast (5000 nodes/move, 10M games) -- the
# original main-dawg (25000 nodes/move) is capped at 100k games in its
# own manifest and can't be raised in place: GAMES/LIMIT_VALUE/WORKERS
# are part of what a run-id's manifest locks in on first use, and
# `--fresh` doesn't delete existing shard files, it just resets progress
# tracking and starts appending games from index 0 again into the *same*
# files -- for an existing run-id with real data already in it, that
# risks duplicate/conflicting game_ids, not a clean restart. Use a new
# RUN_ID for a different games/node/worker count instead, e.g.:
#   RUN_ID=main-dawg GAMES=100000 LIMIT_VALUE=25000 ./scripts/train-mamba.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT/training"
mkdir -p logs

RUN_ID="${RUN_ID:-main-dawg-fast}"
GAMES="${GAMES:-10000000}"
GENERATE_WORKERS="${GENERATE_WORKERS:-32}"
LIMIT_VALUE="${LIMIT_VALUE:-5000}"
TOTAL_STEPS="${TOTAL_STEPS:-200000}"

echo "==> starting data generation (run-id: $RUN_ID, $GAMES games, $LIMIT_VALUE nodes/move)"
uv run bee-training generate \
  --games "$GAMES" \
  --run-id "$RUN_ID" \
  --workers "$GENERATE_WORKERS" \
  --limit-kind nodes \
  --limit-value "$LIMIT_VALUE" \
  --opening-book books/UHO_4060_v2.epd \
  --stockfish-version sf_18 \
  > logs/generate.log 2>&1 &
GENERATE_PID=$!

echo "==> starting training (data: data/$RUN_ID/, checkpoint dir: checkpoints/$RUN_ID)"
uv run python3 -m bee_training.chess_mamba.train \
  --data-glob "data/$RUN_ID/shards/*.positions.jsonl" \
  --total-steps "$TOTAL_STEPS" \
  --checkpoint-dir "checkpoints/$RUN_ID" \
  > logs/train.log 2>&1 &
TRAIN_PID=$!

cleanup() { kill "$GENERATE_PID" "$TRAIN_PID" 2>/dev/null || true; }
trap cleanup EXIT INT TERM

echo "==> logs: training/logs/generate.log, training/logs/train.log"
echo "==> tail -f training/logs/generate.log training/logs/train.log"
wait "$GENERATE_PID" "$TRAIN_PID"
