# mamba

Native Rust engines for Bee-Mamba's PUCT MCTS.

- `mamba_mcts_native/` -- PyO3 extension: tree/PUCT/move-gen in Rust, NN inference stays in Python/PyTorch (`training/src/bee_training/chess_mamba/play_mcts.py`). Built automatically by `training`'s `uv sync`.
- `mamba_mcts_batched_uci/` -- standalone UCI engine binary: same wave-batched search, but NN inference runs natively in Rust via `tch`/libtorch too, no Python process. Point a GUI or lichess-bot straight at it.

## Running `mamba_mcts_batched_uci`

```bash
./mamba_mcts_batched_uci/setup.sh   # detects mac/cuda/rocm/cpu, installs matching torch, builds
./mamba_mcts_batched_uci/run_uci.sh # speaks UCI on stdin/stdout
```

`setup.sh` is idempotent -- rerun it after switching GPU vendor or updating `models/chess_mamba_dynamic.pt`.
