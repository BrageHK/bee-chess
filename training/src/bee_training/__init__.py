"""Bee Chess training and dataset generation.

`bee_training.dataset` generates versioned (FEN, Stockfish eval, best move)
training records via parallel Stockfish self-play (see `bee-training
generate --help`). Model-manifest schemas and the PyTorch training loop
land in follow-up PRs. Per `docs/adr/0001-v1-engine-architecture.md`,
Python owns training and dataset generation; the Rust engine consumes
exported model artifacts through a versioned contract rather than
depending on this package directly.
"""

__version__ = "0.1.0"
