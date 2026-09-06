"""
Export a trained `ChessMamba` checkpoint to ONNX.

Every line/family shape in geometry.py is static (fixed 64 squares, L<=8 per
ray), so tracing (not scripting) captures a fixed op graph with no
data-dependent control flow -- *except* `mamba_core.py`'s default "pscan"
scan backend, whose custom autograd.Function relies on chained in-place view
mutations that the tracer does not reliably capture (a traced pscan call can
come out of `torch.onnx.export` with its `A` input silently dropped, since
the exporter's dead-code elimination loses the data dependency through the
mutation chain -- confirmed by exporting `mambapy.pscan.pscan` in isolation
and finding the ONNX graph only kept 5 nodes and one input). So this module
loads the checkpoint with `scan_backend="sequential"` (see mamba_core.py)
instead: a plain O(L) Python-loop scan with no in-place ops, mathematically
identical to pscan (same (Abar, BX) -> hs contract) and just as fast at this
project's tiny L, but one tracing actually captures correctly.

The resulting graph is exactly what onnxruntime-web's WASM backend would
execute in a browser, so one .onnx file covers both the "onnx" and "wasm"
targets: no separate wasm export step exists for a plain feed-forward graph
like this one.

Run as:
  python -m bee_training.chess_mamba.export_onnx --checkpoint checkpoints/main-dawg-fast-puzzles/best.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import chess
import numpy as np
import torch

from bee_training.chess_mamba.encode import encode_fen
from bee_training.chess_mamba.train import TrainConfig, build_model

# A handful of real, structurally distinct positions (startpos, a mid-game
# position, one with en-passant available, one with no castling rights
# left) to verify against -- not random noise. This model's SSM gate (dt)
# is only ever trained on one-hot piece planes; feeding it N(0,1) noise
# drives some activations into the millions (a real instability, unrelated
# to export correctness) where tiny relative floating-point differences
# between backends produce large absolute deltas and make the comparison
# meaningless.
_VERIFY_FENS = [
    chess.STARTING_FEN,
    "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3",
    "rnbqkbnr/ppp1p1pp/8/3pPp2/8/8/PPPP1PPP/RNBQKBNR w KQkq f6 0 3",
    "8/8/8/4k3/8/8/4K3/4R3 w - - 0 1",
]


def load_model(checkpoint_path: Path) -> tuple[torch.nn.Module, TrainConfig]:
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = TrainConfig(**ckpt["config"])
    config.scan_backend = "sequential"  # force the exportable backend regardless of how it was trained
    model = build_model(config)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, config


def export(checkpoint_path: Path, out_path: Path, opset: int = 17, static_batch: bool = False) -> None:
    """`static_batch=True` bakes batch=1 into the graph instead of a dynamic
    axis. Needed for consumers (e.g. burn-onnx's ONNX-to-Rust codegen) that
    can't compile the `Shape`/`Slice` ops a dynamic batch axis forces inside
    `nn.MultiheadAttention`'s internal reshape -- irrelevant for a one-board-
    at-a-time inference site anyway (see mz-web's ChessMamba integration)."""
    model, _config = load_model(checkpoint_path)
    in_dim = 12 + 8  # N_PIECE_TYPES + N_AUX, matches encode.py (n_history=0)
    batch = 1 if static_batch else 2
    dummy = torch.randn(batch, 64, in_dim)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        (dummy,),
        str(out_path),
        input_names=["board_planes"],
        output_names=["policy_logits", "value_logits"],
        dynamic_axes=None if static_batch else {
            "board_planes": {0: "batch"},
            "policy_logits": {0: "batch"},
            "value_logits": {0: "batch"},
        },
        opset_version=opset,
        dynamo=False,
    )
    print(f"exported -> {out_path}")

    _verify(model, out_path, in_dim, static_batch=static_batch)


def _verify(model: torch.nn.Module, onnx_path: Path, in_dim: int, static_batch: bool = False) -> None:
    """Runs the same real positions through PyTorch and onnxruntime and
    checks they agree -- catches silent tracing mistakes (e.g. a branch the
    tracer only saw one side of) that `torch.onnx.export` completing
    without error would not."""
    import onnxruntime as ort

    fens = _VERIFY_FENS[:1] if static_batch else _VERIFY_FENS  # static graph only accepts batch=1
    x = torch.stack([encode_fen(fen) for fen in fens])
    assert x.shape[-1] == in_dim
    with torch.no_grad():
        ref_policy, ref_value = model(x)

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    onnx_policy, onnx_value = sess.run(None, {"board_planes": x.numpy()})

    max_policy_err = np.abs(onnx_policy - ref_policy.numpy()).max()
    max_value_err = np.abs(onnx_value - ref_value.numpy()).max()
    print(f"max abs error  policy: {max_policy_err:.2e}  value: {max_value_err:.2e}")
    assert max_policy_err < 1e-3, f"policy head mismatch: {max_policy_err}"
    assert max_value_err < 1e-3, f"value head mismatch: {max_value_err}"
    print("onnxruntime output matches PyTorch reference.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/main-dawg-fast-puzzles/best.pt"))
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--static-batch", action="store_true",
                         help="bake batch=1 into the graph instead of a dynamic axis (see export()'s docstring)")
    args = parser.parse_args()

    out = args.out or args.checkpoint.with_suffix(".onnx")
    export(args.checkpoint, out, opset=args.opset, static_batch=args.static_batch)


if __name__ == "__main__":
    main()
