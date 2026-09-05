"""`bee-training` CLI: Stockfish self-play dataset generation."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from bee_training.dataset.config import SelfPlayConfig
from bee_training.dataset.generate import run as run_generate
from bee_training.dataset.stockfish_fetch import ensure_stockfish


def _add_fetch_stockfish_parser(subparsers) -> None:
    p = subparsers.add_parser("fetch-stockfish", help="Download and cache the Stockfish binary.")
    p.add_argument("--version", default="latest", help="Release tag (e.g. sf_18) or 'latest'.")
    p.add_argument("--cache-dir", default=None, help="Override the default cache directory.")
    p.set_defaults(func=_cmd_fetch_stockfish)


def _cmd_fetch_stockfish(args: argparse.Namespace) -> None:
    cache_dir = Path(args.cache_dir) if args.cache_dir else None
    path, tag = ensure_stockfish(version=args.version, cache_dir=cache_dir)
    print(f"Stockfish {tag} ready at {path}")


def _add_generate_parser(subparsers) -> None:
    p = subparsers.add_parser("generate", help="Generate self-play games.")
    p.add_argument("--games", type=int, required=True)
    p.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    p.add_argument("--limit-kind", choices=["nodes", "time", "depth"], default="nodes")
    p.add_argument("--limit-value", type=float, default=25_000.0)
    p.add_argument("--opening-book", default=None, help="Path to an EPD or PGN opening book.")
    p.add_argument("--opening-plies", type=int, default=8, help="Used only without --opening-book.")
    p.add_argument("--resign-cp", type=int, default=1000)
    p.add_argument("--resign-plies", type=int, default=8)
    p.add_argument("--draw-cp", type=int, default=10)
    p.add_argument("--draw-plies", type=int, default=8)
    p.add_argument("--draw-min-ply", type=int, default=40)
    p.add_argument("--max-plies", type=int, default=300)
    p.add_argument("--stockfish-version", default="latest")
    p.add_argument("--output-dir", default="data", help="Relative to the current directory (typically training/).")
    p.add_argument("--run-id", required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-pgn", action="store_true", help="Skip PGN output; JSONL only.")
    p.add_argument("--fresh", action="store_true", help="Discard any existing progress for this run-id.")
    p.set_defaults(func=_cmd_generate)


def _cmd_generate(args: argparse.Namespace) -> None:
    config = SelfPlayConfig(
        games=args.games,
        workers=args.workers,
        limit_kind=args.limit_kind,
        limit_value=args.limit_value,
        opening_book=args.opening_book,
        opening_plies=args.opening_plies,
        resign_cp=args.resign_cp,
        resign_plies=args.resign_plies,
        draw_cp=args.draw_cp,
        draw_plies=args.draw_plies,
        draw_min_ply=args.draw_min_ply,
        max_plies=args.max_plies,
        stockfish_version=args.stockfish_version,
        output_dir=args.output_dir,
        run_id=args.run_id,
        seed=args.seed,
        emit_pgn=not args.no_pgn,
    )
    run_generate(config, fresh=args.fresh)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="bee-training")
    subparsers = parser.add_subparsers(dest="command", required=True)
    _add_fetch_stockfish_parser(subparsers)
    _add_generate_parser(subparsers)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv if argv is not None else sys.argv[1:])
    args.func(args)


if __name__ == "__main__":
    main()
