"""Multiprocessing orchestrator for a self-play dataset generation run."""

from __future__ import annotations

import multiprocessing
import signal
import sys
import time
from dataclasses import replace
from pathlib import Path

from tqdm import tqdm

from bee_training.dataset import manifest as manifest_mod
from bee_training.dataset.config import SelfPlayConfig
from bee_training.dataset.stockfish_fetch import ensure_stockfish
from bee_training.dataset.worker import run_worker


def _worker_entrypoint(worker_id: str, config: SelfPlayConfig, stockfish_path: str, lock) -> None:
    run_worker(worker_id, config, Path(stockfish_path), lock)


def run(config: SelfPlayConfig, *, fresh: bool = False) -> None:
    """Generate `config.games` self-play games. Auto-resumes an existing run at
    `config.run_dir()` whose settings match; raises `ManifestMismatchError`
    (via `manifest_mod.load_or_create`) if they don't. `fresh=True` discards
    any existing manifest/progress for this run_id and starts over.
    """
    print(f"Resolving Stockfish {config.stockfish_version!r}...", file=sys.stderr)
    stockfish_path, resolved_tag = ensure_stockfish(version=config.stockfish_version)
    if config.stockfish_version in (None, "latest"):
        config = replace(config, stockfish_version=resolved_tag)
    print(f"Using Stockfish {resolved_tag} at {stockfish_path}", file=sys.stderr)

    manifest = manifest_mod.load_or_create(config, force_fresh=fresh)
    config.save()

    lock = multiprocessing.Manager().Lock()
    processes = []
    for worker_id in manifest.workers:
        p = multiprocessing.Process(
            target=_worker_entrypoint,
            args=(worker_id, config, str(stockfish_path), lock),
            name=f"selfplay-worker-{worker_id}",
        )
        p.start()
        processes.append(p)

    # On macOS/Windows, multiprocessing's default "spawn" start method means
    # worker processes don't share this process's argv, so killing just the
    # parent (e.g. `kill <pid>`, not an interactive Ctrl+C) would otherwise
    # leave them running as orphans. Translate SIGTERM into the same
    # KeyboardInterrupt-driven shutdown path Ctrl+C already takes.
    def _raise_keyboard_interrupt(signum, frame):
        raise KeyboardInterrupt

    previous_sigterm_handler = signal.signal(signal.SIGTERM, _raise_keyboard_interrupt)
    try:
        _monitor(config, processes)
    except KeyboardInterrupt:
        print(f"[{config.run_id}] interrupted; stopping workers...", file=sys.stderr)
        for p in processes:
            p.terminate()
        raise
    finally:
        signal.signal(signal.SIGTERM, previous_sigterm_handler)
        for p in processes:
            p.join()


def _monitor(
    config: SelfPlayConfig,
    processes: list[multiprocessing.Process],
    interval_s: float = 10.0,
    poll_s: float = 0.5,
) -> None:
    """Drive a tqdm progress bar, refreshed roughly every `interval_s`, but
    return promptly (within `poll_s`) once all workers exit rather than
    always waiting a full `interval_s` -- which matters for near-instant
    resumed/no-op runs.
    """
    initial = manifest_mod.load(config).completed_count()
    with tqdm(total=config.games, initial=initial, unit="game", desc=config.run_id, file=sys.stderr) as bar:
        elapsed = 0.0
        while any(p.is_alive() for p in processes):
            time.sleep(poll_s)
            elapsed += poll_s
            if elapsed >= interval_s:
                elapsed = 0.0
                bar.n = manifest_mod.load(config).completed_count()
                bar.refresh()
        bar.n = manifest_mod.load(config).completed_count()
        bar.refresh()
