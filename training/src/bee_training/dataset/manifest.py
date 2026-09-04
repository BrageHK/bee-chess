"""Resumability state for a self-play generation run.

`manifest.json` is the single source of truth for "what's done": each
worker's assigned game-index range and how far it has completed. It's
fingerprinted against the run's `SelfPlayConfig` so a resume can never
silently mix data generated under different settings into one dataset.
"""

from __future__ import annotations

import json
import os
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass, field
from pathlib import Path

from bee_training.dataset.config import SelfPlayConfig


class ManifestMismatchError(RuntimeError):
    pass


@dataclass
class WorkerState:
    start: int  # inclusive
    end: int  # exclusive
    last_completed_index: int = -1  # -1 means nothing completed yet

    def remaining(self) -> range:
        return range(max(self.start, self.last_completed_index + 1), self.end)


@dataclass
class Manifest:
    run_id: str
    config_fingerprint: str
    total_games_target: int
    workers: dict[str, WorkerState] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "run_id": self.run_id,
            "config_fingerprint": self.config_fingerprint,
            "total_games_target": self.total_games_target,
            "workers": {
                k: {"start": w.start, "end": w.end, "last_completed_index": w.last_completed_index}
                for k, w in self.workers.items()
            },
        }

    @staticmethod
    def from_dict(data: dict) -> Manifest:
        workers = {k: WorkerState(**v) for k, v in data["workers"].items()}
        return Manifest(
            run_id=data["run_id"],
            config_fingerprint=data["config_fingerprint"],
            total_games_target=data["total_games_target"],
            workers=workers,
        )

    def completed_count(self) -> int:
        return sum(
            w.last_completed_index - w.start + 1
            for w in self.workers.values()
            if w.last_completed_index >= w.start
        )


def manifest_path(config: SelfPlayConfig) -> Path:
    return config.run_dir() / "manifest.json"


def build_fresh_manifest(config: SelfPlayConfig) -> Manifest:
    per_worker = config.games // config.workers
    remainder = config.games % config.workers
    workers: dict[str, WorkerState] = {}
    cursor = 0
    for i in range(config.workers):
        count = per_worker + (1 if i < remainder else 0)
        workers[str(i)] = WorkerState(start=cursor, end=cursor + count)
        cursor += count
    return Manifest(
        run_id=config.run_id,
        config_fingerprint=config.fingerprint(),
        total_games_target=config.games,
        workers=workers,
    )


def load(config: SelfPlayConfig) -> Manifest:
    return Manifest.from_dict(json.loads(manifest_path(config).read_text(encoding="utf-8")))


def load_or_create(config: SelfPlayConfig, *, force_fresh: bool = False) -> Manifest:
    path = manifest_path(config)
    if path.exists() and not force_fresh:
        existing = load(config)
        if existing.config_fingerprint != config.fingerprint():
            raise ManifestMismatchError(
                f"Existing manifest at {path} was generated with different settings "
                "(node limit / opening book / stockfish version / etc). Refusing to "
                "resume: pick a new --run-id, or pass --fresh to overwrite."
            )
        if existing.total_games_target != config.games or len(existing.workers) != config.workers:
            raise ManifestMismatchError(
                f"Existing manifest at {path} targets {existing.total_games_target} games across "
                f"{len(existing.workers)} workers, but this invocation requested "
                f"{config.games} games across {config.workers} workers. Refusing to resume with "
                "a different shape: pick a new --run-id, or pass --fresh to overwrite."
            )
        return existing
    manifest = build_fresh_manifest(config)
    save(config, manifest)
    return manifest


def save(config: SelfPlayConfig, manifest: Manifest) -> None:
    path = manifest_path(config)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(".json.tmp")
    tmp_path.write_text(json.dumps(manifest.to_dict(), indent=2), encoding="utf-8")
    os.replace(tmp_path, path)


def mark_completed(
    config: SelfPlayConfig,
    worker_id: str,
    game_index: int,
    lock: AbstractContextManager | None = None,
) -> None:
    """Load, update, and atomically save the manifest after one completed game.

    The manifest file is shared across all worker processes, each of which
    updates only its own entry -- but the read-modify-write is still a
    classic lost-update race if two workers do it concurrently (the second
    writer's read predates the first writer's update, so its write silently
    reverts it). `lock` must be a cross-process lock (e.g. `multiprocessing.
    Lock()`) shared by every worker to serialize this critical section.
    """
    cm = lock if lock is not None else nullcontext()
    with cm:
        manifest = load(config)
        manifest.workers[worker_id].last_completed_index = game_index
        save(config, manifest)
