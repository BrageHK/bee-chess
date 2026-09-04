"""Configuration for a self-play dataset generation run."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class SelfPlayConfig:
    games: int
    workers: int
    limit_kind: str  # "nodes" | "time" | "depth"
    limit_value: float
    opening_book: str | None
    opening_plies: int
    resign_cp: int
    resign_plies: int
    draw_cp: int
    draw_plies: int
    draw_min_ply: int
    max_plies: int
    stockfish_version: str  # resolved tag, e.g. "sf_18" -- never "latest"
    output_dir: str
    run_id: str
    seed: int
    emit_pgn: bool = True

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(data: dict) -> SelfPlayConfig:
        return SelfPlayConfig(**data)

    def fingerprint(self) -> str:
        """Stable hash over everything that determines dataset compatibility.

        Excludes `games`/`workers`/`run_id`/`output_dir`, which only affect
        how much work is done and where, not what a resumed run's existing
        records mean.
        """
        stable = {
            "limit_kind": self.limit_kind,
            "limit_value": self.limit_value,
            "opening_book": self.opening_book,
            "opening_plies": self.opening_plies,
            "resign_cp": self.resign_cp,
            "resign_plies": self.resign_plies,
            "draw_cp": self.draw_cp,
            "draw_plies": self.draw_plies,
            "draw_min_ply": self.draw_min_ply,
            "max_plies": self.max_plies,
            "stockfish_version": self.stockfish_version,
            "seed": self.seed,
            "emit_pgn": self.emit_pgn,
        }
        payload = json.dumps(stable, sort_keys=True).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def run_dir(self) -> Path:
        return Path(self.output_dir) / self.run_id

    def save(self) -> None:
        path = self.run_dir() / "config.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True), encoding="utf-8")

    @staticmethod
    def load(run_dir: Path) -> SelfPlayConfig:
        data = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
        return SelfPlayConfig.from_dict(data)
