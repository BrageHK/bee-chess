"""Versioned record schema for self-play datasets.

Per `CONTRIBUTING.md`, Python-generated data formats must be versioned.
`SCHEMA_VERSION` must be bumped on any incompatible change to `PositionRecord`
or `GameRecord`.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1


@dataclass(frozen=True)
class PositionRecord:
    """One labeled training position: a FEN plus Stockfish's evaluation of it."""

    schema_version: int
    game_id: str
    ply: int
    fen: str
    side_to_move: str  # "w" | "b"
    eval_cp: int | None
    eval_mate: int | None
    depth: int
    best_move: str  # UCI, e.g. "e2e4"
    pv: list[str]
    game_result: str  # "1-0" | "0-1" | "1/2-1/2"
    stockfish_version: str

    def to_json(self) -> str:
        return json.dumps(asdict(self), separators=(",", ":"))

    @staticmethod
    def from_dict(data: dict[str, Any]) -> PositionRecord:
        return PositionRecord(**data)


@dataclass(frozen=True)
class GameRecord:
    """One row of per-game metadata, denormalized alongside the position shard."""

    schema_version: int
    game_id: str
    result: str  # "1-0" | "0-1" | "1/2-1/2"
    termination: str
    ply_count: int
    opening_source: str
    stockfish_version: str
    node_limit: int | None
    time_limit_s: float | None
    depth_limit: int | None
    seed: int

    def to_json(self) -> str:
        return json.dumps(asdict(self), separators=(",", ":"))

    @staticmethod
    def from_dict(data: dict[str, Any]) -> GameRecord:
        return GameRecord(**data)


def append_jsonl(path: Path, lines: list[str]) -> None:
    """Append pre-serialized JSON lines to `path`, creating it if needed."""
    if not lines:
        return
    with path.open("a", encoding="utf-8") as f:
        for line in lines:
            f.write(line)
            f.write("\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    if not path.exists():
        return records
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records
