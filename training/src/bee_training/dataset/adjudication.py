"""Early game termination so self-play games don't run to the 50/75-move limit
once a position is already decisively won or drawn.

Thresholds are constructor parameters, not hardcoded: eval noise scales with
the chosen search node/time limit, so the right thresholds depend on that
choice.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import chess.engine

# (termination reason, final game result)
Adjudication = tuple[str, str]


@dataclass
class Adjudicator:
    resign_cp: int
    resign_plies: int
    draw_cp: int
    draw_plies: int
    draw_min_ply: int

    def __post_init__(self) -> None:
        window = max(self.resign_plies, self.draw_plies)
        self._white_cp_history: deque[int] = deque(maxlen=window)
        self._any_mate_seen = False

    def observe(self, score: chess.engine.PovScore, ply: int) -> Adjudication | None:
        """Feed the score (from White's POV) for the move just searched."""
        white_score = score.pov(True)  # chess.WHITE
        mate = white_score.mate()
        if mate is not None:
            self._any_mate_seen = True
            self._white_cp_history.append(100_000 if mate > 0 else -100_000)
            return None  # A forced mate is already winning; let it play out to checkmate.

        cp = white_score.score()
        if cp is None:
            return None
        self._white_cp_history.append(cp)

        resign = self._check_resign()
        if resign:
            return resign
        return self._check_draw(ply)

    def _check_resign(self) -> Adjudication | None:
        if self._any_mate_seen or len(self._white_cp_history) < self.resign_plies:
            return None
        recent = list(self._white_cp_history)[-self.resign_plies :]
        if all(s >= self.resign_cp for s in recent):
            return "adjudicated_resign", "1-0"
        if all(s <= -self.resign_cp for s in recent):
            return "adjudicated_resign", "0-1"
        return None

    def _check_draw(self, ply: int) -> Adjudication | None:
        if ply < self.draw_min_ply or len(self._white_cp_history) < self.draw_plies:
            return None
        recent = list(self._white_cp_history)[-self.draw_plies :]
        if all(abs(cp) <= self.draw_cp for cp in recent):
            return "adjudicated_draw", "1/2-1/2"
        return None
