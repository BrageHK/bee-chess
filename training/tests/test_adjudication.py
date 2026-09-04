import chess
import chess.engine

from bee_training.dataset.adjudication import Adjudicator


def _cp_score(cp: int) -> chess.engine.PovScore:
    return chess.engine.PovScore(chess.engine.Cp(cp), chess.WHITE)


def _mate_score(moves: int) -> chess.engine.PovScore:
    return chess.engine.PovScore(chess.engine.Mate(moves), chess.WHITE)


def _adjudicator(**overrides) -> Adjudicator:
    defaults = {"resign_cp": 1000, "resign_plies": 4, "draw_cp": 10, "draw_plies": 4, "draw_min_ply": 10}
    defaults.update(overrides)
    return Adjudicator(**defaults)


def test_no_adjudication_for_balanced_eval_before_min_ply() -> None:
    a = _adjudicator()
    result = None
    for ply in range(1, 5):
        result = a.observe(_cp_score(5), ply)
    assert result is None  # draw_min_ply=10 not yet reached


def test_draw_adjudicated_after_sustained_near_zero_eval() -> None:
    a = _adjudicator()
    result = None
    for ply in range(1, 15):
        result = a.observe(_cp_score(3), ply)
    assert result == ("adjudicated_draw", "1/2-1/2")


def test_resign_adjudicated_when_white_winning_decisively() -> None:
    a = _adjudicator()
    result = None
    for ply in range(1, 6):
        result = a.observe(_cp_score(1500), ply)
    assert result == ("adjudicated_resign", "1-0")


def test_resign_adjudicated_when_black_winning_decisively() -> None:
    a = _adjudicator()
    result = None
    for ply in range(1, 6):
        result = a.observe(_cp_score(-1500), ply)
    assert result == ("adjudicated_resign", "0-1")


def test_forced_mate_does_not_trigger_resign_adjudication() -> None:
    a = _adjudicator(resign_plies=1)
    result = a.observe(_mate_score(3), ply=20)
    assert result is None


def test_single_dip_below_threshold_resets_resign_window() -> None:
    a = _adjudicator()
    result = None
    for ply, cp in enumerate([1500, 1500, 500, 1500, 1500], start=1):
        result = a.observe(_cp_score(cp), ply)
    assert result is None
