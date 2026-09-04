"""Play one self-play game and record a labeled position per ply."""

from __future__ import annotations

import random
from dataclasses import dataclass, replace

import chess
import chess.engine
import chess.pgn

from bee_training.dataset.adjudication import Adjudicator
from bee_training.dataset.config import SelfPlayConfig
from bee_training.dataset.openings import random_opening, sample_opening
from bee_training.dataset.schema import SCHEMA_VERSION, GameRecord, PositionRecord


def _limit_for(config: SelfPlayConfig) -> chess.engine.Limit:
    if config.limit_kind == "nodes":
        return chess.engine.Limit(nodes=int(config.limit_value))
    if config.limit_kind == "time":
        return chess.engine.Limit(time=float(config.limit_value))
    if config.limit_kind == "depth":
        return chess.engine.Limit(depth=int(config.limit_value))
    raise ValueError(f"Unknown limit_kind: {config.limit_kind!r}")


def _node_time_depth(config: SelfPlayConfig) -> tuple[int | None, float | None, int | None]:
    nodes = int(config.limit_value) if config.limit_kind == "nodes" else None
    time_s = float(config.limit_value) if config.limit_kind == "time" else None
    depth = int(config.limit_value) if config.limit_kind == "depth" else None
    return nodes, time_s, depth


@dataclass
class GameOutcome:
    positions: list[PositionRecord]
    game: GameRecord
    pgn_text: str


def play_one_game(
    engine: chess.engine.SimpleEngine,
    config: SelfPlayConfig,
    game_id: str,
    rng: random.Random,
    book: list[str] | None,
) -> GameOutcome:
    if book:
        board = sample_opening(book, rng)
        opening_source = "book"
    else:
        board = random_opening(rng, config.opening_plies)
        opening_source = "random_plies"

    pgn_game = chess.pgn.Game()
    if board.fen() != chess.STARTING_FEN:
        pgn_game.setup(board)
    pgn_node = pgn_game

    limit = _limit_for(config)
    nodes, time_s, depth = _node_time_depth(config)
    adjudicator = Adjudicator(
        resign_cp=config.resign_cp,
        resign_plies=config.resign_plies,
        draw_cp=config.draw_cp,
        draw_plies=config.draw_plies,
        draw_min_ply=config.draw_min_ply,
    )

    positions: list[PositionRecord] = []
    termination = "max_plies"
    result = "1/2-1/2"
    ply = 0

    while ply < config.max_plies:
        if board.is_game_over(claim_draw=True):
            outcome = board.outcome(claim_draw=True)
            result = outcome.result()
            termination = "checkmate" if outcome.termination == chess.Termination.CHECKMATE else "draw_rule"
            break

        play_result = engine.play(board, limit, info=chess.engine.INFO_ALL)
        move = play_result.move
        if move is None:
            termination = "draw_rule"
            break
        info = play_result.info

        score = info.get("score")
        eval_cp: int | None = None
        eval_mate: int | None = None
        if score is not None:
            mover_score = score.pov(board.turn)
            if mover_score.is_mate():
                eval_mate = mover_score.mate()
            else:
                eval_cp = mover_score.score()

        pv_moves = info.get("pv", [])
        pv_uci = [m.uci() for m in pv_moves]

        positions.append(
            PositionRecord(
                schema_version=SCHEMA_VERSION,
                game_id=game_id,
                ply=ply,
                fen=board.fen(),
                side_to_move="w" if board.turn == chess.WHITE else "b",
                eval_cp=eval_cp,
                eval_mate=eval_mate,
                depth=info.get("depth", 0),
                best_move=move.uci(),
                pv=pv_uci,
                game_result="",  # filled in below once the final result is known
                stockfish_version=config.stockfish_version,
            )
        )

        board.push(move)
        pgn_node = pgn_node.add_variation(move)
        ply += 1

        if score is not None:
            adjudication = adjudicator.observe(score, ply)
            if adjudication is not None:
                termination, result = adjudication
                break

    # game_result isn't known until the loop above ends, so backfill it now
    # (PositionRecord is frozen, hence replace() rather than mutation).
    positions = [replace(p, game_result=result) for p in positions]

    pgn_game.headers["Event"] = "bee-training self-play"
    pgn_game.headers["Result"] = result
    pgn_game.headers["White"] = f"Stockfish {config.stockfish_version}"
    pgn_game.headers["Black"] = f"Stockfish {config.stockfish_version}"
    exporter = chess.pgn.StringExporter(headers=True, variations=False, comments=False)
    pgn_text = pgn_game.accept(exporter)

    game_record = GameRecord(
        schema_version=SCHEMA_VERSION,
        game_id=game_id,
        result=result,
        termination=termination,
        ply_count=ply,
        opening_source=opening_source,
        stockfish_version=config.stockfish_version,
        node_limit=nodes,
        time_limit_s=time_s,
        depth_limit=depth,
        seed=config.seed,
    )
    return GameOutcome(positions=positions, game=game_record, pgn_text=pgn_text)
