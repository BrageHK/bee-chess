import type { AnalysisReport, AnalysisRun, ReviewMove } from "./analysisClient";
import { INITIAL_FEN } from "chessops/fen";

export const analysisRunFixture: AnalysisRun = { id: 1, engine: "Stockfish test", nodes_per_position: 100000,
  multipv: 1, schema_version: 2, created_at: 0, configuration: '{"players":["Bee"]}' };
export const reviewMoveFixture: ReviewMove = {
  id: 7, analysis_run_id: 1, game_id: "abc123", ply: 0, fen_before: INITIAL_FEN,
  played_move: "g2g4", best_move: "e2e4", eval_before_cp: 28, eval_after_cp: -384,
  eval_played_cp: null, mate_played: null, played_pv: null,
  centipawn_loss: 412, mate_before: null, mate_after: null, phase: "opening", mover_color: "white",
  is_bee: true, pv: "e2e4 e7e5 g1f3", white: "Bee", black: "Opponent", result: "0-1",
  original_game_url: "https://lichess.org/abc123#1",
};
const stats = { bee_moves: 10, cp_moves: 10, avg_cpl: 61, over_100: 4, over_200: 2, over_400: 1, mate_moves: 0, score_disagreements: 0, games_over_100: 2, games_over_200: 1, games_over_400: 1 };
export const analysisReportFixture: AnalysisReport = {
  run: analysisRunFixture,
  summary: { ...stats, games_analyzed: 2, phases: [
    { ...stats, phase: "opening", avg_cpl: 32 },
    { ...stats, phase: "middlegame", avg_cpl: 78 },
    { ...stats, phase: "endgame", avg_cpl: 49 },
  ] },
  matching_moves: 1, moves: [reviewMoveFixture],
  losses: [{ worst_move: reviewMoveFixture, total_cpl: 500, worst_move_share: 0.824 }],
};
