import { LAB_BASE_URL, type Color } from "./labClient";

export type GamePhase = "opening" | "middlegame" | "endgame";
export interface AnalysisRun {
  id: number;
  engine: string;
  nodes_per_position: number | null;
  multipv: number | null;
  schema_version: number;
  created_at: number;
  configuration: string | null;
}
export interface MoveStats {
  bee_moves: number;
  cp_moves: number;
  avg_cpl: number | null;
  over_100: number;
  over_200: number;
  over_400: number;
  games_over_100: number;
  games_over_200: number;
  games_over_400: number;
  mate_moves: number;
  score_disagreements: number;
}
export interface ReviewMove {
  id: number;
  analysis_run_id: number;
  game_id: string;
  ply: number;
  fen_before: string;
  played_move: string;
  best_move: string | null;
  eval_before_cp: number | null;
  eval_after_cp: number | null;
  eval_played_cp: number | null;
  mate_played: number | null;
  played_pv: string | null;
  centipawn_loss: number | null;
  mate_before: number | null;
  mate_after: number | null;
  phase: GamePhase;
  mover_color: Color | null;
  is_bee: boolean | null;
  pv: string | null;
  white: string | null;
  black: string | null;
  result: string | null;
  original_game_url: string | null;
}
export interface AnalysisReport {
  run: AnalysisRun;
  summary: MoveStats & { games_analyzed: number; phases: Array<MoveStats & { phase: GamePhase }> };
  matching_moves: number;
  moves: ReviewMove[];
  losses: Array<{ worst_move: ReviewMove; total_cpl: number; worst_move_share: number | null }>;
}
export interface AnalysisFilter {
  phase?: GamePhase;
  losses_only?: boolean;
  unique_games?: boolean;
  over_cp?: number;
  offset?: number;
  limit?: number;
}

async function read<T>(path: string): Promise<T> {
  const response = await fetch(`${LAB_BASE_URL}/api/analysis/${path}`);
  if (!response.ok) throw new Error((await response.text()) || `Analysis request failed (${response.status}).`);
  return response.json() as Promise<T>;
}

export const listAnalysisRuns = () => read<AnalysisRun[]>("runs");
export const getAnalysisMove = (run: number, id: number) => read<ReviewMove>(`runs/${run}/moves/${id}`);
export function getAnalysisReport(run: number, filter: AnalysisFilter = {}) {
  const params = new URLSearchParams();
  for (const [key, value] of Object.entries(filter)) {
    if (value !== undefined) params.set(key, String(value));
  }
  return read<AnalysisReport>(`runs/${run}?${params}`);
}
