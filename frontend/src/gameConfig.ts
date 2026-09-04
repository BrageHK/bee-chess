/**
 * Quick game settings: how strong Stockfish should play, and how long
 * each engine gets to think per move.
 *
 * Deliberately small. No engine selection, no depth/node modes, no real
 * chess clocks, no presets, no persistence -- see the PR description for
 * the full list of what's intentionally left for later.
 */
export interface GameConfig {
  stockfishElo: number;
  moveTimeMs: number;
}

export const DEFAULT_GAME_CONFIG: GameConfig = {
  stockfishElo: 1600,
  moveTimeMs: 100,
};

/** Stockfish's own supported range for `UCI_Elo` (`go docs`/UCI option range). */
export const MIN_STOCKFISH_ELO = 1320;
export const MAX_STOCKFISH_ELO = 3190;

export const MIN_MOVE_TIME_MS = 1;

/**
 * Validates a `GameConfig`. Returns an error message for the first
 * invalid field found, or `null` if the config is valid. Used both to
 * disable the Start button and to explain why.
 */
export function validateGameConfig(config: GameConfig): string | null {
  if (!Number.isFinite(config.stockfishElo)) {
    return "Stockfish Elo must be a number.";
  }
  if (!Number.isInteger(config.stockfishElo)) {
    return "Stockfish Elo must be a whole number.";
  }
  if (config.stockfishElo < MIN_STOCKFISH_ELO || config.stockfishElo > MAX_STOCKFISH_ELO) {
    return `Stockfish Elo must be between ${MIN_STOCKFISH_ELO} and ${MAX_STOCKFISH_ELO}.`;
  }

  if (!Number.isFinite(config.moveTimeMs)) {
    return "Time per move must be a number.";
  }
  if (!Number.isInteger(config.moveTimeMs)) {
    return "Time per move must be a whole number of milliseconds.";
  }
  if (config.moveTimeMs < MIN_MOVE_TIME_MS) {
    return "Time per move must be greater than zero.";
  }

  return null;
}
