/**
 * Game settings for a human vs Bee-Mamba game: which color the human
 * plays, and how long the engine gets to think per move.
 */
import { MIN_MOVE_TIME_MS } from "./gameConfig";

export interface PlayConfig {
  humanColor: "white" | "black";
  moveTimeMs: number;
}

export const DEFAULT_PLAY_CONFIG: PlayConfig = {
  humanColor: "white",
  moveTimeMs: 500,
};

/**
 * Validates a `PlayConfig`. Returns an error message for the first
 * invalid field found, or `null` if the config is valid.
 */
export function validatePlayConfig(config: PlayConfig): string | null {
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
