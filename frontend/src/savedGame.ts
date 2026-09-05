import type { Participant } from "./participant";

/**
 * A game in progress, persisted to `sessionStorage` so a page refresh
 * resumes it instead of dropping back to the setup screen (#55).
 * `sessionStorage` (not `localStorage`) is deliberate: a closed tab
 * abandoning the game is the expected reset, only a same-tab refresh
 * should resume it.
 */
export interface SavedGame {
  white: Participant;
  black: Participant;
  /** Every move played so far, as UCI -- the same list `bestMove` is
   * given each turn, so replaying it is enough to reconstruct the
   * position (and `posRef`) exactly. */
  moves: string[];
}

const STORAGE_KEY = "bee-chess:game";

/** Returns the saved game, or `null` if there isn't one or it doesn't
 * parse as one (a stale format from a previous version of this app,
 * say) -- never throws. */
export function loadSavedGame(): SavedGame | null {
  try {
    const raw = sessionStorage.getItem(STORAGE_KEY);
    if (!raw) return null;
    const parsed: unknown = JSON.parse(raw);
    if (
      !parsed ||
      typeof parsed !== "object" ||
      !("white" in parsed) ||
      !("black" in parsed) ||
      !Array.isArray((parsed as { moves: unknown }).moves)
    ) {
      return null;
    }
    return parsed as SavedGame;
  } catch {
    return null;
  }
}

/** Never throws -- `sessionStorage` can fail (private browsing, quota),
 * and losing resume-on-refresh isn't worth failing the game over. */
export function saveGame(game: SavedGame): void {
  try {
    sessionStorage.setItem(STORAGE_KEY, JSON.stringify(game));
  } catch {
    // see above
  }
}

export function clearSavedGame(): void {
  try {
    sessionStorage.removeItem(STORAGE_KEY);
  } catch {
    // see saveGame
  }
}
