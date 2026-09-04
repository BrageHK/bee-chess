import type { UciLogLine } from "./engine";

/** Keeps browser memory bounded during long searches/games. */
export const MAX_LINES = 500;

/**
 * Appends `line` to `prev`, trimming from the front once `MAX_LINES` is
 * reached so history stays bounded. Kept pure and separate from
 * UciLogPanel.tsx (which must only export the component itself for
 * Fast Refresh) so this can be unit tested without rendering anything.
 */
export function appendLogLine(prev: readonly UciLogLine[], line: UciLogLine): UciLogLine[] {
  const next = prev.length >= MAX_LINES ? prev.slice(prev.length - MAX_LINES + 1) : prev;
  return [...next, line];
}
