/**
 * Parses UCI `info` search-progress lines (depth/score/nodes/nps/time/
 * pv) into a structured shape, for display in a live search-stats
 * panel.
 *
 * This is a different concern from the debug diagnostics channel
 * (`info string ...`, see #42/#44): a diagnostic is a one-off human
 * message, while these are exactly the fields real UCI search
 * telemetry uses. `info string` lines are deliberately not parsed
 * here at all -- they carry no fields this shape understands, and
 * fall out of parseUciInfo naturally (no `depth` token to anchor on).
 */

/** One engine's most recent reported search status. */
export interface UciInfo {
  depth?: number;
  seldepth?: number;
  /** In pawns, from the reporting engine's own perspective (positive
   * favors the side to move), unless `scoreMate` is set instead. */
  scoreCp?: number;
  /** Moves to mate, from the reporting engine's own perspective
   * (positive: this side mates; negative: this side gets mated). */
  scoreMate?: number;
  nodes?: number;
  nps?: number;
  /** Milliseconds spent on this search so far. */
  timeMs?: number;
  /** The principal variation, as UCI move tokens (e.g. "e2e4"). */
  pv?: string[];
}

/**
 * Parses one raw UCI line. Returns `null` if it isn't an `info` line
 * carrying at least one recognized field (this covers `bestmove`,
 * `id`, `uciok`, `readyok`, `info string ...`, and anything else that
 * isn't search-progress telemetry).
 */
export function parseUciInfo(line: string): UciInfo | null {
  const tokens = line.trim().split(/\s+/);
  if (tokens[0] !== "info") return null;

  const info: UciInfo = {};
  let matchedAnyField = false;

  for (let i = 1; i < tokens.length; i++) {
    const token = tokens[i];
    switch (token) {
      case "depth":
        info.depth = parseIntField(tokens[++i]);
        matchedAnyField ||= info.depth !== undefined;
        break;
      case "seldepth":
        info.seldepth = parseIntField(tokens[++i]);
        matchedAnyField ||= info.seldepth !== undefined;
        break;
      case "nodes":
        info.nodes = parseIntField(tokens[++i]);
        matchedAnyField ||= info.nodes !== undefined;
        break;
      case "nps":
        info.nps = parseIntField(tokens[++i]);
        matchedAnyField ||= info.nps !== undefined;
        break;
      case "time":
        info.timeMs = parseIntField(tokens[++i]);
        matchedAnyField ||= info.timeMs !== undefined;
        break;
      case "score": {
        const kind = tokens[++i];
        const value = parseIntField(tokens[++i]);
        if (value === undefined) break;
        if (kind === "cp") {
          info.scoreCp = value;
          matchedAnyField = true;
        } else if (kind === "mate") {
          info.scoreMate = value;
          matchedAnyField = true;
        } else {
          i--; // unknown score kind: back up, don't consume a field token as its value
        }
        break;
      }
      case "pv":
        // pv is always last in a real info line (everything after it
        // is more move tokens), so consume the rest of the line.
        info.pv = tokens.slice(i + 1);
        matchedAnyField ||= info.pv.length > 0;
        i = tokens.length;
        break;
      default:
        // Unrecognized field (multipv, hashfull, tbhits, currmove,
        // string, ...): skip just this token, not its value, since we
        // don't know its arity. Fields we don't parse are simply
        // absent from the result rather than causing a parse failure.
        break;
    }
  }

  return matchedAnyField ? info : null;
}

function parseIntField(token: string | undefined): number | undefined {
  if (token === undefined) return undefined;
  const value = Number(token);
  return Number.isInteger(value) ? value : undefined;
}

/** Formats a score for display: "+0.31", "-1.20", "M4", "-M2". */
export function formatScore(info: Pick<UciInfo, "scoreCp" | "scoreMate">): string | undefined {
  if (info.scoreMate !== undefined) {
    return info.scoreMate >= 0 ? `M${info.scoreMate}` : `-M${Math.abs(info.scoreMate)}`;
  }
  if (info.scoreCp !== undefined) {
    const pawns = info.scoreCp / 100;
    const sign = pawns >= 0 ? "+" : "-";
    return `${sign}${Math.abs(pawns).toFixed(2)}`;
  }
  return undefined;
}

/** Formats a node/count value with thousands separators: 42122 -> "42,122". */
export function formatCount(n: number): string {
  return n.toLocaleString("en-US");
}

/** Formats a nodes-per-second value compactly: 1053050 -> "1.05M". */
export function formatNps(nps: number): string {
  if (nps >= 1_000_000) return `${(nps / 1_000_000).toFixed(2)}M`;
  if (nps >= 1_000) return `${(nps / 1_000).toFixed(1)}k`;
  return String(nps);
}
