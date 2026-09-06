import { useEffect, useState } from "react";
import type { UciLogLine } from "./engine";
import { evalBarFraction, formatScore, parseUciInfo, toWhitePerspective, type UciInfo } from "./uciInfo";

/** The bar's fill position clamps a score to this range (in pawns);
 * the numeric label always shows the real, unclamped value. */
const CLAMP_PAWNS = 3;
const HEIGHT = 480; // matches the board's own fixed size, see Chessground.tsx

interface EvalBarProps {
  /** Which color slot the tracked engine is playing. UCI scores are
   * reported from the side-to-move's own perspective, not white's, so
   * this is what lets the bar read consistently as "white's
   * advantage" regardless of which side the engine is on. */
  color: "white" | "black";
  /** Subscribes to an engine's log lines; returns an unsubscribe function. */
  subscribe: (listener: (line: UciLogLine) => void) => () => void;
}

/**
 * Live, broadcast-style eval bar for one engine's most recently
 * reported `info score`: fills from the bottom for a white advantage,
 * from the top for black. Resets to neutral on every new game since
 * this remounts along with the rest of `Game` (see its own `key`).
 */
export function EvalBar({ color, subscribe }: EvalBarProps) {
  const [info, setInfo] = useState<UciInfo>({});

  useEffect(() => {
    return subscribe((line) => {
      if (line.direction !== "received") return;
      const parsed = parseUciInfo(line.text);
      if (parsed) setInfo((prev) => ({ ...prev, ...parsed }));
    });
  }, [subscribe]);

  const white = toWhitePerspective(info, color);
  const whiteFraction = evalBarFraction(white, CLAMP_PAWNS);
  const label = formatScore(white);

  return (
    // Fixed light/dark colors, not theme tokens -- like chessground's
    // own piece/board colors, this bar is a literal light-vs-dark
    // visual metaphor (white's advantage vs. black's) rather than page
    // chrome, so it shouldn't flip with the app's light/dark theme.
    <div
      title="Stockfish's evaluation, from white's perspective"
      className="flex w-7 shrink-0 flex-col-reverse overflow-hidden rounded-md bg-[#3a3a3a]"
      style={{ height: HEIGHT }}
    >
      <div
        className="flex items-start justify-center bg-[#f0f0f0] transition-[height] duration-200 ease-out"
        style={{ height: `${whiteFraction * 100}%` }}
      >
        {label && <span className="py-0.5 font-mono text-[10px] text-[#111]">{label}</span>}
      </div>
    </div>
  );
}
