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
    <div
      title="Stockfish's evaluation, from white's perspective"
      style={{
        width: 28,
        height: HEIGHT,
        display: "flex",
        flexDirection: "column-reverse",
        background: "#3a3a3a",
        borderRadius: 4,
        overflow: "hidden",
        flexShrink: 0,
      }}
    >
      <div
        style={{
          height: `${whiteFraction * 100}%`,
          background: "#f0f0f0",
          display: "flex",
          alignItems: "flex-start",
          justifyContent: "center",
          transition: "height 200ms ease-out",
        }}
      >
        {label && (
          <span style={{ fontSize: 10, fontFamily: "monospace", color: "#111", padding: "2px 0" }}>
            {label}
          </span>
        )}
      </div>
    </div>
  );
}
