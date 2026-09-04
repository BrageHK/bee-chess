import { useEffect, useRef, useState } from "react";
import type { UciLogLine } from "./engine";
import { appendLogLine } from "./uciLog";

/**
 * Raw per-engine UCI log panel: everything sent (→) and received (←)
 * on one engine's connection, in order. Deliberately dumb -- no
 * parsing of `info` fields yet (that's a follow-up panel); this is
 * for seeing protocol bugs, hangs, missing `info`, and wrong
 * `position` commands as they happen.
 */

interface UciLogPanelProps {
  name: string;
  /** Subscribes to an engine's log lines; returns an unsubscribe function. */
  subscribe: (listener: (line: UciLogLine) => void) => () => void;
}

/**
 * To clear history on a new game, render this with a `key` that
 * changes per game (e.g. `key={gameSeq}`) rather than passing a
 * "clear" signal as a prop -- that lets React remount (and so reset)
 * the panel's state the normal way instead of clearing it inside an
 * effect.
 */
export function UciLogPanel({ name, subscribe }: UciLogPanelProps) {
  const [lines, setLines] = useState<UciLogLine[]>([]);
  const scrollRef = useRef<HTMLDivElement>(null);
  const autoScrollRef = useRef(true);

  useEffect(() => {
    return subscribe((line) => setLines((prev) => appendLogLine(prev, line)));
  }, [subscribe]);

  useEffect(() => {
    const el = scrollRef.current;
    if (el && autoScrollRef.current) el.scrollTop = el.scrollHeight;
  }, [lines]);

  return (
    <section style={{ display: "grid", gap: 4, minWidth: 320, textAlign: "left" }}>
      <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between" }}>
        <strong>{name}</strong>
        <button type="button" onClick={() => setLines([])}>
          Clear logs
        </button>
      </div>
      <div
        ref={scrollRef}
        onScroll={(e) => {
          const el = e.currentTarget;
          // Only keep auto-scrolling if the user is already at (or
          // near) the bottom; otherwise a manual scroll-up to read
          // history would get yanked back down on the next line.
          autoScrollRef.current = el.scrollHeight - el.scrollTop - el.clientHeight < 32;
        }}
        style={{
          height: 200,
          overflowY: "auto",
          background: "#111",
          color: "#ddd",
          fontFamily: "monospace",
          fontSize: 12,
          padding: 8,
          borderRadius: 4,
          whiteSpace: "pre-wrap",
          textAlign: "left",
        }}
      >
        {lines.length === 0 && <div style={{ color: "#777" }}>(no traffic yet)</div>}
        {lines.map((line, index) => (
          <div
            key={index}
            style={{ color: line.direction === "sent" ? "#7fd1ff" : "#8fe38f" }}
          >
            {line.direction === "sent" ? "→ " : "← "}
            {line.text}
          </div>
        ))}
      </div>
    </section>
  );
}
