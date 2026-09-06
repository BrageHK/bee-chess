import { useEffect, useRef, useState } from "react";
import type { UciLogLine } from "./engine";
import { appendLogLine } from "./uciLog";
import { Button } from "./components/ui/Button";
import { Panel, PanelBody, PanelHeader } from "./components/ui/Panel";

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
    // flex: "1 1 0" makes every panel in the row claim an equal share
    // of the row's fixed width, unconditionally -- not "as much as its
    // content wants" (the default flex-basis: auto). Flex children
    // also default to min-width: auto, which lets a long UCI line (a
    // long PV, say) override that equal share and force the panel
    // wider than intended; min-w-0 removes that override so the
    // panel's width never depends on its own content, only on the
    // row's fixed width.
    <Panel className="min-w-0 flex-1 text-left">
      <PanelHeader className="flex items-center justify-between gap-2 font-normal">
        <strong className="font-medium text-text">{name}</strong>
        <Button variant="secondary" onClick={() => setLines([])}>
          Clear logs
        </Button>
      </PanelHeader>
      <PanelBody className="p-0">
        <div
          ref={scrollRef}
          onScroll={(e) => {
            const el = e.currentTarget;
            // Only keep auto-scrolling if the user is already at (or
            // near) the bottom; otherwise a manual scroll-up to read
            // history would get yanked back down on the next line.
            autoScrollRef.current = el.scrollHeight - el.scrollTop - el.clientHeight < 32;
          }}
          className="h-[200px] w-full select-text overflow-auto bg-surface-subtle p-2 text-left font-mono text-xs"
        >
          {lines.length === 0 && <div className="text-subtle">(no traffic yet)</div>}
          {lines.map((line, index) => (
            <div
              key={index}
              className={
                line.direction === "sent"
                  ? "whitespace-pre text-accent"
                  : "whitespace-pre text-success"
              }
            >
              {line.direction === "sent" ? "→ " : "← "}
              {line.text}
            </div>
          ))}
        </div>
      </PanelBody>
    </Panel>
  );
}
