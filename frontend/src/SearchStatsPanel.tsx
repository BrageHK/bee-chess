import { useEffect, useState } from "react";
import type { UciLogLine } from "./engine";
import { formatCount, formatNps, formatScore, parseUciInfo, type UciInfo } from "./uciInfo";

/**
 * Live search-stats panel: the most recently reported depth/eval/
 * nodes/nps/time/pv for one engine, parsed from its raw `info` lines.
 * Unlike the raw log panel (which accumulates history), this only
 * ever shows the latest status -- each new `info` line replaces the
 * fields it carries, keeping whatever the previous line reported for
 * fields the new one doesn't mention (e.g. a line with `pv` but no
 * `nps` doesn't blank out the last known nps).
 */

interface SearchStatsPanelProps {
  name: string;
  /** Subscribes to an engine's log lines; returns an unsubscribe function. */
  subscribe: (listener: (line: UciLogLine) => void) => () => void;
}

const ROW_LABEL_WIDTH = 72;

export function SearchStatsPanel({ name, subscribe }: SearchStatsPanelProps) {
  const [info, setInfo] = useState<UciInfo>({});

  useEffect(() => {
    return subscribe((line) => {
      if (line.direction !== "received") return;
      const parsed = parseUciInfo(line.text);
      if (parsed) setInfo((prev) => ({ ...prev, ...parsed }));
    });
  }, [subscribe]);

  const score = formatScore(info);

  return (
    <section style={{ display: "grid", gap: 4, flex: "1 1 0", minWidth: 0, textAlign: "left" }}>
      <strong>{name}</strong>
      <div
        style={{
          background: "#111",
          color: "#ddd",
          fontFamily: "monospace",
          fontSize: 12,
          padding: 8,
          borderRadius: 4,
          display: "grid",
          gridTemplateColumns: `${ROW_LABEL_WIDTH}px 1fr`,
          rowGap: 2,
        }}
      >
        <Row label="Depth" value={formatDepth(info)} />
        <Row label="Eval" value={score} />
        <Row label="Nodes" value={info.nodes !== undefined ? formatCount(info.nodes) : undefined} />
        <Row label="NPS" value={info.nps !== undefined ? formatNps(info.nps) : undefined} />
        <Row label="Time" value={info.timeMs !== undefined ? `${info.timeMs} ms` : undefined} />
        <Row
          label="PV"
          value={info.pv && info.pv.length > 0 ? info.pv.join(" ") : undefined}
          truncate
        />
      </div>
    </section>
  );
}

function formatDepth(info: UciInfo): string | undefined {
  if (info.depth === undefined) return undefined;
  return info.seldepth !== undefined ? `${info.depth}/${info.seldepth}` : String(info.depth);
}

function Row({
  label,
  value,
  truncate,
}: {
  label: string;
  value: string | undefined;
  truncate?: boolean;
}) {
  return (
    <>
      <span style={{ color: "#777" }}>{label}</span>
      <span
        style={
          truncate
            ? { overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }
            : undefined
        }
      >
        {value ?? "—"}
      </span>
    </>
  );
}
