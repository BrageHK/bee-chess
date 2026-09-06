import { useEffect, useState } from "react";
import { getExperiment, type ExperimentSnapshot } from "./labClient";
import { Badge } from "./components/ui/Badge";
import { Button } from "./components/ui/Button";
import { Panel, PanelBody, PanelHeader } from "./components/ui/Panel";
import { Stack } from "./components/ui/Stack";

/** How often to re-fetch an in-progress experiment's snapshot. No
 * live WebSocket stream exists for experiments (unlike games) -- see
 * `labClient.ts`'s `getExperiment` docs -- so this is the only
 * mechanism, not a fallback under a socket the way `Game.tsx`'s own
 * polling is. */
const POLL_INTERVAL_MS = 1000;

/**
 * Live progress for one running or completed A/B experiment: overall
 * score, win/draw/loss tally, a progress bar, and every game it ran,
 * each linking into the ordinary game viewer (`?game=<id>`) rather
 * than a second, experiment-specific board -- see `lab::experiment`'s
 * module docs on why an experiment's games are ordinary Lab games.
 */
export function ExperimentView({
  experimentId,
  onOpenGame,
  onBackToSetup,
}: {
  experimentId: string;
  onOpenGame: (gameId: string) => void;
  onBackToSetup: () => void;
}) {
  const [snapshot, setSnapshot] = useState<ExperimentSnapshot | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    let timer: ReturnType<typeof setTimeout> | null = null;

    const poll = () => {
      getExperiment(experimentId).then(
        (next) => {
          if (cancelled) return;
          setSnapshot(next);
          setError(null);
          if (next.status === "running") {
            timer = setTimeout(poll, POLL_INTERVAL_MS);
          }
        },
        (err: unknown) => {
          if (cancelled) return;
          setError(err instanceof Error ? err.message : String(err));
          timer = setTimeout(poll, POLL_INTERVAL_MS);
        },
      );
    };
    poll();

    return () => {
      cancelled = true;
      if (timer) clearTimeout(timer);
    };
  }, [experimentId]);

  if (error && !snapshot) {
    return (
      <Stack gap={2} align="center" className="text-center">
        <p className="m-0 text-sm text-danger">{error}</p>
        <Button onClick={onBackToSetup}>Back to setup</Button>
      </Stack>
    );
  }
  if (!snapshot) {
    return <p className="m-0 text-sm text-muted">Loading experiment…</p>;
  }

  const progress = snapshot.completed_games / snapshot.requested_games;

  return (
    <Stack gap={4} align="center" className="w-full max-w-3xl">
      <Stack gap={1} align="center" className="text-center">
        <h1 className="text-2xl font-medium">
          {snapshot.label_a} vs {snapshot.label_b}
        </h1>
        <Badge tone={snapshot.status === "completed" ? "success" : "accent"}>
          {snapshot.status === "completed" ? "Completed" : "Running"}
        </Badge>
      </Stack>

      <Panel className="w-full">
        <PanelHeader>
          {snapshot.completed_games} / {snapshot.requested_games} games
        </PanelHeader>
        <PanelBody className="grid gap-3">
          <div className="h-2 w-full overflow-hidden rounded-sm bg-surface-subtle">
            <div
              className="h-full bg-accent transition-[width] duration-300 ease-out"
              style={{ width: `${Math.round(progress * 100)}%` }}
            />
          </div>
          <div className="grid grid-cols-4 gap-2 text-center font-mono text-sm">
            <Stat label={`${snapshot.label_a} wins`} value={snapshot.wins_a} />
            <Stat label="Draws" value={snapshot.draws} />
            <Stat label={`${snapshot.label_b} wins`} value={snapshot.wins_b} />
            <Stat
              label={`${snapshot.label_a} score`}
              value={snapshot.score_a === null ? "—" : `${Math.round(snapshot.score_a * 100)}%`}
            />
          </div>
        </PanelBody>
      </Panel>

      <Panel className="w-full">
        <PanelHeader>Stats</PanelHeader>
        <PanelBody className="grid grid-cols-4 gap-2 text-center font-mono text-sm">
          <Stat label="Avg game length" value={formatDurationSeconds(snapshot.stats.avg_game_duration_ms)} />
          <Stat label="Avg plies" value={formatRounded(snapshot.stats.avg_plies)} />
          <Stat label="Runtime" value={formatDurationSeconds(snapshot.stats.runtime_ms)} />
          <Stat label="Games/hour" value={formatOneDecimal(snapshot.stats.games_per_hour)} />
        </PanelBody>
      </Panel>

      <Panel className="w-full text-left">
        <PanelHeader>Games</PanelHeader>
        <PanelBody className="grid gap-1 p-0">
          {snapshot.games.length === 0 && <p className="m-0 p-4 text-sm text-muted">No games started yet.</p>}
          {snapshot.games.map((game, index) => (
            <button
              key={game.game_id}
              type="button"
              onClick={() => onOpenGame(game.game_id)}
              className="flex items-center justify-between gap-2 border-b border-border px-4 py-2 text-left text-sm last:border-b-0 hover:bg-surface-hover"
            >
              <span>
                #{index + 1} — {game.variant_a_is_white ? snapshot.label_a : snapshot.label_b} (white) vs{" "}
                {game.variant_a_is_white ? snapshot.label_b : snapshot.label_a} (black)
                {game.plies !== null && <span className="text-subtle"> · {game.plies} plies</span>}
              </span>
              <OutcomeBadge outcome={game.outcome} />
            </button>
          ))}
        </PanelBody>
      </Panel>

      <Button onClick={onBackToSetup}>New experiment</Button>
    </Stack>
  );
}

/** `null`/`undefined` render as "—" throughout this view -- see
 * `ExperimentStats`'s own docs on why an average with no data yet is
 * `null` rather than a misleading `0`. */
function formatDurationSeconds(ms: number | null): string {
  if (ms === null) return "—";
  return `${(ms / 1000).toFixed(1)}s`;
}

function formatRounded(value: number | null): string {
  return value === null ? "—" : String(Math.round(value));
}

function formatOneDecimal(value: number | null): string {
  return value === null ? "—" : value.toFixed(1);
}

function Stat({ label, value }: { label: string; value: string | number }) {
  return (
    <div className="grid gap-0.5">
      <span className="text-lg text-text">{value}</span>
      <span className="text-xs font-sans text-subtle">{label}</span>
    </div>
  );
}

function OutcomeBadge({ outcome }: { outcome: ExperimentSnapshot["games"][number]["outcome"] }) {
  if (outcome.status === "pending") {
    return <Badge tone="neutral">live</Badge>;
  }
  if (outcome.status === "aborted") {
    return <Badge tone="danger">aborted</Badge>;
  }
  const label =
    outcome.result === "draw"
      ? "draw"
      : outcome.result === "white_wins"
        ? "white wins"
        : "black wins";
  return <Badge tone={outcome.result === "draw" ? "neutral" : "success"}>{label}</Badge>;
}
