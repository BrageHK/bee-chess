import { useEffect, useState, type ReactNode } from "react";
import { listExperiments, listGames, type ExperimentSnapshot, type GameSnapshot } from "./labClient";
import { Badge } from "./components/ui/Badge";
import { Button } from "./components/ui/Button";
import { Panel, PanelBody, PanelHeader } from "./components/ui/Panel";
import { Stack } from "./components/ui/Stack";

/** Home screen: start a new game or experiment, and see everything
 * Lab currently knows about -- split into what's still running and
 * what's already finished, newest first (see `labClient.ts`'s
 * `listGames`/`listExperiments`). This is a point-in-time fetch, not
 * a live subscription -- a "Refresh" action re-fetches rather than
 * this component polling on its own, since a dashboard the user is
 * just glancing at doesn't need the same live-update urgency as
 * `ExperimentView` following one specific run.
 */
export function Dashboard({
  onNewGame,
  onNewExperiment,
  onOpenGame,
  onOpenExperiment,
}: {
  onNewGame: () => void;
  onNewExperiment: () => void;
  onOpenGame: (gameId: string) => void;
  onOpenExperiment: (experimentId: string) => void;
}) {
  const [games, setGames] = useState<GameSnapshot[] | null>(null);
  const [experiments, setExperiments] = useState<ExperimentSnapshot[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [refreshSeq, setRefreshSeq] = useState(0);

  useEffect(() => {
    let cancelled = false;
    Promise.all([listGames(), listExperiments()]).then(
      ([gamesResult, experimentsResult]) => {
        if (cancelled) return;
        setGames(gamesResult);
        setExperiments(experimentsResult);
        setError(null);
      },
      (err: unknown) => {
        if (cancelled) return;
        setError(err instanceof Error ? err.message : String(err));
      },
    );
    return () => {
      cancelled = true;
    };
  }, [refreshSeq]);

  const runningGames = games?.filter((g) => g.status === "running") ?? [];
  const pastGames = games?.filter((g) => g.status !== "running") ?? [];
  const runningExperiments = experiments?.filter((e) => e.status === "running") ?? [];
  const pastExperiments = experiments?.filter((e) => e.status !== "running") ?? [];

  return (
    <Stack gap={4} align="center" className="w-full max-w-3xl">
      <Stack gap={3} align="center" className="w-full">
        <div className="flex flex-wrap justify-center gap-3">
          <Button variant="primary" onClick={onNewGame}>
            New game
          </Button>
          <Button variant="primary" onClick={onNewExperiment}>
            New experiment
          </Button>
          <Button onClick={() => setRefreshSeq((n) => n + 1)}>Refresh</Button>
        </div>
        {error && <p className="m-0 text-sm text-danger">{error}</p>}
      </Stack>

      {!games || !experiments ? (
        <p className="m-0 text-sm text-muted">Loading…</p>
      ) : (
        <>
          <Section title="Running" empty="Nothing running right now." rowCount={runningExperiments.length + runningGames.length}>
            {runningExperiments.map((experiment) => (
              <ExperimentRow key={experiment.id} experiment={experiment} onOpen={onOpenExperiment} />
            ))}
            {runningGames.map((game) => (
              <GameRow key={game.id} game={game} onOpen={onOpenGame} />
            ))}
          </Section>

          <Section title="Past" empty="No finished games or experiments yet." rowCount={pastExperiments.length + pastGames.length}>
            {pastExperiments.map((experiment) => (
              <ExperimentRow key={experiment.id} experiment={experiment} onOpen={onOpenExperiment} />
            ))}
            {pastGames.map((game) => (
              <GameRow key={game.id} game={game} onOpen={onOpenGame} />
            ))}
          </Section>
        </>
      )}
    </Stack>
  );
}

function Section({
  title,
  empty,
  rowCount,
  children,
}: {
  title: string;
  empty: string;
  /** Total number of actual rows across every list passed as
   * `children` -- `children`'s own length is always the number of
   * `{...}` expressions in JSX (here, one per list rendered), not the
   * number of rows those lists produced, so it can't be used to
   * detect "nothing to show" on its own. */
  rowCount: number;
  children: ReactNode;
}) {
  return (
    <Panel className="w-full text-left">
      <PanelHeader>{title}</PanelHeader>
      <PanelBody className="grid gap-1 p-0">
        {rowCount === 0 ? <p className="m-0 p-4 text-sm text-muted">{empty}</p> : children}
      </PanelBody>
    </Panel>
  );
}

function Row({ onClick, children }: { onClick: () => void; children: ReactNode }) {
  return (
    <button
      type="button"
      onClick={onClick}
      className="flex items-center justify-between gap-2 border-b border-border px-4 py-2 text-left text-sm last:border-b-0 hover:bg-surface-hover"
    >
      {children}
    </button>
  );
}

function GameRow({ game, onOpen }: { game: GameSnapshot; onOpen: (gameId: string) => void }) {
  const white = game.white.kind === "human" ? "You" : game.white.name;
  const black = game.black.kind === "human" ? "You" : game.black.name;
  return (
    <Row onClick={() => onOpen(game.id)}>
      <span>
        Game — {white} vs {black}
      </span>
      <GameStatusBadge game={game} />
    </Row>
  );
}

function GameStatusBadge({ game }: { game: GameSnapshot }) {
  if (game.status === "running") {
    return <Badge tone="accent">running</Badge>;
  }
  if (game.status === "aborted") {
    return <Badge tone="danger">aborted</Badge>;
  }
  return (
    <Badge tone={game.result === "draw" ? "neutral" : "success"}>
      {game.result === "draw" ? "draw" : game.result === "white_wins" ? "white wins" : "black wins"}
    </Badge>
  );
}

function ExperimentRow({
  experiment,
  onOpen,
}: {
  experiment: ExperimentSnapshot;
  onOpen: (experimentId: string) => void;
}) {
  return (
    <Row onClick={() => onOpen(experiment.id)}>
      <span>
        Experiment — {experiment.label_a} vs {experiment.label_b} ({experiment.completed_games}/
        {experiment.requested_games})
      </span>
      <Badge tone={experiment.status === "completed" ? "success" : "accent"}>{experiment.status}</Badge>
    </Row>
  );
}
