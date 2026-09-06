import { useEffect, useState } from "react";
import { GameSetup } from "./GameSetup";
import { Game, type GameSource } from "./Game";
import { ExperimentSetup } from "./ExperimentSetup";
import { ExperimentView } from "./ExperimentView";
import { AppShell } from "./components/ui/AppShell";
import { Button } from "./components/ui/Button";
import { Toolbar } from "./components/ui/Toolbar";
import "@lichess-org/chessground/assets/chessground.base.css";
import "@lichess-org/chessground/assets/chessground.brown.css";
import "@lichess-org/chessground/assets/chessground.cburnett.css";

const GAME_ID_PARAM = "game";
const EXPERIMENT_ID_PARAM = "experiment";

type Screen =
  | { phase: "setup" }
  | { phase: "playing"; source: GameSource; gameSeq: number }
  | { phase: "experiment-setup" }
  | { phase: "experiment"; experimentId: string };

/** Reads `?game=<id>`/`?experiment=<id>` from the current URL, if
 * present -- see #69's resume-after-refresh requirement, extended to
 * cover an in-progress experiment the same way. Bee Lab's
 * `GameSnapshot`/`ExperimentSnapshot` each carry everything needed to
 * reconstruct their own screen from just an id, so the URL only needs
 * to remember which one (and its id), nothing about configuration. */
function screenFromUrl(): Screen {
  const params = new URLSearchParams(window.location.search);
  const gameId = params.get(GAME_ID_PARAM);
  if (gameId) return { phase: "playing", source: { kind: "resume", gameId }, gameSeq: 0 };
  const experimentId = params.get(EXPERIMENT_ID_PARAM);
  if (experimentId) return { phase: "experiment", experimentId };
  return { phase: "setup" };
}

/** Adds (or replaces) `?game=<id>`/`?experiment=<id>` in the URL
 * without a page navigation/reload -- `history.replaceState` rather
 * than `pushState`, since starting a game/experiment isn't a "back
 * button should undo this" moment; it's establishing what a refresh
 * at this URL now means. The two params are mutually exclusive --
 * setting one clears the other, since only one screen is ever
 * "current" at a time. */
function setResumeParamInUrl(param: typeof GAME_ID_PARAM | typeof EXPERIMENT_ID_PARAM | null, id?: string) {
  const url = new URL(window.location.href);
  url.searchParams.delete(GAME_ID_PARAM);
  url.searchParams.delete(EXPERIMENT_ID_PARAM);
  if (param && id) url.searchParams.set(param, id);
  window.history.replaceState(null, "", url);
}

export default function App() {
  const [screen, setScreen] = useState<Screen>(screenFromUrl);

  // Keeps the URL in sync with `screen` -- not the other way around
  // (this only ever *writes* the URL; reading it back happens once,
  // in the `useState` initializer above, on the initial page load).
  useEffect(() => {
    if (screen.phase === "playing" && screen.source.kind === "resume") {
      setResumeParamInUrl(GAME_ID_PARAM, screen.source.gameId);
    } else if (screen.phase === "experiment") {
      setResumeParamInUrl(EXPERIMENT_ID_PARAM, screen.experimentId);
    } else if (screen.phase === "setup" || screen.phase === "experiment-setup") {
      setResumeParamInUrl(null);
    }
    // A freshly started ("start"-kind) game doesn't know its id yet --
    // Game.tsx reports it back once Lab creates it, via onGameCreated.
  }, [screen]);

  return (
    // Nothing here is meant to be copy-pasted, so a drag anywhere on
    // the page (e.g. overshooting a piece drag past the board's edge)
    // shouldn't fan out into a multi-element text selection (#52).
    // UciLogPanel opts back in locally -- its raw traffic is worth
    // copying for debugging.
    <main className="select-none">
      <AppShell>
        <Toolbar>
          <h1 className="text-xl font-medium">Bee Chess</h1>
          <div className="flex gap-2">
            <Button
              variant={screen.phase === "setup" || screen.phase === "playing" ? "primary" : "secondary"}
              onClick={() => setScreen({ phase: "setup" })}
            >
              Play
            </Button>
            <Button
              variant={screen.phase === "experiment-setup" || screen.phase === "experiment" ? "primary" : "secondary"}
              onClick={() => setScreen({ phase: "experiment-setup" })}
            >
              Experiments
            </Button>
          </div>
        </Toolbar>
        <div className="flex flex-1 flex-col items-center justify-center gap-2 text-center">
          {screen.phase === "setup" ? (
            <GameSetup
              onStart={(white, black) =>
                setScreen({ phase: "playing", source: { kind: "start", white, black }, gameSeq: Date.now() })
              }
            />
          ) : screen.phase === "playing" ? (
            <Game
              key={screen.gameSeq}
              source={screen.source}
              onGameCreated={(gameId) => setResumeParamInUrl(GAME_ID_PARAM, gameId)}
              onBackToSetup={() => setScreen({ phase: "setup" })}
            />
          ) : screen.phase === "experiment-setup" ? (
            <ExperimentSetup onStarted={(experimentId) => setScreen({ phase: "experiment", experimentId })} />
          ) : (
            <ExperimentView
              experimentId={screen.experimentId}
              onOpenGame={(gameId) =>
                setScreen({ phase: "playing", source: { kind: "resume", gameId }, gameSeq: Date.now() })
              }
              onBackToSetup={() => setScreen({ phase: "experiment-setup" })}
            />
          )}
        </div>
      </AppShell>
    </main>
  );
}
