import { useEffect, useRef, useState } from "react";
import { GameSetup } from "./GameSetup";
import { Game, type GameSource } from "./Game";
import { ExperimentSetup } from "./ExperimentSetup";
import { ExperimentView } from "./ExperimentView";
import { Dashboard } from "./Dashboard";
import { AppShell } from "./components/ui/AppShell";
import { Button } from "./components/ui/Button";
import { Toolbar } from "./components/ui/Toolbar";
import "@lichess-org/chessground/assets/chessground.base.css";
import "@lichess-org/chessground/assets/chessground.brown.css";
import "@lichess-org/chessground/assets/chessground.cburnett.css";

type Screen =
  | { phase: "dashboard" }
  | { phase: "game-setup" }
  | { phase: "experiment-setup" }
  | { phase: "playing"; source: GameSource; gameSeq: number }
  | { phase: "experiment"; experimentId: string };

/** One `Screen` reduced to what actually identifies "where the user
 * is" for URL/history purposes -- `gameSeq` (a `Date.now()` used only
 * to force `Game`'s remount on a fresh "start") deliberately isn't
 * part of this, since it changes on every new game from the same
 * screen without that being a real navigation. */
type LocationKey =
  | { phase: "dashboard" }
  | { phase: "game-setup" }
  | { phase: "experiment-setup" }
  | { phase: "playing"; gameId: string | null }
  | { phase: "experiment"; experimentId: string };

function locationKey(screen: Screen): LocationKey {
  switch (screen.phase) {
    case "playing":
      return { phase: "playing", gameId: screen.source.kind === "resume" ? screen.source.gameId : null };
    case "experiment":
      return { phase: "experiment", experimentId: screen.experimentId };
    default:
      return { phase: screen.phase };
  }
}

function sameLocation(a: LocationKey, b: LocationKey): boolean {
  if (a.phase !== b.phase) return false;
  if (a.phase === "playing" && b.phase === "playing") return a.gameId === b.gameId;
  if (a.phase === "experiment" && b.phase === "experiment") return a.experimentId === b.experimentId;
  return true;
}

/** Builds the URL a `Screen` should correspond to -- `?game=<id>`/
 * `?experiment=<id>` for resume-after-refresh (see #69's original
 * requirement), `?new=game`/`?new=experiment` for the setup screens
 * (so refreshing mid-setup doesn't silently drop back to the
 * dashboard), and no query at all for the dashboard itself. */
function urlForScreen(screen: Screen): string {
  const url = new URL(window.location.href);
  url.search = "";
  if (screen.phase === "playing" && screen.source.kind === "resume") {
    url.searchParams.set("game", screen.source.gameId);
  } else if (screen.phase === "experiment") {
    url.searchParams.set("experiment", screen.experimentId);
  } else if (screen.phase === "game-setup") {
    url.searchParams.set("new", "game");
  } else if (screen.phase === "experiment-setup") {
    url.searchParams.set("new", "experiment");
  }
  return url.toString();
}

/** The inverse of `urlForScreen`, for the initial page load and for
 * restoring state on a `popstate` (browser back/forward) -- see the
 * component's history effect below. A freshly-started ("start"-kind)
 * game/an experiment just submitted from setup are deliberately not
 * representable here: those only exist as live in-memory transitions
 * (`onStart`/`onStarted` callbacks), never reconstructed from a URL
 * alone, since a "start" carries a full `Participant` configuration
 * the URL never stores. */
function screenFromUrl(): Screen {
  const params = new URLSearchParams(window.location.search);
  const gameId = params.get("game");
  if (gameId) return { phase: "playing", source: { kind: "resume", gameId }, gameSeq: 0 };
  const experimentId = params.get("experiment");
  if (experimentId) return { phase: "experiment", experimentId };
  const setupKind = params.get("new");
  if (setupKind === "game") return { phase: "game-setup" };
  if (setupKind === "experiment") return { phase: "experiment-setup" };
  return { phase: "dashboard" };
}

export default function App() {
  const [screen, setScreen] = useState<Screen>(screenFromUrl);

  // Keeps the URL and browser history in sync with `screen`. A
  // transition to a *different* location (dashboard -> a game,
  // experiment -> dashboard, ...) pushes a new history entry, so the
  // browser's back button actually has somewhere to go back to --
  // unlike this app's original replaceState-only version, where every
  // transition silently overwrote the current entry and back/forward
  // did nothing a user would recognize as "going back." A change that
  // keeps the *same* location (a freshly-started game learning its
  // real id via onGameCreated, still "playing", just now resumable)
  // replaces instead: that's completing the current navigation, not a
  // new one to add to history.
  //
  // A screen change caused by the browser's own back/forward
  // (popstate, handled below) must NOT push again here -- history
  // already moved; this effect just needs to update `previousLocation`
  // to match. `fromPopState` is how the popstate handler tells this
  // effect "the next run is that kind of change, don't push."
  const [previousLocation, setPreviousLocation] = useState(() => locationKey(screen));
  const fromPopState = useRef(false);
  useEffect(() => {
    const next = locationKey(screen);
    if (fromPopState.current) {
      fromPopState.current = false;
    } else {
      const url = urlForScreen(screen);
      if (sameLocation(next, previousLocation)) {
        window.history.replaceState(next, "", url);
      } else {
        window.history.pushState(next, "", url);
      }
    }
    setPreviousLocation(next);
    // previousLocation is intentionally not a dependency: comparing
    // against it and then updating it every render (rather than only
    // when `screen` changes) would push/replace on every render this
    // effect ever ran, not just on an actual screen transition.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [screen]);

  // The other half of real history support: the browser's own back/
  // forward buttons fire `popstate`, not a `screen` change from
  // inside this app. Sets `fromPopState` before updating `screen` so
  // the effect above (which runs after this state update) knows this
  // particular change already has its history entry -- it's the one
  // the user just navigated *to* -- and only needs to record it in
  // `previousLocation`, not push or replace anything.
  useEffect(() => {
    const onPopState = () => {
      fromPopState.current = true;
      setScreen(screenFromUrl());
    };
    window.addEventListener("popstate", onPopState);
    return () => window.removeEventListener("popstate", onPopState);
  }, []);

  return (
    // Nothing here is meant to be copy-pasted, so a drag anywhere on
    // the page (e.g. overshooting a piece drag past the board's edge)
    // shouldn't fan out into a multi-element text selection (#52).
    // UciLogPanel opts back in locally -- its raw traffic is worth
    // copying for debugging.
    <main className="select-none">
      <AppShell>
        <Toolbar>
          <button
            type="button"
            onClick={() => setScreen({ phase: "dashboard" })}
            className="cursor-pointer border-0 bg-transparent p-0 text-xl font-medium text-text"
          >
            Bee Chess
          </button>
          {screen.phase !== "dashboard" && (
            <Button variant="secondary" onClick={() => setScreen({ phase: "dashboard" })}>
              Dashboard
            </Button>
          )}
        </Toolbar>
        <div className="flex flex-1 flex-col items-center justify-center gap-2 text-center">
          {screen.phase === "dashboard" ? (
            <Dashboard
              onNewGame={() => setScreen({ phase: "game-setup" })}
              onNewExperiment={() => setScreen({ phase: "experiment-setup" })}
              onOpenGame={(gameId) =>
                setScreen({ phase: "playing", source: { kind: "resume", gameId }, gameSeq: Date.now() })
              }
              onOpenExperiment={(experimentId) => setScreen({ phase: "experiment", experimentId })}
            />
          ) : screen.phase === "game-setup" ? (
            <GameSetup
              onStart={(white, black) =>
                setScreen({ phase: "playing", source: { kind: "start", white, black }, gameSeq: Date.now() })
              }
            />
          ) : screen.phase === "playing" ? (
            <Game
              key={screen.gameSeq}
              source={screen.source}
              onGameCreated={(gameId) =>
                setScreen((current) =>
                  current.phase === "playing"
                    ? { ...current, source: { kind: "resume", gameId } }
                    : current,
                )
              }
              onBackToSetup={() => setScreen({ phase: "dashboard" })}
              onOpenExperiment={(experimentId) => setScreen({ phase: "experiment", experimentId })}
            />
          ) : screen.phase === "experiment-setup" ? (
            <ExperimentSetup onStarted={(experimentId) => setScreen({ phase: "experiment", experimentId })} />
          ) : (
            <ExperimentView
              experimentId={screen.experimentId}
              onOpenGame={(gameId) =>
                setScreen({ phase: "playing", source: { kind: "resume", gameId }, gameSeq: Date.now() })
              }
              onBackToSetup={() => setScreen({ phase: "dashboard" })}
            />
          )}
        </div>
      </AppShell>
    </main>
  );
}
