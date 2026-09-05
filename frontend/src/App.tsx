import { useEffect, useState } from "react";
import { GameSetup } from "./GameSetup";
import { Game, type GameSource } from "./Game";
import "@lichess-org/chessground/assets/chessground.base.css";
import "@lichess-org/chessground/assets/chessground.brown.css";
import "@lichess-org/chessground/assets/chessground.cburnett.css";

const GAME_ID_PARAM = "game";

type Screen = { phase: "setup" } | { phase: "playing"; source: GameSource; gameSeq: number };

/** Reads `?game=<id>` from the current URL, if present -- see #69's
 * resume-after-refresh requirement. Bee Lab's `GameSnapshot` carries
 * everything needed to reconstruct a game's UI (position, moves,
 * status, and now who's playing each side -- see `ParticipantInfo`),
 * so the URL only needs to remember the game's id, nothing about its
 * configuration. */
function gameIdFromUrl(): string | null {
  return new URLSearchParams(window.location.search).get(GAME_ID_PARAM);
}

/** Adds (or replaces) `?game=<id>` in the URL without a page
 * navigation/reload -- `history.replaceState` rather than `pushState`,
 * since starting a game isn't a "back button should undo this"
 * moment; it's establishing what a refresh at this URL now means. */
function setGameIdInUrl(gameId: string | null) {
  const url = new URL(window.location.href);
  if (gameId) url.searchParams.set(GAME_ID_PARAM, gameId);
  else url.searchParams.delete(GAME_ID_PARAM);
  window.history.replaceState(null, "", url);
}

export default function App() {
  const [screen, setScreen] = useState<Screen>(() => {
    const gameId = gameIdFromUrl();
    return gameId
      ? { phase: "playing", source: { kind: "resume", gameId }, gameSeq: 0 }
      : { phase: "setup" };
  });

  // Keeps the URL in sync with `screen` -- not the other way around
  // (this only ever *writes* the URL; reading it back happens once,
  // in the `useState` initializer above, on the initial page load).
  useEffect(() => {
    if (screen.phase === "playing" && screen.source.kind === "resume") {
      setGameIdInUrl(screen.source.gameId);
    } else if (screen.phase === "setup") {
      setGameIdInUrl(null);
    }
    // A freshly started ("start"-kind) game doesn't know its id yet --
    // Game.tsx reports it back once Lab creates it, via onGameCreated.
  }, [screen]);

  return (
    <main
      style={{
        display: "grid",
        // An explicit 1fr column (rather than relying on the default
        // auto-sized implicit track) makes every child's available
        // width equal to <main>'s own width, not the width of the
        // widest child -- otherwise the grid track itself grows and
        // shrinks with content, and everything centered inside it
        // reflows along with it.
        gridTemplateColumns: "1fr",
        justifyItems: "center",
        alignItems: "center",
        gap: 8,
        padding: 24,
        textAlign: "center",
        // Nothing here is meant to be copy-pasted, so a drag anywhere
        // on the page (e.g. overshooting a piece drag past the board's
        // edge) shouldn't fan out into a multi-element text selection
        // (#52). UciLogPanel opts back in locally -- its raw traffic is
        // worth copying for debugging.
        userSelect: "none",
      }}
    >
      <h1>Bee Chess</h1>
      {screen.phase === "setup" ? (
        <GameSetup
          onStart={(white, black) =>
            setScreen({ phase: "playing", source: { kind: "start", white, black }, gameSeq: Date.now() })
          }
        />
      ) : (
        <Game
          key={screen.gameSeq}
          source={screen.source}
          onGameCreated={(gameId) => setGameIdInUrl(gameId)}
          onBackToSetup={() => setScreen({ phase: "setup" })}
        />
      )}
    </main>
  );
}
