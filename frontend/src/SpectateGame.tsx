import { useRef, useState } from "react";
import { Chess } from "chessops/chess";
import { makeFen } from "chessops/fen";
import { parseUci } from "chessops/util";
import type { Key } from "@lichess-org/chessground/types";
import { Chessground } from "./Chessground";
import { whiteEngine, blackEngine } from "./engine";
import { UciLogPanel } from "./UciLogPanel";
import {
  DEFAULT_GAME_CONFIG,
  MAX_STOCKFISH_ELO,
  MIN_MOVE_TIME_MS,
  MIN_STOCKFISH_ELO,
  validateGameConfig,
  type GameConfig,
} from "./gameConfig";

/** Claim a draw once the fifty-move counter is full; chessops does not. */
const MAX_HALFMOVES = 100;

/** Board position before any move; avoids reading the ref during render. */
const START_FEN = makeFen(Chess.default().toSetup());

type GamePhase = "config" | "playing" | "finished";

/** Watch Stockfish (white) and Bee (black) play each other -- no human
 * input, the board is view-only. See `PlayVsMamba` for the interactive
 * mode. */
export function SpectateGame() {
  const posRef = useRef(Chess.default());
  // Bumped on every startGame() call; a running loop checks it still
  // matches the value it started with before writing state, so a stale
  // loop (superseded by a newer "Start Game" click) can't clobber a
  // later game's state after the fact.
  const gameIdRef = useRef(0);
  const [phase, setPhase] = useState<GamePhase>("config");
  const [config, setConfig] = useState<GameConfig>(DEFAULT_GAME_CONFIG);
  const [fen, setFen] = useState(START_FEN);
  const [lastMove, setLastMove] = useState<Key[] | undefined>();
  const [status, setStatus] = useState("");
  // Mirrors gameIdRef in state (a ref alone can't drive a re-render).
  // Used as part of each UciLogPanel's `key` below so a new game
  // remounts (and so clears) both panels instead of carrying over the
  // previous game's log history.
  const [gameSeq, setGameSeq] = useState(0);

  const startGame = async (gameConfig: GameConfig) => {
    const gameId = ++gameIdRef.current;
    const current = () => gameId === gameIdRef.current;

    posRef.current = Chess.default();
    setFen(START_FEN);
    setLastMove(undefined);
    setPhase("playing");
    setStatus("connecting to the bridge…");
    setGameSeq(gameId);

    const moves: string[] = [];

    try {
      await Promise.all([whiteEngine.init(), blackEngine.init()]);
      await whiteEngine.setOption("UCI_LimitStrength", true);
      await whiteEngine.setOption("UCI_Elo", gameConfig.stockfishElo);
    } catch (err) {
      if (current()) {
        setStatus(message(err));
        setPhase("finished");
      }
      return;
    }

    while (current() && !posRef.current.isEnd()) {
      if (posRef.current.halfmoves >= MAX_HALFMOVES) {
        setStatus("draw by the fifty-move rule");
        setPhase("finished");
        return;
      }

      const white = posRef.current.turn === "white";
      const engine = white ? whiteEngine : blackEngine;
      setStatus(`${engine.name} thinking…`);

      let uci: string;
      try {
        uci = await engine.bestMove(moves, gameConfig.moveTimeMs);
      } catch (err) {
        if (current()) {
          setStatus(message(err));
          setPhase("finished");
        }
        return;
      }
      if (!current()) return;

      const move = parseUci(uci);
      if (!move || !posRef.current.isLegal(move)) {
        setStatus(`${engine.name} played an illegal move: ${uci}`);
        setPhase("finished");
        return;
      }

      posRef.current.play(move);
      moves.push(uci);
      setFen(makeFen(posRef.current.toSetup()));
      setLastMove([uci.slice(0, 2), uci.slice(2, 4)] as Key[]);
    }

    if (current()) {
      setStatus(outcome(posRef.current));
      setPhase("finished");
    }
  };

  return (
    <section
      style={{
        display: "grid",
        gridTemplateColumns: "1fr",
        justifyItems: "center",
        alignItems: "center",
        gap: 8,
        textAlign: "center",
      }}
    >
      <h1>
        {whiteEngine.name} (white) vs {blackEngine.name} (black)
      </h1>
      {phase === "config" ? (
        <GameConfigForm initial={config} onStart={(c) => { setConfig(c); void startGame(c); }} />
      ) : (
        <>
          <Chessground config={{ fen, lastMove, viewOnly: true }} />
          <p>{status}</p>
          {phase === "finished" && (
            <button onClick={() => setPhase("config")}>New game</button>
          )}
          {/*
            Fixed width, no wrap: with flexWrap and content-dependent
            panel widths, whether the two panels sat side by side or
            stacked flipped depending on how much log text had
            accumulated in each -- an unstable layout that changed
            shape as a game ran. A fixed-width row where each panel
            always takes an equal, fixed share (flex: "1 1 0",
            minWidth: 0 so it can't grow past that share) removes the
            two things that made width depend on content.
          */}
          <div style={{ display: "flex", gap: 16, width: "100%", maxWidth: 900 }}>
            <UciLogPanel
              key={`white-${gameSeq}`}
              name={whiteEngine.name}
              subscribe={(l) => whiteEngine.onLog(l)}
            />
            <UciLogPanel
              key={`black-${gameSeq}`}
              name={blackEngine.name}
              subscribe={(l) => blackEngine.onLog(l)}
            />
          </div>
        </>
      )}
    </section>
  );
}

function GameConfigForm({
  initial,
  onStart,
}: {
  initial: GameConfig;
  onStart: (config: GameConfig) => void;
}) {
  // Kept as strings while editing so the field can be temporarily empty
  // or mid-edit (e.g. "16") without snapping back to a number.
  const [eloText, setEloText] = useState(String(initial.stockfishElo));
  const [moveTimeText, setMoveTimeText] = useState(String(initial.moveTimeMs));

  const config: GameConfig = {
    stockfishElo: Number(eloText),
    moveTimeMs: Number(moveTimeText),
  };
  const error = validateGameConfig(config);

  return (
    <form
      style={{ display: "grid", gap: 12, justifyItems: "start" }}
      onSubmit={(e) => {
        e.preventDefault();
        if (!error) onStart(config);
      }}
    >
      <label style={{ display: "grid", gap: 4 }}>
        Stockfish Elo
        <input
          type="number"
          value={eloText}
          min={MIN_STOCKFISH_ELO}
          max={MAX_STOCKFISH_ELO}
          step={1}
          onChange={(e) => setEloText(e.target.value)}
        />
      </label>
      <label style={{ display: "grid", gap: 4 }}>
        Time per move (ms)
        <input
          type="number"
          value={moveTimeText}
          min={MIN_MOVE_TIME_MS}
          step={1}
          onChange={(e) => setMoveTimeText(e.target.value)}
        />
      </label>
      {error && <p style={{ color: "crimson", margin: 0 }}>{error}</p>}
      <button type="submit" disabled={error !== null}>
        Start Game
      </button>
    </form>
  );
}

const message = (err: unknown) => (err instanceof Error ? err.message : String(err));

function outcome(pos: Chess): string {
  const winner = pos.outcome()?.winner;
  return winner ? `game over — ${winner} wins` : "game over — draw";
}
