import { useRef, useState } from "react";
import { Chess } from "chessops/chess";
import { makeFen } from "chessops/fen";
import { chessgroundDests } from "chessops/compat";
import { parseUci } from "chessops/util";
import type { Key } from "@lichess-org/chessground/types";
import type { Dests } from "@lichess-org/chessground/types";
import { Chessground } from "./Chessground";
import { mambaEngine } from "./engine";
import { UciLogPanel } from "./UciLogPanel";
import {
  DEFAULT_PLAY_CONFIG,
  validatePlayConfig,
  type PlayConfig,
} from "./playConfig";
import { MIN_MOVE_TIME_MS } from "./gameConfig";

/** Claim a draw once the fifty-move counter is full; chessops does not. */
const MAX_HALFMOVES = 100;

/** Board position before any move; avoids reading the ref during render. */
const START_FEN = makeFen(Chess.default().toSetup());

type GamePhase = "config" | "playing" | "finished";

/** Play a real game against the trained Bee-Mamba checkpoint: drag
 * pieces on the board yourself, the engine replies over the same UCI
 * bridge the spectator mode uses. See `SpectateGame` for the
 * engine-vs-engine view-only mode. */
export function PlayVsMamba() {
  const posRef = useRef(Chess.default());
  // Bumped on every startGame() call; a running loop (and any human
  // move handler from a superseded game) checks it still matches the
  // value it started with before writing state -- see SpectateGame's
  // gameIdRef for the same reasoning.
  const gameIdRef = useRef(0);
  const movesRef = useRef<string[]>([]);
  // The config a running game started with, read from every closure
  // below instead of the `config` state -- state updates are async and
  // this must stay stable for the lifetime of one game (again, same
  // reasoning as SpectateGame's use of its `gameConfig` parameter, but
  // here the human-move handler also needs it, outside startGame's own
  // closure).
  const activeConfigRef = useRef<PlayConfig>(DEFAULT_PLAY_CONFIG);

  const [phase, setPhase] = useState<GamePhase>("config");
  const [config, setConfig] = useState<PlayConfig>(DEFAULT_PLAY_CONFIG);
  const [fen, setFen] = useState(START_FEN);
  const [lastMove, setLastMove] = useState<Key[] | undefined>();
  // `turn`/`dests` mirror posRef.current so render never reads a ref
  // directly (React ref values aren't meant to drive what's rendered --
  // see the `movable` config below, which is the only place these are
  // read); both are recomputed every time the position changes.
  const [turn, setTurn] = useState<"white" | "black">("white");
  const [dests, setDests] = useState<Dests>(new Map());
  const [status, setStatus] = useState("");
  const [gameSeq, setGameSeq] = useState(0);

  const humanTurn = () =>
    (posRef.current.turn === "white") === (activeConfigRef.current.humanColor === "white");

  /** Ends the game if it's over, returning whether it did. */
  const finishIfOver = (gameId: number): boolean => {
    if (gameId !== gameIdRef.current) return true;
    if (posRef.current.isEnd()) {
      setStatus(outcome(posRef.current));
      setPhase("finished");
      return true;
    }
    if (posRef.current.halfmoves >= MAX_HALFMOVES) {
      setStatus("draw by the fifty-move rule");
      setPhase("finished");
      return true;
    }
    return false;
  };

  /** Plays `uci` if legal, updating both the game state and the board.
   * Returns whether it was legal. */
  const applyMove = (gameId: number, uci: string): boolean => {
    const move = parseUci(uci);
    if (!move || !posRef.current.isLegal(move)) return false;
    posRef.current.play(move);
    movesRef.current.push(uci);
    if (gameId === gameIdRef.current) {
      setFen(makeFen(posRef.current.toSetup()));
      setLastMove([uci.slice(0, 2), uci.slice(2, 4)] as Key[]);
      setTurn(posRef.current.turn);
      setDests(chessgroundDests(posRef.current));
    }
    return true;
  };

  /** Lets Bee-Mamba reply for as long as it's its turn (normally once,
   * but this also drives its opening move when the human plays black). */
  const runEngineTurn = async (gameId: number) => {
    while (gameId === gameIdRef.current && !humanTurn()) {
      if (finishIfOver(gameId)) return;
      setStatus(`${mambaEngine.name} thinking…`);

      let uci: string;
      try {
        uci = await mambaEngine.bestMove(movesRef.current, activeConfigRef.current.moveTimeMs);
      } catch (err) {
        if (gameId === gameIdRef.current) {
          setStatus(message(err));
          setPhase("finished");
        }
        return;
      }
      if (gameId !== gameIdRef.current) return;

      if (!applyMove(gameId, uci)) {
        setStatus(`${mambaEngine.name} played an illegal move: ${uci}`);
        setPhase("finished");
        return;
      }
    }
    if (finishIfOver(gameId)) return;
    setStatus("your move");
  };

  const startGame = async (playConfig: PlayConfig) => {
    const gameId = ++gameIdRef.current;

    posRef.current = Chess.default();
    movesRef.current = [];
    activeConfigRef.current = playConfig;
    setConfig(playConfig);
    setFen(START_FEN);
    setLastMove(undefined);
    setTurn("white");
    setDests(playConfig.humanColor === "white" ? chessgroundDests(posRef.current) : new Map());
    setPhase("playing");
    setStatus("connecting to the bridge…");
    setGameSeq(gameId);

    try {
      await mambaEngine.init();
    } catch (err) {
      if (gameId === gameIdRef.current) {
        setStatus(message(err));
        setPhase("finished");
      }
      return;
    }
    if (gameId !== gameIdRef.current) return;

    await runEngineTurn(gameId);
  };

  /** A piece was dropped on `dest`: chessground already restricted
   * `orig` to a legal source and `dest` to one of its legal
   * destinations (see `movable.dests` below), so the only ambiguity
   * left is pawn promotion -- try the plain move first, and only add
   * the queen-promotion suffix if the board actually calls for one. */
  const onHumanMove = (gameId: number) => (orig: Key, dest: Key) => {
    if (gameId !== gameIdRef.current || !humanTurn()) return;

    const plain = `${orig}${dest}`;
    const plainMove = parseUci(plain);
    const uci = plainMove && posRef.current.isLegal(plainMove) ? plain : `${plain}q`;

    if (!applyMove(gameId, uci)) return;
    if (finishIfOver(gameId)) return;
    void runEngineTurn(gameId);
  };

  const canMove = phase === "playing" && turn === config.humanColor;

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
      <h1>You vs {mambaEngine.name}</h1>
      {phase === "config" ? (
        <PlayConfigForm initial={config} onStart={(c) => void startGame(c)} />
      ) : (
        <>
          <Chessground
            config={{
              fen,
              lastMove,
              orientation: config.humanColor,
              turnColor: turn,
              movable: {
                free: false,
                color: canMove ? config.humanColor : undefined,
                dests: canMove ? dests : new Map(),
                events: { after: onHumanMove(gameSeq) },
              },
            }}
          />
          <p>{status}</p>
          {phase === "finished" && (
            <button onClick={() => setPhase("config")}>New game</button>
          )}
          <div style={{ width: "100%", maxWidth: 450 }}>
            <UciLogPanel
              key={`mamba-${gameSeq}`}
              name={mambaEngine.name}
              subscribe={(l) => mambaEngine.onLog(l)}
            />
          </div>
        </>
      )}
    </section>
  );
}

function PlayConfigForm({
  initial,
  onStart,
}: {
  initial: PlayConfig;
  onStart: (config: PlayConfig) => void;
}) {
  const [humanColor, setHumanColor] = useState(initial.humanColor);
  // Kept as a string while editing so the field can be temporarily
  // empty or mid-edit (e.g. "50") without snapping back to a number.
  const [moveTimeText, setMoveTimeText] = useState(String(initial.moveTimeMs));

  const config: PlayConfig = { humanColor, moveTimeMs: Number(moveTimeText) };
  const error = validatePlayConfig(config);

  return (
    <form
      style={{ display: "grid", gap: 12, justifyItems: "start" }}
      onSubmit={(e) => {
        e.preventDefault();
        if (!error) onStart(config);
      }}
    >
      <label style={{ display: "grid", gap: 4 }}>
        Play as
        <select value={humanColor} onChange={(e) => setHumanColor(e.target.value as "white" | "black")}>
          <option value="white">White</option>
          <option value="black">Black</option>
        </select>
      </label>
      <label style={{ display: "grid", gap: 4 }}>
        Engine time per move (ms)
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
