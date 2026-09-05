import { useEffect, useRef, useState, type RefObject } from "react";
import { Chess } from "chessops/chess";
import { parseFen } from "chessops/fen";
import { chessgroundDests } from "chessops/compat";
import type { Key, Dests } from "@lichess-org/chessground/types";
import { Chessground } from "./Chessground";
import { UciLogPanel } from "./UciLogPanel";
import { SearchStatsPanel } from "./SearchStatsPanel";
import { EvalBar } from "./EvalBar";
import type { UciLogLine } from "./engine";
import type { Participant } from "./participant";
import {
  createGame,
  getGame,
  postMove,
  subscribeToGameEvents,
  type Color,
  type GameEvent,
  type GameSnapshot,
  type ParticipantRequest,
} from "./labClient";

/** Board position before Lab has responded to `createGame` yet. */
const START_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

/** How this component polls `GET /api/games/:id` to stay in sync while
 * connected, as a fallback under the live WebSocket stream -- see
 * `useEffect` below for why both exist rather than relying on the
 * socket alone. */
const POLL_INTERVAL_MS = 500;

/**
 * Plays one game between whatever `white`/`black` are configured as --
 * any mix of human and bots. Per #69/67b: this component owns none of
 * position/turn/legality/result itself. It asks Bee Lab to create a
 * game, renders whatever `GameSnapshot` Lab reports (via an initial
 * `getGame` plus a `GameEvent::Updated` stream, with polling as a
 * fallback in case the socket drops), and for a human's move asks Lab
 * to apply it via `postMove` -- trusting Lab's answer either way. Lab
 * itself drives any engine-controlled side automatically; this
 * component never talks to an engine process directly.
 *
 * Bee-Mamba has no Lab-side engine yet (see #66/#70) -- picking it for
 * either slot shows an "unavailable" message instead of attempting to
 * create a game, rather than silently falling back to some other path.
 *
 * The parent renders this with a `key` that changes per game, so a new
 * game is a fresh mount rather than this instance being reset in
 * place.
 */
export function Game({
  white,
  black,
  onBackToSetup,
}: {
  white: Participant;
  black: Participant;
  onBackToSetup: () => void;
}) {
  const gameIdRef = useRef<string | null>(null);
  // Every raw GameEvent this game has seen, for EvalBar/SearchStatsPanel/
  // UciLogPanel's `subscribe` props below -- see `logSubscribeFor`.
  // Declared before the mount effect below since that effect forwards
  // events into this set as they arrive over the WebSocket.
  const logListenersRef = useRef(new Set<(event: GameEvent) => void>());

  const [snapshot, setSnapshot] = useState<GameSnapshot | null>(null);
  const [status, setStatus] = useState("connecting to Bee Lab…");
  const [unavailable, setUnavailable] = useState<string | null>(null);

  const participantFor = (color: Color): Participant => (color === "white" ? white : black);
  const nameFor = (color: Color): string => {
    const participant = participantFor(color);
    if (participant.kind === "human") return "You";
    return participant.kind === "bee-mamba" ? "Bee-Mamba" : participant.kind;
  };

  const humanTurnColor = (): Color | null => {
    if (!snapshot || snapshot.status !== "running") return null;
    const turn: Color = snapshot.moves.length % 2 === 0 ? "white" : "black";
    return participantFor(turn).kind === "human" ? turn : null;
  };

  const applySnapshot = (next: GameSnapshot) => {
    setSnapshot(next);
    if (next.status === "running") {
      setStatus(humanTurnColorFor(next, white, black) ? "your move" : "thinking…");
    } else if (next.status === "finished") {
      setStatus(next.result === "draw" ? "game over — draw" : `game over — ${next.result.replace("_wins", "")} wins`);
    } else {
      setStatus(`game aborted: ${next.reason}`);
    }
  };

  // Creates the Lab-side game and starts polling/subscribing. Runs
  // once per mount, i.e. once per game (see the component doc comment
  // above) -- guarded against React StrictMode's dev-only double-invoke
  // the same way the previous client-driven version was: a `cancelled`
  // flag, checked before ever touching state, so a superseded first run
  // never races a second `createGame` call against this one.
  useEffect(() => {
    let cancelled = false;
    let unsubscribe: (() => void) | null = null;
    let pollTimer: ReturnType<typeof setInterval> | null = null;

    const bad = badParticipant(white) ?? badParticipant(black);
    if (bad) {
      setUnavailable(bad);
      return;
    }

    void (async () => {
      let created: GameSnapshot;
      try {
        created = await createGame({
          white: toParticipantRequest(white),
          black: toParticipantRequest(black),
          moveTimeMs: moveTimeMsFor(white, black),
        });
      } catch (err) {
        if (cancelled) return;
        setStatus(message(err));
        return;
      }
      if (cancelled) return;

      gameIdRef.current = created.id;
      applySnapshot(created);

      // The WebSocket stream is the primary way this component learns
      // about moves/status changes (and the only way it sees live UCI
      // traffic for EvalBar/SearchStatsPanel/UciLogPanel below) --
      // polling `getGame` is purely a fallback for a dropped/never-
      // connected socket, since `GET /api/games/:id` is the
      // authoritative resync mechanism regardless (see labClient.ts).
      unsubscribe = subscribeToGameEvents(created.id, (event: GameEvent) => {
        if (cancelled) return;
        if (event.type === "updated") applySnapshot(event.snapshot);
        for (const listener of logListenersRef.current) listener(event);
      });

      pollTimer = setInterval(() => {
        void getGame(created.id).then(
          (fresh) => {
            if (!cancelled) applySnapshot(fresh);
          },
          () => {
            /* transient fetch failure -- next poll retries */
          },
        );
      }, POLL_INTERVAL_MS);
    })();

    return () => {
      cancelled = true;
      unsubscribe?.();
      if (pollTimer) clearInterval(pollTimer);
    };
    // Intentionally empty deps: white/black/onBackToSetup are fixed
    // for this component's lifetime (a new game remounts it via a
    // fresh `key` instead of these props changing in place).
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  /** A piece was dropped on `dest`: chessground already restricted
   * `orig` to a legal source and `dest` to one of its legal
   * destinations (see `movable.dests` below, computed from the same
   * position Lab holds), so the only ambiguity left is pawn promotion
   * -- try the plain move first, and only add the queen-promotion
   * suffix if Lab rejects it. Lab is the one deciding legality either
   * way; this component doesn't pre-validate anything, just picks
   * which move text to send. */
  const onHumanMove = (orig: Key, dest: Key) => {
    const gameId = gameIdRef.current;
    if (!gameId || !humanTurnColor()) return;

    const plain = `${orig}${dest}`;

    void postMove(gameId, plain).then(
      (next) => applySnapshot(next),
      async () => {
        // Plain move was rejected -- the only reason a chessground-
        // legal-looking drop would be is a pawn reaching the back rank
        // needing a promotion suffix Lab requires explicitly.
        try {
          const withQueen = await postMove(gameId, `${plain}q`);
          applySnapshot(withQueen);
        } catch (err) {
          setStatus(message(err));
        }
      },
    );
  };

  if (unavailable) {
    return (
      <section style={{ display: "grid", gap: 8, textAlign: "center" }}>
        <p>{unavailable}</p>
        <button onClick={onBackToSetup}>Back to setup</button>
      </section>
    );
  }

  const canMoveColor = humanTurnColor();
  const fen = snapshot?.fen ?? START_FEN;
  const lastMove = lastMoveKeys(snapshot);
  const dests = canMoveColor ? chessgroundDestsFromFen(fen) : new Map<Key, Key[]>();
  const finished = snapshot ? snapshot.status !== "running" : false;

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
        {nameFor("white")} (white) vs {nameFor("black")} (black)
      </h1>
      <div style={{ display: "flex", gap: 8, alignItems: "flex-start" }}>
        {white.kind !== "human" && (
          <EvalBar color="white" subscribe={logSubscribeFor(logListenersRef, "white")} />
        )}
        <Chessground
          config={{
            fen,
            lastMove,
            coordinatesOnSquares: true,
            viewOnly: canMoveColor === null,
            turnColor: snapshot && snapshot.moves.length % 2 === 1 ? "black" : "white",
            movable: {
              free: false,
              color: canMoveColor ?? undefined,
              dests,
              events: { after: onHumanMove },
            },
          }}
        />
        {black.kind !== "human" && (
          <EvalBar color="black" subscribe={logSubscribeFor(logListenersRef, "black")} />
        )}
      </div>
      <p>{status}</p>
      {finished && <button onClick={onBackToSetup}>New game</button>}
      <BotPanels color="white" participant={white} logListenersRef={logListenersRef} />
      <BotPanels color="black" participant={black} logListenersRef={logListenersRef} />
    </section>
  );
}

/** Renders the stats + log panels for one slot, if it's a bot -- human
 * slots have no engine traffic and so nothing to show here. */
function BotPanels({
  color,
  participant,
  logListenersRef,
}: {
  color: Color;
  participant: Participant;
  logListenersRef: RefObject<Set<(event: GameEvent) => void>>;
}) {
  if (participant.kind === "human") return null;

  const name = participant.kind === "bee-mamba" ? "Bee-Mamba" : participant.kind;
  const subscribe = logSubscribeFor(logListenersRef, color);

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 8, width: "100%", maxWidth: 900 }}>
      <SearchStatsPanel name={`${name} (${color})`} subscribe={subscribe} />
      <UciLogPanel name={`${name} (${color})`} subscribe={subscribe} />
    </div>
  );
}

/** Adapts the raw `GameEvent` stream (shared across both colors) into
 * the per-color `UciLogLine` subscription shape `EvalBar`/
 * `SearchStatsPanel`/`UciLogPanel` already expect -- this is what lets
 * all three keep working completely unchanged against Lab's WebSocket
 * instead of a direct browser-to-engine connection. */
function logSubscribeFor(
  logListenersRef: RefObject<Set<(event: GameEvent) => void>>,
  color: Color,
): (listener: (line: UciLogLine) => void) => () => void {
  return (listener) => {
    const handler = (event: GameEvent) => {
      if (event.type !== "uci" || event.color !== color) return;
      listener({ direction: event.direction, text: event.line, timestamp: Date.now() });
    };
    logListenersRef.current.add(handler);
    return () => logListenersRef.current.delete(handler);
  };
}

const message = (err: unknown) => (err instanceof Error ? err.message : String(err));

/** Bee-Mamba has no Lab-side engine yet (#66/#70) -- returns an
 * explanatory message if `participant` picks it, else `null`. */
function badParticipant(participant: Participant): string | null {
  return participant.kind === "bee-mamba"
    ? "Bee-Mamba isn't available yet during the Bee Lab migration (see #66/#70)."
    : null;
}

/** Maps a frontend `Participant` to the request shape `createGame`
 * expects, or `undefined` for a human slot. */
function toParticipantRequest(participant: Participant): ParticipantRequest | undefined {
  switch (participant.kind) {
    case "human":
      return undefined;
    case "stockfish":
      return {
        engine: "stockfish",
        options: { UCI_LimitStrength: true, UCI_Elo: participant.elo },
        debug: participant.debug,
      };
    case "bee":
      return { engine: "bee", debug: participant.debug };
    case "bee-mamba":
      // Unreachable: badParticipant already redirected to the
      // unavailable-message screen before createGame is ever called.
      return undefined;
  }
}

/** Lab's `POST /api/games` takes one `move_time_ms` for the whole
 * game, but the frontend's `Participant` model carries it per side --
 * a real, documented simplification (see labClient.ts's docs) rather
 * than something silently papered over. Prefers White's configured
 * value, then Black's, since a single shared budget has to pick one. */
function moveTimeMsFor(white: Participant, black: Participant): number | undefined {
  if ("moveTimeMs" in white) return white.moveTimeMs;
  if ("moveTimeMs" in black) return black.moveTimeMs;
  return undefined;
}

function humanTurnColorFor(snapshot: GameSnapshot, white: Participant, black: Participant): boolean {
  if (snapshot.status !== "running") return false;
  const turn: Color = snapshot.moves.length % 2 === 0 ? "white" : "black";
  const participant = turn === "white" ? white : black;
  return participant.kind === "human";
}

function lastMoveKeys(snapshot: GameSnapshot | null): Key[] | undefined {
  const last = snapshot?.moves.at(-1);
  if (!last) return undefined;
  return [last.slice(0, 2), last.slice(2, 4)] as Key[];
}

/**
 * Computes chessground's `Dests` (legal destination squares per piece,
 * needed for drag-and-drop to work at all) from `snapshot`'s FEN via a
 * read-only `chessops` position. This is a **UI affordance only**, not
 * a second source of chess-rules truth: Lab still decides whether any
 * given move is actually legal when `postMove` is called, and its
 * answer always wins regardless of what this function computed (a
 * stale/wrong `dests` would only ever make chessground offer a square
 * that Lab then rejects, never the reverse). Recomputed fresh from
 * `snapshot` on every call rather than cached, since there's no
 * meaningful state to keep here beyond "what does this FEN allow" --
 * this deliberately does not construct a `Chess` object that
 * accumulates moves over the game the way the pre-#69 client-owned
 * position did.
 */
function chessgroundDestsFromFen(fen: string): Dests {
  const setup = parseFen(fen);
  if (setup.isErr) return new Map();
  const position = Chess.fromSetup(setup.value);
  if (position.isErr) return new Map();
  return chessgroundDests(position.value);
}
