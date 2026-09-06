import { useEffect, useRef, useState, type RefObject } from "react";
import { Chess } from "chessops/chess";
import { parseFen } from "chessops/fen";
import { chessgroundDests } from "chessops/compat";
import type { Key, Dests } from "@lichess-org/chessground/types";
import { Chessground } from "./Chessground";
import { UciLogPanel } from "./UciLogPanel";
import { SearchStatsPanel } from "./SearchStatsPanel";
import { EvalBar } from "./EvalBar";
import { Button } from "./components/ui/Button";
import { Inline, Stack } from "./components/ui/Stack";
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
  type ParticipantInfo,
  type ParticipantRequest,
} from "./labClient";

/** Board position before Lab has responded yet. */
const START_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

/** How this component polls `GET /api/games/:id` to stay in sync while
 * connected, as a fallback under the live WebSocket stream -- see
 * `useEffect` below for why both exist rather than relying on the
 * socket alone. */
const POLL_INTERVAL_MS = 500;

/** Either start a brand-new game (the setup screen's `Participant`
 * configuration for each side) or resume an existing one by id alone
 * -- see the component doc comment below for why resuming needs
 * nothing but the id. */
export type GameSource = { kind: "start"; white: Participant; black: Participant } | { kind: "resume"; gameId: string };

/**
 * Plays one game. Per #69/67b: this component owns none of
 * position/turn/legality/result/participant-configuration itself --
 * `GameSnapshot` (from Bee Lab) is the single source of truth for all
 * of it, including *who is playing each side* (`ParticipantInfo`),
 * which is why a page refresh can resume a game by persisting only its
 * id (see `App.tsx`'s `?game=` URL param) rather than needing to
 * remember the original `Participant` configuration too -- Lab already
 * knows.
 *
 * For a fresh game (`source.kind === "start"`), this component calls
 * `createGame` with `white`/`black` mapped from the setup screen's
 * `Participant` config, then renders from the returned snapshot exactly
 * like a resumed one from then on. For a resumed game
 * (`source.kind === "resume"`), it skips straight to `getGame(gameId)`.
 * Either way, once a snapshot exists, it's kept fresh via the
 * WebSocket event stream primarily, with polling `GET /api/games/:id`
 * as a fallback for a dropped/never-connected socket -- `GET` is the
 * authoritative resync mechanism regardless, so polling it is never
 * wrong, just occasionally redundant with a live event that already
 * arrived. A human's move goes through `postMove`, trusting Lab's
 * answer either way.
 *
 * Bee-Mamba has no Lab-side engine yet (see #66/#70) -- picking it for
 * either slot on a fresh game shows an "unavailable" message instead
 * of attempting to create a game.
 *
 * The parent renders this with a `key` that changes per game, so a new
 * game is a fresh mount rather than this instance being reset in
 * place.
 */
export function Game({
  source,
  onGameCreated,
  onBackToSetup,
  onOpenExperiment,
}: {
  source: GameSource;
  /** Called once with the game's id, as soon as it's known -- for a
   * "start"-kind source, that's only after `createGame` resolves (its
   * id doesn't exist beforehand); for "resume", it's the same id
   * `source` already carried. Lets `App.tsx` keep the `?game=` URL
   * param in sync even for a freshly started game, not just a resumed
   * one. */
  onGameCreated: (gameId: string) => void;
  onBackToSetup: () => void;
  /** Called with the game's `experiment_id` when the user asks to go
   * back to it -- only ever offered when the snapshot actually has one
   * (see the "Back to experiment" button below), i.e. this game was
   * created by an A/B experiment (#109/#111) rather than started from
   * the ordinary setup screen. */
  onOpenExperiment: (experimentId: string) => void;
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

  const participantInfoFor = (color: Color): ParticipantInfo | null =>
    snapshot ? (color === "white" ? snapshot.white : snapshot.black) : null;
  const nameFor = (color: Color): string => {
    const info = participantInfoFor(color);
    if (!info) return "…";
    return info.kind === "human" ? "You" : info.name;
  };

  const humanTurnColor = (): Color | null => {
    if (!snapshot || snapshot.status !== "running") return null;
    const turn: Color = snapshot.moves.length % 2 === 0 ? "white" : "black";
    return participantInfoFor(turn)?.kind === "human" ? turn : null;
  };

  const applySnapshot = (next: GameSnapshot) => {
    setSnapshot(next);
    if (next.status === "running") {
      const turn: Color = next.moves.length % 2 === 0 ? "white" : "black";
      const onMove = turn === "white" ? next.white : next.black;
      setStatus(onMove.kind === "human" ? "your move" : "thinking…");
    } else if (next.status === "finished") {
      setStatus(next.result === "draw" ? "game over — draw" : `game over — ${next.result.replace("_wins", "")} wins`);
    } else {
      setStatus(`game aborted: ${next.reason}`);
    }
  };

  // Starts or resumes the Lab-side game and begins polling/subscribing.
  // Runs once per mount, i.e. once per game (see the component doc
  // comment above) -- guarded against React StrictMode's dev-only
  // double-invoke the same way the previous client-driven version was:
  // a `cancelled` flag, checked before ever touching state, so a
  // superseded first run never races a second `createGame`/`getGame`
  // call against this one.
  useEffect(() => {
    let cancelled = false;
    let unsubscribe: (() => void) | null = null;
    let pollTimer: ReturnType<typeof setInterval> | null = null;

    if (source.kind === "start") {
      const bad = badParticipant(source.white) ?? badParticipant(source.black);
      if (bad) {
        setUnavailable(bad);
        return;
      }
    }

    void (async () => {
      let initial: GameSnapshot;
      try {
        initial =
          source.kind === "resume"
            ? await getGame(source.gameId)
            : await createGame({
                white: toParticipantRequest(source.white),
                black: toParticipantRequest(source.black),
                moveTimeMs: moveTimeMsFor(source.white, source.black),
              });
      } catch (err) {
        if (cancelled) return;
        setStatus(message(err));
        return;
      }
      if (cancelled) return;

      gameIdRef.current = initial.id;
      onGameCreated(initial.id);
      applySnapshot(initial);

      // The WebSocket stream is the primary way this component learns
      // about moves/status changes (and the only way it sees live UCI
      // traffic for EvalBar/SearchStatsPanel/UciLogPanel below) --
      // polling `getGame` is purely a fallback for a dropped/never-
      // connected socket, since `GET /api/games/:id` is the
      // authoritative resync mechanism regardless (see labClient.ts).
      unsubscribe = subscribeToGameEvents(initial.id, (event: GameEvent) => {
        if (cancelled) return;
        if (event.type === "updated") applySnapshot(event.snapshot);
        for (const listener of logListenersRef.current) listener(event);
      });

      pollTimer = setInterval(() => {
        void getGame(initial.id).then(
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
    // Intentionally empty deps: source/onBackToSetup are fixed for
    // this component's lifetime (a new game remounts it via a fresh
    // `key` instead of these props changing in place).
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
      <Stack gap={2} align="center" className="text-center">
        <p className="m-0 text-sm text-text">{unavailable}</p>
        <Button onClick={onBackToSetup}>Back to setup</Button>
      </Stack>
    );
  }

  const canMoveColor = humanTurnColor();
  const fen = snapshot?.fen ?? START_FEN;
  const lastMove = lastMoveKeys(snapshot);
  // Put the human player's pieces nearest them. White remains the default
  // while the snapshot is loading and for games without a human Black side.
  const orientation = snapshot?.black.kind === "human" ? "black" : "white";
  const dests = canMoveColor ? chessgroundDestsFromFen(fen) : new Map<Key, Key[]>();
  const finished = snapshot ? snapshot.status !== "running" : false;
  const experimentId = snapshot?.experiment_id ?? null;

  return (
    <Stack gap={2} align="center" className="w-full text-center">
      <h1 className="text-2xl font-medium">
        {nameFor("white")} (white) vs {nameFor("black")} (black)
      </h1>
      {experimentId && (
        <Button variant="secondary" onClick={() => onOpenExperiment(experimentId)}>
          ← Back to experiment
        </Button>
      )}
      <Inline gap={2} align="start" className="justify-center">
        {participantInfoFor("white")?.kind === "engine" && (
          <EvalBar color="white" subscribe={logSubscribeFor(logListenersRef, "white")} />
        )}
        <Chessground
          config={{
            fen,
            lastMove,
            orientation,
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
        {participantInfoFor("black")?.kind === "engine" && (
          <EvalBar color="black" subscribe={logSubscribeFor(logListenersRef, "black")} />
        )}
      </Inline>
      <p className="m-0 text-sm text-muted">{status}</p>
      {finished && <Button onClick={onBackToSetup}>New game</Button>}
      <BotPanels color="white" info={participantInfoFor("white")} logListenersRef={logListenersRef} />
      <BotPanels color="black" info={participantInfoFor("black")} logListenersRef={logListenersRef} />
    </Stack>
  );
}

/** Renders the stats + log panels for one slot, if it's engine-driven
 * -- a human slot has no engine traffic and so nothing to show here.
 * `info` is `null` until the initial snapshot arrives. */
function BotPanels({
  color,
  info,
  logListenersRef,
}: {
  color: Color;
  info: ParticipantInfo | null;
  logListenersRef: RefObject<Set<(event: GameEvent) => void>>;
}) {
  if (!info || info.kind === "human") return null;

  const subscribe = logSubscribeFor(logListenersRef, color);

  return (
    <Stack gap={2} className="w-full max-w-[900px]">
      <SearchStatsPanel name={`${info.name} (${color})`} subscribe={subscribe} />
      <UciLogPanel name={`${info.name} (${color})`} subscribe={subscribe} />
    </Stack>
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
 * explanatory message if `participant` picks it, else `null`. Only
 * relevant for a fresh game: a resumed one's participants already went
 * through this check when it was first created. */
function badParticipant(participant: Participant): string | null {
  return participant.kind === "bee-mamba"
    ? "Bee-Mamba isn't available yet during the Bee Lab migration (see #66/#70)."
    : null;
}

/** Maps a frontend `Participant` (the setup screen's configuration) to
 * the request shape `createGame` expects, or `undefined` for a human
 * slot. Only used when starting a fresh game -- a resumed game's
 * participants come from the snapshot's `ParticipantInfo` instead. */
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
      return { engine: "bee", options: participant.options, debug: participant.debug };
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
