import { useEffect, useRef, useState, type CSSProperties } from "react";
import { Chess } from "chessops/chess";
import { makeFen } from "chessops/fen";
import { chessgroundDests } from "chessops/compat";
import { makeSanAndPlay } from "chessops/san";
import { parseUci } from "chessops/util";
import type { Key, Dests } from "@lichess-org/chessground/types";
import { Chessground } from "./Chessground";
import { createBotClient, type UciClient } from "./engine";
import { UciLogPanel } from "./UciLogPanel";
import { SearchStatsPanel } from "./SearchStatsPanel";
import { EvalBar } from "./EvalBar";
import type { Participant } from "./participant";
import { pushPly, startNav, type Nav } from "./gameHistory";

/** Claim a draw once the fifty-move counter is full; chessops does not. */
const MAX_HALFMOVES = 100;

/** Board position before any move; avoids reading the ref during render. */
const START_FEN = makeFen(Chess.default().toSetup());

type Color = "white" | "black";

/**
 * Plays one game between whatever `white`/`black` are configured as --
 * any mix of human and bots. A bot slot gets its own `UciClient`
 * instance created fresh for this game (so the same bot kind on both
 * sides never shares a connection/position); a human slot has none,
 * and moves come from dragging pieces on the board instead.
 *
 * The parent renders this with a `key` that changes per game, so a new
 * game is a fresh mount rather than this instance being reset in
 * place -- there is deliberately no "is this still the current game"
 * guard anywhere below, since a superseded game's in-flight promises
 * belong to an unmounted component's own closure/refs and can't reach
 * a new game's state no matter how long they take to settle.
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
  const posRef = useRef(Chess.default());
  const movesRef = useRef<string[]>([]);
  const clientsRef = useRef<{ white?: UciClient; black?: UciClient }>({});

  const [nav, setNav] = useState<Nav>(() => startNav(START_FEN));
  const [turn, setTurn] = useState<Color>("white");
  const [dests, setDests] = useState<Dests>(new Map());
  const [status, setStatus] = useState("starting…");
  const [finished, setFinished] = useState(false);

  const clientFor = (color: Color): UciClient | undefined => clientsRef.current[color];
  const participantFor = (color: Color): Participant => (color === "white" ? white : black);
  const nameFor = (color: Color): string => {
    const participant = participantFor(color);
    return participant.kind === "human" ? "You" : (clientFor(color)?.name ?? participant.kind);
  };

  const humanTurnColor = (): Color | null => {
    if (white.kind === "human" && posRef.current.turn === "white") return "white";
    if (black.kind === "human" && posRef.current.turn === "black") return "black";
    return null;
  };

  const syncBoardState = () => {
    setTurn(posRef.current.turn);
    setDests(humanTurnColor() ? chessgroundDests(posRef.current) : new Map());
  };

  /** Ends the game if it's over, returning whether it did. */
  const finishIfOver = (): boolean => {
    if (posRef.current.isEnd()) {
      setStatus(outcome(posRef.current));
      setFinished(true);
      return true;
    }
    if (posRef.current.halfmoves >= MAX_HALFMOVES) {
      setStatus("draw by the fifty-move rule");
      setFinished(true);
      return true;
    }
    return false;
  };

  /** Plays `uci` if legal, updating both the game state and the board.
   * Returns whether it was legal. */
  const applyMove = (uci: string): boolean => {
    const move = parseUci(uci);
    if (!move || !posRef.current.isLegal(move)) return false;
    // makeSanAndPlay reads the SAN off `posRef.current` *before* playing
    // the move, then plays it -- same mutation as the old bare `.play`,
    // plus the string this history list wants.
    const san = makeSanAndPlay(posRef.current, move);
    movesRef.current.push(uci);
    const lastMove = [uci.slice(0, 2), uci.slice(2, 4)] as Key[];
    setNav((prev) => pushPly(prev, { fen: makeFen(posRef.current.toSetup()), lastMove, san }));
    syncBoardState();
    return true;
  };

  /** Drives bot turns for as long as it's a bot's move (may be more
   * than one ply in a row if both slots are bots, or zero if it's
   * immediately a human's turn).
   *
   * `isCancelled` is checked right after each `bestMove` await
   * resolves, before touching any shared ref/state -- it's how the
   * mount effect below aborts an in-progress loop from a superseded
   * (StrictMode double-invoked) run instead of letting its `bestMove`
   * reply land as a second, racing move applied on top of a game a
   * fresher run has since moved on from. Defaults to "never
   * cancelled" for `onHumanMove`'s call site, a real event handler
   * with no such superseded-run concern. */
  const runBotTurns = async (isCancelled: () => boolean = () => false) => {
    while (!finishIfOver()) {
      const color: Color = posRef.current.turn;
      const participant = participantFor(color);
      if (participant.kind === "human") {
        setStatus("your move");
        return;
      }

      const client = clientFor(color);
      if (!client) return; // should not happen: every bot slot gets a client below
      setStatus(`${client.name} thinking…`);

      let uci: string;
      try {
        uci = await client.bestMove(movesRef.current, participant.moveTimeMs);
      } catch (err) {
        if (isCancelled()) return;
        setStatus(message(err));
        setFinished(true);
        return;
      }

      if (isCancelled()) return;

      if (!applyMove(uci)) {
        setStatus(`${client.name} played an illegal move: ${uci}`);
        setFinished(true);
        return;
      }
    }
  };

  // Connects each bot slot's client and starts the game loop. Runs
  // once per mount, i.e. once per game (see the component doc comment
  // above for why "on mount" and "on new game" are the same event
  // here) -- *except* under React StrictMode's dev-only
  // mount/unmount/remount cycle, which deliberately double-invokes
  // every effect to surface exactly the bug this guards against: an
  // effect with no cleanup runs its whole async body twice, and here
  // that meant two concurrent `runBotTurns()` loops sharing the same
  // `movesRef`/`posRef`, both calling `bestMove` from `startpos`
  // before either had applied a move -- the second reply then landed
  // as a stale, now-illegal move once the first had already advanced
  // the game. `cancelled` (flipped by this effect's cleanup) is
  // checked before ever touching shared refs/state and at every
  // meaningful await boundary, so a StrictMode-aborted first run's
  // continuations become no-ops instead of a second, racing game loop.
  // `createdClients` lets the cleanup close every client this run
  // opened, even ones aborted before `runBotTurns` ever started.
  useEffect(() => {
    let cancelled = false;
    const createdClients: UciClient[] = [];

    void (async () => {
      setStatus("connecting to the bridge…");

      const pending: Promise<void>[] = [];
      for (const color of ["white", "black"] as const) {
        const participant = participantFor(color);
        if (participant.kind === "human") continue;
        const client = createBotClient(participant.kind);
        createdClients.push(client);
        if (cancelled) return; // aborted before this slot's client could be tracked
        clientsRef.current[color] = client;
        pending.push(client.init());
        if (participant.kind === "stockfish") {
          pending.push(
            client
              .setOption("UCI_LimitStrength", true)
              .then(() => client.setOption("UCI_Elo", participant.elo)),
          );
        }
        if ("debug" in participant) {
          pending.push(client.setDebug(participant.debug));
        }
      }

      try {
        await Promise.all(pending);
      } catch (err) {
        if (cancelled) return;
        setStatus(message(err));
        setFinished(true);
        return;
      }
      if (cancelled) return;

      if (cancelled) return;
      syncBoardState();
      await runBotTurns(() => cancelled);
    })();

    return () => {
      cancelled = true;
      for (const client of createdClients) client.close();
    };
    // Intentionally empty deps: white/black/onBackToSetup are fixed
    // for this component's lifetime (a new game remounts it via a
    // fresh `key` instead of these props changing in place).
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  /** A piece was dropped on `dest`: chessground already restricted
   * `orig` to a legal source and `dest` to one of its legal
   * destinations (see `movable.dests` below), so the only ambiguity
   * left is pawn promotion -- try the plain move first, and only add
   * the queen-promotion suffix if the board actually calls for one. */
  const onHumanMove = (orig: Key, dest: Key) => {
    if (!following || !humanTurnColor()) return;

    const plain = `${orig}${dest}`;
    const plainMove = parseUci(plain);
    const uci = plainMove && posRef.current.isLegal(plainMove) ? plain : `${plain}q`;

    if (!applyMove(uci)) return;
    if (finishIfOver()) return;
    void runBotTurns();
  };

  // Stepping through history is view-only, regardless of whose turn it
  // really is -- there's no "play a move from an earlier position"
  // here, only browsing (see the issue this closes: "go forward and
  // backward in time", not branch the game).
  const following = nav.viewIndex === nav.history.length - 1;
  const canMoveColor = following ? humanTurnColor() : null;
  const shownPly = nav.history[nav.viewIndex];

  // Arrow keys step through history the same as the Prev/Next buttons.
  useEffect(() => {
    const onKeyDown = (e: KeyboardEvent) => {
      if (e.key === "ArrowLeft") {
        setNav((prev) => ({ ...prev, viewIndex: Math.max(0, prev.viewIndex - 1) }));
      } else if (e.key === "ArrowRight") {
        setNav((prev) => ({
          ...prev,
          viewIndex: Math.min(prev.history.length - 1, prev.viewIndex + 1),
        }));
      }
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, []);

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
        {clientFor("white") && (
          <EvalBar color="white" subscribe={(l) => clientFor("white")!.onLog(l)} />
        )}
        <Chessground
          config={{
            fen: shownPly.fen,
            lastMove: shownPly.lastMove,
            // Default `coordinates: true` floats rank/file labels a few px
            // inside the board's own edge, overlapping back-rank pieces on
            // our fixed 480x480 board (#51). on-square labels avoid that.
            coordinatesOnSquares: true,
            viewOnly: canMoveColor === null,
            turnColor: turn,
            movable: {
              free: false,
              color: canMoveColor ?? undefined,
              dests: canMoveColor ? dests : new Map(),
              events: { after: onHumanMove },
            },
          }}
        />
        {clientFor("black") && (
          <EvalBar color="black" subscribe={(l) => clientFor("black")!.onLog(l)} />
        )}
        <MoveNav
          nav={nav}
          status={status}
          onSelect={(viewIndex) => setNav((prev) => ({ ...prev, viewIndex }))}
        />
      </div>
      {finished && <button onClick={onBackToSetup}>New game</button>}
      <BotPanels color="white" participant={white} client={clientFor("white")} />
      <BotPanels color="black" participant={black} client={clientFor("black")} />
    </section>
  );
}

/** Move list (click any move to jump to the position right after it)
 * plus Prev/Next stepping and a way back to the live position -- the
 * board always shows whatever ply is selected here, the arrow keys
 * step it, and the underlying game (and any bot replying) keeps
 * running regardless of what's currently on screen.
 *
 * Laid out lichess/chess.com-style: a boxed panel to the right of the
 * board, same height as it, with a scrolling two-column move table on
 * top and the step controls pinned to the bottom. */
function MoveNav({
  nav,
  status,
  onSelect,
}: {
  nav: Nav;
  status: string;
  onSelect: (viewIndex: number) => void;
}) {
  const following = nav.viewIndex === nav.history.length - 1;
  const atStart = nav.viewIndex === 0;

  // Pair up (white, black) plies per move number for the two-column
  // table; a game ending mid-move leaves `black` undefined.
  const moves = nav.history.slice(1);
  const rows: { num: number; white: { ply: (typeof moves)[number]; viewIndex: number }; black?: { ply: (typeof moves)[number]; viewIndex: number } }[] = [];
  for (let i = 0; i < moves.length; i += 2) {
    rows.push({
      num: i / 2 + 1,
      white: { ply: moves[i], viewIndex: i + 1 },
      black: moves[i + 1] ? { ply: moves[i + 1], viewIndex: i + 2 } : undefined,
    });
  }

  const cellStyle = (viewIndex: number): CSSProperties => ({
    padding: "2px 6px",
    borderRadius: 4,
    cursor: "pointer",
    color: "var(--text-h)",
    fontWeight: viewIndex === nav.viewIndex ? "bold" : "normal",
    background: viewIndex === nav.viewIndex ? "var(--accent-bg)" : "transparent",
  });

  return (
    <div
      style={{
        display: "flex",
        flexDirection: "column",
        width: 260,
        height: 480,
        border: "1px solid var(--border)",
        borderRadius: 6,
        overflow: "hidden",
        textAlign: "left",
        background: "var(--code-bg)",
        color: "var(--text)",
      }}
    >
      <div style={{ padding: "6px 10px", borderBottom: "1px solid var(--border)", fontSize: 14 }}>
        {status}
      </div>
      <div style={{ flex: 1, overflowY: "auto" }}>
        <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 14 }}>
          <tbody>
            {rows.map((row) => (
              <tr key={row.num}>
                <td style={{ padding: "2px 6px", color: "var(--text)", width: 32 }}>{row.num}.</td>
                <td style={cellStyle(row.white.viewIndex)} onClick={() => onSelect(row.white.viewIndex)}>
                  {row.white.ply.san}
                </td>
                <td
                  style={row.black ? cellStyle(row.black.viewIndex) : undefined}
                  onClick={row.black ? () => onSelect(row.black!.viewIndex) : undefined}
                >
                  {row.black?.ply.san ?? ""}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <div style={{ display: "flex", borderTop: "1px solid var(--border)" }}>
        <button type="button" disabled={atStart} onClick={() => onSelect(0)} style={navButtonStyle}>
          |◀
        </button>
        <button
          type="button"
          disabled={atStart}
          onClick={() => onSelect(nav.viewIndex - 1)}
          style={navButtonStyle}
        >
          ◀
        </button>
        <button
          type="button"
          disabled={following}
          onClick={() => onSelect(nav.viewIndex + 1)}
          style={navButtonStyle}
        >
          ▶
        </button>
        <button
          type="button"
          disabled={following}
          onClick={() => onSelect(nav.history.length - 1)}
          style={navButtonStyle}
        >
          ▶|
        </button>
      </div>
    </div>
  );
}

const navButtonStyle: CSSProperties = {
  flex: 1,
  padding: "8px 0",
  border: "none",
  borderRight: "1px solid var(--border)",
  background: "transparent",
  color: "var(--text-h)",
  cursor: "pointer",
};

/** Renders the stats + log panels for one slot, if it's a bot -- human
 * slots have no UciClient and so nothing to show here. */
function BotPanels({
  color,
  participant,
  client,
}: {
  color: Color;
  participant: Participant;
  client: UciClient | undefined;
}) {
  if (participant.kind === "human" || !client) return null;

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 8, width: "100%", maxWidth: 900 }}>
      <SearchStatsPanel name={`${client.name} (${color})`} subscribe={(l) => client.onLog(l)} />
      <UciLogPanel name={`${client.name} (${color})`} subscribe={(l) => client.onLog(l)} />
    </div>
  );
}

const message = (err: unknown) => (err instanceof Error ? err.message : String(err));

function outcome(pos: Chess): string {
  const winner = pos.outcome()?.winner;
  return winner ? `game over — ${winner} wins` : "game over — draw";
}
