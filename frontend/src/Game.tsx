import { useEffect, useRef, useState } from "react";
import { Chess } from "chessops/chess";
import { makeFen } from "chessops/fen";
import { chessgroundDests } from "chessops/compat";
import { parseUci } from "chessops/util";
import type { Key, Dests } from "@lichess-org/chessground/types";
import { Chessground } from "./Chessground";
import { createBotClient, type UciClient } from "./engine";
import { UciLogPanel } from "./UciLogPanel";
import { SearchStatsPanel } from "./SearchStatsPanel";
import type { Participant } from "./participant";

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

  const [fen, setFen] = useState(START_FEN);
  const [lastMove, setLastMove] = useState<Key[] | undefined>();
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
    setFen(makeFen(posRef.current.toSetup()));
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
    posRef.current.play(move);
    movesRef.current.push(uci);
    setLastMove([uci.slice(0, 2), uci.slice(2, 4)] as Key[]);
    syncBoardState();
    return true;
  };

  /** Drives bot turns for as long as it's a bot's move (may be more
   * than one ply in a row if both slots are bots, or zero if it's
   * immediately a human's turn). */
  const runBotTurns = async () => {
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
        setStatus(message(err));
        setFinished(true);
        return;
      }

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
  // here).
  useEffect(() => {
    void (async () => {
      setStatus("connecting to the bridge…");

      const pending: Promise<void>[] = [];
      for (const color of ["white", "black"] as const) {
        const participant = participantFor(color);
        if (participant.kind === "human") continue;
        const client = createBotClient(participant.kind);
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
        setStatus(message(err));
        setFinished(true);
        return;
      }

      syncBoardState();
      await runBotTurns();
    })();
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
    if (!humanTurnColor()) return;

    const plain = `${orig}${dest}`;
    const plainMove = parseUci(plain);
    const uci = plainMove && posRef.current.isLegal(plainMove) ? plain : `${plain}q`;

    if (!applyMove(uci)) return;
    if (finishIfOver()) return;
    void runBotTurns();
  };

  const canMoveColor = humanTurnColor();

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
      <Chessground
        config={{
          fen,
          lastMove,
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
      <p>{status}</p>
      {finished && <button onClick={onBackToSetup}>New game</button>}
      <BotPanels color="white" participant={white} client={clientFor("white")} />
      <BotPanels color="black" participant={black} client={clientFor("black")} />
    </section>
  );
}

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
