import { useEffect, useRef, useState } from "react";
import { Chess } from "chessops/chess";
import { makeFen } from "chessops/fen";
import { parseUci } from "chessops/util";
import type { Key } from "@lichess-org/chessground/types";
import { Chessground } from "./Chessground";
import { whiteEngine, blackEngine } from "./engine";
import "@lichess-org/chessground/assets/chessground.base.css";
import "@lichess-org/chessground/assets/chessground.brown.css";
import "@lichess-org/chessground/assets/chessground.cburnett.css";

const START = 3 * 60 * 1000; // 3 min
const INC = 2000; // +2s

/** Claim a draw once the fifty-move counter is full; chessops does not. */
const MAX_HALFMOVES = 100;

/** Board position before any move; avoids reading the ref during render. */
const START_FEN = makeFen(Chess.default().toSetup());

const fmt = (t: number) =>
  `${Math.floor(t / 60000)}:${String(Math.floor((t % 60000) / 1000)).padStart(2, "0")}`;

export default function App() {
  const posRef = useRef(Chess.default());
  const [fen, setFen] = useState(START_FEN);
  const [lastMove, setLastMove] = useState<Key[] | undefined>();
  const [clocks, setClocks] = useState({ w: START, b: START });
  const [status, setStatus] = useState("connecting to the bridge…");

  useEffect(() => {
    let alive = true;

    void (async () => {
      const clk = { w: START, b: START };
      const moves: string[] = [];

      try {
        await Promise.all([whiteEngine.init(), blackEngine.init()]);
      } catch (err) {
        if (alive) setStatus(message(err));
        return;
      }

      while (alive && !posRef.current.isEnd()) {
        if (posRef.current.halfmoves >= MAX_HALFMOVES) {
          setStatus("draw by the fifty-move rule");
          return;
        }

        const white = posRef.current.turn === "white";
        const engine = white ? whiteEngine : blackEngine;
        setStatus(`${engine.name} thinking…`);

        const t0 = performance.now();
        let uci: string;
        try {
          uci = await engine.bestMove(
            moves,
            { wtime: clk.w, btime: clk.b, winc: INC, binc: INC },
            white,
          );
        } catch (err) {
          if (alive) setStatus(message(err));
          return;
        }
        const spent = performance.now() - t0;
        if (!alive) return;

        if (white) clk.w = clk.w - spent + INC;
        else clk.b = clk.b - spent + INC;
        if (clk.w <= 0 || clk.b <= 0) {
          setClocks({ ...clk });
          setStatus(
            `${clk.w <= 0 ? whiteEngine.name : blackEngine.name} flags — out of time`,
          );
          return;
        }

        const move = parseUci(uci);
        if (!move || !posRef.current.isLegal(move)) {
          setStatus(`${engine.name} played an illegal move: ${uci}`);
          return;
        }

        posRef.current.play(move);
        moves.push(uci);
        setFen(makeFen(posRef.current.toSetup()));
        setLastMove([uci.slice(0, 2), uci.slice(2, 4)] as Key[]);
        setClocks({ ...clk });
      }

      if (alive) setStatus(outcome(posRef.current));
    })();

    return () => {
      alive = false;
    };
  }, []);

  return (
    <main style={{ display: "grid", placeItems: "center", gap: 8, padding: 24 }}>
      <h1>
        {whiteEngine.name} (white) vs {blackEngine.name} (black)
      </h1>
      <p>
        white {fmt(clocks.w)} — black {fmt(clocks.b)}
      </p>
      <Chessground config={{ fen, lastMove, viewOnly: true }} />
      <p>{status}</p>
    </main>
  );
}

const message = (err: unknown) => (err instanceof Error ? err.message : String(err));

function outcome(pos: Chess): string {
  const winner = pos.outcome()?.winner;
  return winner ? `game over — ${winner} wins` : "game over — draw";
}
