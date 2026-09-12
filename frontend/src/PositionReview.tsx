import { useEffect, useMemo, useRef, useState } from "react";
import type { Key } from "@lichess-org/chessground/types";
import { Chessground } from "./Chessground";
import type { ReviewMove } from "./analysisClient";
import { moveSan, reviewFrames, scoreLabel } from "./analysisPosition";
import { Button, Panel, PanelBody, PanelHeader } from "./components/ui";

export function PositionReview({ move, onClose }: { move: ReviewMove; onClose: () => void }) {
  const reviewRef = useRef<HTMLDivElement>(null);
  useEffect(() => { reviewRef.current?.scrollIntoView?.({ behavior: "smooth", block: "start" }); }, []);
  const [step, setStep] = useState(0);
  const [line, setLine] = useState<"pv" | "played">("pv");
  const [flipped, setFlipped] = useState(false);
  const variation = useMemo(() => {
    try {
      return { frames: reviewFrames(move.fen_before, line === "played" ? [move.played_move] : (move.pv?.split(/\s+/).filter(Boolean) ?? [])), error: null };
    } catch (error) {
      return { frames: [{ fen: move.fen_before, uci: "", san: "", label: "Before move" }], error: String(error) };
    }
  }, [move, line]);
  const frame = variation.frames[Math.min(step, variation.frames.length - 1)];
  const beeColor = move.mover_color ?? "white";
  const orientation = flipped ? (beeColor === "white" ? "black" : "white") : beeColor;
  const showLine = (line: "pv" | "played") => { setLine(line); setStep(0); };
  return (
    <div ref={reviewRef}><Panel className="w-full text-left select-text" aria-label="Position review">
      <PanelHeader className="flex flex-wrap items-center justify-between gap-2">
        <h2>Position review · {move.game_id} · ply {move.ply}</h2>
        <Button onClick={onClose}>Close review</Button>
      </PanelHeader>
      <PanelBody className="grid min-w-0 grid-cols-1 gap-4">
        <p className="m-0 text-sm text-muted">{move.white ?? "White"} vs {move.black ?? "Black"} · {move.result ?? "Unknown result"} · {move.phase}</p>
        <div className="flex flex-wrap gap-6">
          <div className="w-[480px] max-w-full shrink-0 overflow-hidden">
            <Chessground responsive config={{
              fen: frame.fen, orientation, viewOnly: true, coordinates: true,
              lastMove: frame.uci ? [frame.uci.slice(0, 2), frame.uci.slice(2, 4)] as Key[] : undefined,
              drawable: { enabled: false },
            }} />
          </div>
          <div className="flex min-w-56 flex-1 flex-col gap-4">
            <dl className="grid grid-cols-2 gap-2 text-sm">
              <dt className="text-muted">Bee move</dt><dd className="m-0 font-medium">{moveSan(move.fen_before, move.played_move)}</dd>
              <dt className="text-muted">Stockfish best</dt><dd className="m-0 font-medium">{moveSan(move.fen_before, move.best_move)}</dd>
              <dt className="text-muted">Before</dt><dd className="m-0">{scoreLabel(move.eval_before_cp, move.mate_before)}</dd>
              <dt className="text-muted">After</dt><dd className="m-0">{scoreLabel(move.eval_after_cp, move.mate_after)}</dd>
              <dt className="text-muted">Loss</dt><dd className="m-0 font-medium text-danger">{move.centipawn_loss === null ? "Mate transition" : `${move.centipawn_loss}cp`}</dd>
            </dl>
            <p className="m-0 text-xs text-muted">Evaluations are from Bee’s perspective. Ply numbers start at 0.</p>
            {move.centipawn_loss !== null && move.centipawn_loss > 0 && move.best_move === move.played_move && <p className="m-0 text-sm text-warning">Stockfish recommended the played move. This score drop is a disagreement between the before/after searches, so it needs deeper review before being counted as a move error.</p>}
            <div className="flex flex-wrap gap-2">
              <Button aria-pressed={line === "pv"} onClick={() => showLine("pv")}>Stockfish PV</Button>
              <Button aria-pressed={line === "played"} onClick={() => { showLine("played"); setStep(1); }}>Bee’s played move</Button>
              <Button onClick={() => setFlipped((v) => !v)}>Flip board</Button>
            </div>
            {variation.error && <p role="alert" className="m-0 text-sm text-danger">{variation.error}</p>}
            {!move.pv && line === "pv" && <p className="m-0 text-muted">No PV stored for this move.</p>}
            <div className="flex flex-wrap gap-1" aria-label="Variation moves">
              {variation.frames.map((entry, index) => (
                <Button key={index} aria-pressed={step === index} variant={step === index ? "primary" : "secondary"} onClick={() => setStep(index)}>
                  {entry.label}
                </Button>
              ))}
            </div>
            <div className="flex flex-wrap gap-2">
              <Button disabled={step === 0} onClick={() => setStep((n) => Math.max(0, n - 1))}>Previous position</Button>
              <Button disabled={step >= variation.frames.length - 1} onClick={() => setStep((n) => n + 1)}>Next position</Button>
            </div>
            <p className="m-0 text-sm" aria-live="polite">Viewing: {frame.label}</p>
            {move.original_game_url && <a className="text-accent underline" href={move.original_game_url} target="_blank" rel="noreferrer">Open original game ↗</a>}
          </div>
        </div>
        <label className="grid gap-1 text-sm text-muted">
          FEN before Bee’s move
          <input readOnly aria-label="FEN before Bee’s move" value={move.fen_before} className="w-full rounded-md border border-border bg-surface-subtle p-2 font-mono text-xs text-text" />
        </label>
      </PanelBody>
    </Panel></div>
  );
}
