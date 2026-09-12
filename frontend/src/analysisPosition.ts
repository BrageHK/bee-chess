import { Chess, normalizeMove } from "chessops/chess";
import { makeFen, parseFen } from "chessops/fen";
import { makeSanAndPlay } from "chessops/san";
import { parseUci } from "chessops/util";

export interface ReviewFrame { fen: string; uci: string; san: string; label: string }

/** Presentation of stored variations only. This never changes a Lab game. */
export function reviewFrames(fen: string, line: string[]): ReviewFrame[] {
  const setup = parseFen(fen);
  if (setup.isErr) throw new Error("The stored position has an invalid FEN.");
  const position = Chess.fromSetup(setup.value);
  if (position.isErr) throw new Error("The stored position cannot be reviewed.");
  const frames: ReviewFrame[] = [{ fen, uci: "", san: "", label: "Before move" }];
  for (const uci of line) {
    const parsed = parseUci(uci);
    if (!parsed) throw new Error(`The stored variation contains an invalid move: ${uci}`);
    const move = normalizeMove(position.value, parsed);
    if (!position.value.isLegal(move)) throw new Error(`The stored variation contains an illegal move: ${uci}`);
    const number = `${position.value.fullmoves}${position.value.turn === "white" ? "." : "..."}`;
    const san = makeSanAndPlay(position.value, move);
    frames.push({ fen: makeFen(position.value.toSetup()), uci, san, label: `${number} ${san}` });
  }
  return frames;
}

export function moveSan(fen: string, uci: string | null): string {
  if (!uci) return "—";
  try { return reviewFrames(fen, [uci])[1].san; }
  catch { return uci; }
}

export function scoreLabel(cp: number | null, mate: number | null): string {
  if (mate === 0) return "Checkmate delivered";
  if (mate !== null) return `${mate > 0 ? "Winning" : "Losing"} mate (${Math.abs(mate)} ${Math.abs(mate) === 1 ? "ply" : "plies"})`;
  return cp === null ? "Unavailable" : `${cp >= 0 ? "+" : ""}${(cp / 100).toFixed(2)}`;
}
