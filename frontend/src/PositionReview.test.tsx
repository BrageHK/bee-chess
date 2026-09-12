import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import type { Config } from "@lichess-org/chessground/config";
import { PositionReview } from "./PositionReview";
import { reviewMoveFixture as move } from "./analysisFixtures";

vi.mock("./Chessground", () => ({ Chessground: ({ config }: { config: Config }) =>
  <div data-testid="board" data-fen={config.fen} data-orientation={config.orientation} /> }));

describe("Position review", () => {
  it("flags a score drop when Stockfish recommended the played move", () => {
    render(<PositionReview move={{ ...move, best_move: move.played_move }} onClose={() => {}} />);
    expect(screen.getByText(/Stockfish recommended the played move/)).toBeInTheDocument();
  });
  it("steps through the PV, resets to before, and can inspect Bee's played move", async () => {
    const user = userEvent.setup();
    render(<PositionReview move={move} onClose={() => {}} />);
    expect(screen.getByTestId("board")).toHaveAttribute("data-fen", move.fen_before);
    expect(screen.getByText("+0.28")).toBeInTheDocument();
    expect(screen.getByText("-3.84")).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "Next position" }));
    expect(screen.getByTestId("board").getAttribute("data-fen")).toContain("4P3");
    expect(screen.getByText("Viewing: 1. e4")).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "1... e5" }));
    expect(screen.getByText("Viewing: 1... e5")).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "Before move" }));
    expect(screen.getByTestId("board")).toHaveAttribute("data-fen", move.fen_before);
    await user.click(screen.getByRole("button", { name: "Bee’s played move" }));
    expect(screen.getByTestId("board").getAttribute("data-fen")).toContain("6P1");
    expect(screen.getByText("Viewing: 1. g4")).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "Flip board" }));
    expect(screen.getByTestId("board")).toHaveAttribute("data-orientation", "black");
  });

  it("shows malformed variations and absent PVs without crashing the reviewer", () => {
    const { rerender } = render(<PositionReview move={{ ...move, pv: "e2e5" }} onClose={() => {}} />);
    expect(screen.getByRole("alert")).toHaveTextContent("illegal move");
    rerender(<PositionReview move={{ ...move, pv: null, centipawn_loss: null, eval_after_cp: null, mate_after: -1 }} onClose={() => {}} />);
    expect(screen.getByText("No PV stored for this move.")).toBeInTheDocument();
    expect(screen.getByText("Losing mate (1 ply)")).toBeInTheDocument();
    expect(screen.getByText("Mate transition")).toBeInTheDocument();
  });
});
