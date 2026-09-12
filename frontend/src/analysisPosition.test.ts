import { describe, expect, it } from "vitest";
import { reviewFrames, scoreLabel } from "./analysisPosition";

describe("stored variation presentation", () => {
  it("handles UCI castling, en passant, and underpromotion with SAN labels", () => {
    const castle = reviewFrames("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1", ["e1g1", "e8c8"]);
    expect(castle.slice(1).map((f) => f.san)).toEqual(["O-O", "O-O-O"]);
    expect(reviewFrames("4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1", ["e5d6"])[1].san).toBe("exd6");
    expect(reviewFrames("4k3/P7/8/8/8/8/8/4K3 w - - 0 1", ["a7a8n"])[1].san).toBe("a8=N");
  });
  it("preserves mover-perspective signs and distinguishes mate from missing data", () => {
    expect(scoreLabel(-384, null)).toBe("-3.84");
    expect(scoreLabel(0, null)).toBe("+0.00");
    expect(scoreLabel(null, 0)).toBe("Checkmate delivered");
    expect(scoreLabel(null, null)).toBe("Unavailable");
  });
});
