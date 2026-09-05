import { describe, expect, it } from "vitest";
import {
  evalBarFraction,
  formatCount,
  formatNps,
  formatScore,
  parseUciInfo,
  toWhitePerspective,
} from "./uciInfo";

describe("parseUciInfo", () => {
  it("parses the issue's example line", () => {
    const info = parseUciInfo(
      "info depth 12 score cp 31 nodes 42122 nps 1053050 time 40 pv e2e4 e7e5",
    );
    expect(info).toEqual({
      depth: 12,
      scoreCp: 31,
      nodes: 42122,
      nps: 1053050,
      timeMs: 40,
      pv: ["e2e4", "e7e5"],
    });
  });

  it("parses seldepth alongside depth", () => {
    const info = parseUciInfo("info depth 8 seldepth 11 score cp 22 nodes 4532 nps 755333 time 6");
    expect(info?.depth).toBe(8);
    expect(info?.seldepth).toBe(11);
  });

  it("parses score mate, positive and negative", () => {
    expect(parseUciInfo("info depth 5 score mate 4")?.scoreMate).toBe(4);
    expect(parseUciInfo("info depth 5 score mate -2")?.scoreMate).toBe(-2);
  });

  it("does not set scoreCp when scoreMate is present or vice versa", () => {
    const mateInfo = parseUciInfo("info depth 5 score mate 4");
    expect(mateInfo?.scoreCp).toBeUndefined();

    const cpInfo = parseUciInfo("info depth 5 score cp 31");
    expect(cpInfo?.scoreMate).toBeUndefined();
  });

  it("returns null for non-info lines", () => {
    expect(parseUciInfo("bestmove e2e4")).toBeNull();
    expect(parseUciInfo("id name Bee")).toBeNull();
    expect(parseUciInfo("uciok")).toBeNull();
    expect(parseUciInfo("readyok")).toBeNull();
  });

  it("returns null for info string diagnostics", () => {
    // info string is the debug-diagnostics channel (#42/#44), a
    // different concern from search telemetry -- it carries no
    // depth/score/nodes/etc. fields this parser understands.
    expect(parseUciInfo("info string ignored unknown UCI command: foo")).toBeNull();
  });

  it("returns null for an info line with no recognized fields", () => {
    expect(parseUciInfo("info currmove e2e4 currmovenumber 1")).toBeNull();
  });

  it("ignores unrecognized fields without corrupting the rest of the line", () => {
    const info = parseUciInfo(
      "info depth 10 multipv 1 score cp 15 nodes 1000 hashfull 234 nps 500000 time 20 pv d2d4",
    );
    expect(info).toEqual({
      depth: 10,
      scoreCp: 15,
      nodes: 1000,
      nps: 500000,
      timeMs: 20,
      pv: ["d2d4"],
    });
  });

  it("parses a pv with no moves as an empty pv, not a crash", () => {
    const info = parseUciInfo("info depth 1 pv");
    expect(info?.pv).toEqual([]);
  });

  it("parses a pv field even with nothing else recognized before it", () => {
    const info = parseUciInfo("info pv e2e4 e7e5 g1f3");
    expect(info?.pv).toEqual(["e2e4", "e7e5", "g1f3"]);
  });
});

describe("formatScore", () => {
  it("formats a positive centipawn score with a leading +", () => {
    expect(formatScore({ scoreCp: 31 })).toBe("+0.31");
  });

  it("formats a negative centipawn score with a leading -", () => {
    expect(formatScore({ scoreCp: -120 })).toBe("-1.20");
  });

  it("formats zero as +0.00", () => {
    expect(formatScore({ scoreCp: 0 })).toBe("+0.00");
  });

  it("formats a positive mate score as M<n>", () => {
    expect(formatScore({ scoreMate: 4 })).toBe("M4");
  });

  it("formats a negative mate score as -M<n>", () => {
    expect(formatScore({ scoreMate: -2 })).toBe("-M2");
  });

  it("prefers mate over cp when both are somehow present", () => {
    expect(formatScore({ scoreCp: 31, scoreMate: 4 })).toBe("M4");
  });

  it("returns undefined when neither is present", () => {
    expect(formatScore({})).toBeUndefined();
  });
});

describe("toWhitePerspective", () => {
  it("passes a white engine's score through unchanged", () => {
    expect(toWhitePerspective({ scoreCp: 31 }, "white")).toEqual({
      scoreCp: 31,
      scoreMate: undefined,
    });
  });

  it("negates a black engine's score", () => {
    expect(toWhitePerspective({ scoreCp: 31 }, "black")).toEqual({
      scoreCp: -31,
      scoreMate: undefined,
    });
  });

  it("negates a black engine's mate score", () => {
    expect(toWhitePerspective({ scoreMate: 4 }, "black")).toEqual({
      scoreCp: undefined,
      scoreMate: -4,
    });
  });

  it("leaves undefined fields undefined regardless of color", () => {
    expect(toWhitePerspective({}, "black")).toEqual({ scoreCp: undefined, scoreMate: undefined });
  });
});

describe("evalBarFraction", () => {
  it("is neutral (0.5) when there is no score yet", () => {
    expect(evalBarFraction({}, 3)).toBe(0.5);
  });

  it("is 0.5 at an exactly even score", () => {
    expect(evalBarFraction({ scoreCp: 0 }, 3)).toBe(0.5);
  });

  it("grows toward 1 as white's advantage grows, up to the clamp", () => {
    expect(evalBarFraction({ scoreCp: 150 }, 3)).toBeCloseTo(0.75);
    expect(evalBarFraction({ scoreCp: 300 }, 3)).toBe(1);
  });

  it("clamps beyond the given range instead of overflowing", () => {
    expect(evalBarFraction({ scoreCp: 800 }, 3)).toBe(1);
    expect(evalBarFraction({ scoreCp: -800 }, 3)).toBe(0);
  });

  it("shrinks toward 0 as black's advantage grows", () => {
    expect(evalBarFraction({ scoreCp: -150 }, 3)).toBeCloseTo(0.25);
  });

  it("pins fully to 1 for a positive (white) mate score regardless of the clamp", () => {
    expect(evalBarFraction({ scoreMate: 12 }, 3)).toBe(1);
  });

  it("pins fully to 0 for a negative (black) mate score regardless of the clamp", () => {
    expect(evalBarFraction({ scoreMate: -1 }, 3)).toBe(0);
  });
});

describe("formatCount", () => {
  it("adds thousands separators", () => {
    expect(formatCount(42122)).toBe("42,122");
    expect(formatCount(1000000)).toBe("1,000,000");
    expect(formatCount(500)).toBe("500");
    expect(formatCount(0)).toBe("0");
  });
});

describe("formatNps", () => {
  it("formats millions with an M suffix", () => {
    expect(formatNps(1053050)).toBe("1.05M");
  });

  it("formats thousands with a k suffix", () => {
    expect(formatNps(755000)).toBe("755.0k");
  });

  it("formats small values as-is", () => {
    expect(formatNps(500)).toBe("500");
  });
});
