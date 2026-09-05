import { describe, expect, it } from "vitest";
import { pushPly, startNav, type Nav, type Ply } from "./gameHistory";

function ply(fen: string, san?: string): Ply {
  return { fen, san };
}

describe("startNav", () => {
  it("starts at the given position with viewIndex 0", () => {
    expect(startNav("start-fen")).toEqual({ history: [{ fen: "start-fen" }], viewIndex: 0 });
  });
});

describe("pushPly", () => {
  it("appends a ply and advances viewIndex when following the latest one", () => {
    const nav: Nav = { history: [ply("a")], viewIndex: 0 };
    const result = pushPly(nav, ply("b", "e4"));
    expect(result).toEqual({ history: [ply("a"), ply("b", "e4")], viewIndex: 1 });
  });

  it("keeps viewIndex pinned when the user had stepped back to browse history", () => {
    const nav: Nav = { history: [ply("a"), ply("b", "e4"), ply("c", "e5")], viewIndex: 0 };
    const result = pushPly(nav, ply("d", "Nf3"));
    expect(result.history).toEqual([ply("a"), ply("b", "e4"), ply("c", "e5"), ply("d", "Nf3")]);
    expect(result.viewIndex).toBe(0);
  });

  it("does not mutate the input nav", () => {
    const nav: Nav = { history: [ply("a")], viewIndex: 0 };
    pushPly(nav, ply("b", "e4"));
    expect(nav).toEqual({ history: [ply("a")], viewIndex: 0 });
  });
});
