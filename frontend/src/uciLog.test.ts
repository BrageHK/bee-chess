import { describe, expect, it } from "vitest";
import type { UciLogLine } from "./engine";
import { MAX_LINES, appendLogLine } from "./uciLog";

function line(text: string): UciLogLine {
  return { direction: "sent", text, timestamp: 0 };
}

describe("appendLogLine", () => {
  it("appends to an empty list", () => {
    expect(appendLogLine([], line("uci"))).toEqual([line("uci")]);
  });

  it("appends in order below the bound", () => {
    const result = [line("a"), line("b")].reduce(
      (acc, l) => appendLogLine(acc, l),
      [] as UciLogLine[],
    );
    expect(result.map((l) => l.text)).toEqual(["a", "b"]);
  });

  it("keeps history bounded at MAX_LINES, dropping the oldest first", () => {
    let lines: UciLogLine[] = [];
    for (let i = 0; i < MAX_LINES + 10; i++) {
      lines = appendLogLine(lines, line(`line ${i}`));
    }

    expect(lines).toHaveLength(MAX_LINES);
    // The oldest 10 (0-9) should have been dropped; the newest
    // (MAX_LINES + 9) should still be present.
    expect(lines[0].text).toBe("line 10");
    expect(lines[lines.length - 1].text).toBe(`line ${MAX_LINES + 9}`);
  });

  it("does not mutate the input array", () => {
    const prev = [line("a")];
    const result = appendLogLine(prev, line("b"));
    expect(prev).toEqual([line("a")]);
    expect(result).not.toBe(prev);
  });
});
