import { describe, expect, it } from "vitest";
import {
  MAX_STOCKFISH_ELO,
  MIN_STOCKFISH_ELO,
  defaultParticipant,
  validateParticipant,
  type Participant,
} from "./participant";

describe("defaultParticipant", () => {
  it("human has no config to validate", () => {
    expect(validateParticipant(defaultParticipant("human"))).toBeNull();
  });

  it("every default is itself valid", () => {
    for (const kind of ["human", "stockfish", "bee", "bee-mamba"] as const) {
      expect(validateParticipant(defaultParticipant(kind))).toBeNull();
    }
  });

  it("stockfish defaults to 1600 elo, 100ms", () => {
    const p = defaultParticipant("stockfish");
    expect(p).toMatchObject({ kind: "stockfish", elo: 1600, moveTimeMs: 100, debug: false });
  });

  it("bee defaults to 100ms, debug off", () => {
    expect(defaultParticipant("bee")).toMatchObject({
      kind: "bee",
      moveTimeMs: 100,
      debug: false,
      settings: { Evaluator: "Positional" },
    });
  });

  it("bee-mamba defaults to 500ms", () => {
    expect(defaultParticipant("bee-mamba")).toMatchObject({ kind: "bee-mamba", moveTimeMs: 500 });
  });
});

describe("validateParticipant", () => {
  it("rejects zero or negative move time for every bot kind", () => {
    const zero: Participant = {
      kind: "bee",
      moveTimeMs: 0,
      debug: false,
      settings: { Evaluator: "Positional" },
    };
    const negative: Participant = { kind: "bee-mamba", moveTimeMs: -1 };
    expect(validateParticipant(zero)).not.toBeNull();
    expect(validateParticipant(negative)).not.toBeNull();
  });

  it("rejects a non-integer move time", () => {
    const p: Participant = {
      kind: "bee",
      moveTimeMs: 50.5,
      debug: false,
      settings: { Evaluator: "Positional" },
    };
    expect(validateParticipant(p)).not.toBeNull();
  });

  it("rejects stockfish elo outside the supported range", () => {
    const tooLow: Participant = {
      kind: "stockfish",
      elo: MIN_STOCKFISH_ELO - 1,
      moveTimeMs: 100,
      debug: false,
    };
    const tooHigh: Participant = {
      kind: "stockfish",
      elo: MAX_STOCKFISH_ELO + 1,
      moveTimeMs: 100,
      debug: false,
    };
    expect(validateParticipant(tooLow)).not.toBeNull();
    expect(validateParticipant(tooHigh)).not.toBeNull();
  });

  it("accepts stockfish elo at the boundaries", () => {
    const low: Participant = {
      kind: "stockfish",
      elo: MIN_STOCKFISH_ELO,
      moveTimeMs: 100,
      debug: false,
    };
    const high: Participant = {
      kind: "stockfish",
      elo: MAX_STOCKFISH_ELO,
      moveTimeMs: 100,
      debug: false,
    };
    expect(validateParticipant(low)).toBeNull();
    expect(validateParticipant(high)).toBeNull();
  });

  it("bee-mamba has no elo/debug field to validate beyond move time", () => {
    const p: Participant = { kind: "bee-mamba", moveTimeMs: 250 };
    expect(validateParticipant(p)).toBeNull();
  });

  it("rejects an unknown Bee evaluator", () => {
    const p: Participant = {
      kind: "bee",
      moveTimeMs: 100,
      debug: false,
      settings: { Evaluator: "Unknown" },
    };
    expect(validateParticipant(p)).toMatch(/valid Bee evaluator/);
  });
});
