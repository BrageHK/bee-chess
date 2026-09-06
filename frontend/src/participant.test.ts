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

  it("bee defaults to 100ms, debug off, and no options yet", () => {
    // `options` starts empty -- it's seeded from GET /api/engines/bee/
    // options once that resolves (see GameSetup.tsx), not hardcoded
    // here. See participant.ts's own docs on why this module no longer
    // knows Bee's option names.
    expect(defaultParticipant("bee")).toMatchObject({
      kind: "bee",
      moveTimeMs: 100,
      debug: false,
      options: {},
    });
  });

  it("bee-mamba defaults to 500ms", () => {
    expect(defaultParticipant("bee-mamba")).toMatchObject({ kind: "bee-mamba", moveTimeMs: 500 });
  });
});

describe("validateParticipant", () => {
  it("rejects zero or negative move time for every bot kind", () => {
    const zero: Participant = { kind: "bee", moveTimeMs: 0, debug: false, options: {} };
    const negative: Participant = { kind: "bee-mamba", moveTimeMs: -1 };
    expect(validateParticipant(zero)).not.toBeNull();
    expect(validateParticipant(negative)).not.toBeNull();
  });

  it("rejects a non-integer move time", () => {
    const p: Participant = { kind: "bee", moveTimeMs: 50.5, debug: false, options: {} };
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

  it("does not validate the contents of a bee participant's options map", () => {
    // Deliberately not this module's job anymore -- see
    // validateParticipant's docs. An unrecognized/invalid option value
    // is Lab/the engine's problem to reject, not the setup screen's.
    const p: Participant = {
      kind: "bee",
      moveTimeMs: 100,
      debug: false,
      options: { Evaluator: "NotARealEvaluator" },
    };
    expect(validateParticipant(p)).toBeNull();
  });
});
