import { describe, expect, it } from "vitest";
import {
  DEFAULT_GAME_CONFIG,
  MAX_STOCKFISH_ELO,
  MIN_STOCKFISH_ELO,
  validateGameConfig,
  type GameConfig,
} from "./gameConfig";

describe("validateGameConfig", () => {
  it("accepts the defaults (1600 Elo, 100ms)", () => {
    expect(DEFAULT_GAME_CONFIG).toEqual({ stockfishElo: 1600, moveTimeMs: 100 });
    expect(validateGameConfig(DEFAULT_GAME_CONFIG)).toBeNull();
  });

  it("rejects zero move time", () => {
    const config: GameConfig = { ...DEFAULT_GAME_CONFIG, moveTimeMs: 0 };
    expect(validateGameConfig(config)).not.toBeNull();
  });

  it("rejects negative move time", () => {
    const config: GameConfig = { ...DEFAULT_GAME_CONFIG, moveTimeMs: -50 };
    expect(validateGameConfig(config)).not.toBeNull();
  });

  it("accepts a move time of 1ms (the minimum)", () => {
    const config: GameConfig = { ...DEFAULT_GAME_CONFIG, moveTimeMs: 1 };
    expect(validateGameConfig(config)).toBeNull();
  });

  it("rejects a non-integer move time", () => {
    const config: GameConfig = { ...DEFAULT_GAME_CONFIG, moveTimeMs: 100.5 };
    expect(validateGameConfig(config)).not.toBeNull();
  });

  it("rejects Elo below Stockfish's supported minimum", () => {
    const config: GameConfig = {
      ...DEFAULT_GAME_CONFIG,
      stockfishElo: MIN_STOCKFISH_ELO - 1,
    };
    expect(validateGameConfig(config)).not.toBeNull();
  });

  it("rejects Elo above Stockfish's supported maximum", () => {
    const config: GameConfig = {
      ...DEFAULT_GAME_CONFIG,
      stockfishElo: MAX_STOCKFISH_ELO + 1,
    };
    expect(validateGameConfig(config)).not.toBeNull();
  });

  it("accepts Elo at the boundaries", () => {
    expect(
      validateGameConfig({ ...DEFAULT_GAME_CONFIG, stockfishElo: MIN_STOCKFISH_ELO }),
    ).toBeNull();
    expect(
      validateGameConfig({ ...DEFAULT_GAME_CONFIG, stockfishElo: MAX_STOCKFISH_ELO }),
    ).toBeNull();
  });

  it("rejects a non-integer Elo", () => {
    const config: GameConfig = { ...DEFAULT_GAME_CONFIG, stockfishElo: 1600.5 };
    expect(validateGameConfig(config)).not.toBeNull();
  });

  it("rejects NaN in either field", () => {
    expect(
      validateGameConfig({ ...DEFAULT_GAME_CONFIG, stockfishElo: Number.NaN }),
    ).not.toBeNull();
    expect(
      validateGameConfig({ ...DEFAULT_GAME_CONFIG, moveTimeMs: Number.NaN }),
    ).not.toBeNull();
  });
});
