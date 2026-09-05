import { describe, expect, it } from "vitest";
import { DEFAULT_PLAY_CONFIG, validatePlayConfig, type PlayConfig } from "./playConfig";

describe("validatePlayConfig", () => {
  it("accepts the defaults (white, 500ms)", () => {
    expect(DEFAULT_PLAY_CONFIG).toEqual({ humanColor: "white", moveTimeMs: 500 });
    expect(validatePlayConfig(DEFAULT_PLAY_CONFIG)).toBeNull();
  });

  it("accepts playing black", () => {
    const config: PlayConfig = { ...DEFAULT_PLAY_CONFIG, humanColor: "black" };
    expect(validatePlayConfig(config)).toBeNull();
  });

  it("rejects zero move time", () => {
    const config: PlayConfig = { ...DEFAULT_PLAY_CONFIG, moveTimeMs: 0 };
    expect(validatePlayConfig(config)).not.toBeNull();
  });

  it("rejects negative move time", () => {
    const config: PlayConfig = { ...DEFAULT_PLAY_CONFIG, moveTimeMs: -50 };
    expect(validatePlayConfig(config)).not.toBeNull();
  });

  it("accepts a move time of 1ms (the minimum)", () => {
    const config: PlayConfig = { ...DEFAULT_PLAY_CONFIG, moveTimeMs: 1 };
    expect(validatePlayConfig(config)).toBeNull();
  });

  it("rejects a non-integer move time", () => {
    const config: PlayConfig = { ...DEFAULT_PLAY_CONFIG, moveTimeMs: 100.5 };
    expect(validatePlayConfig(config)).not.toBeNull();
  });

  it("rejects NaN move time", () => {
    expect(
      validatePlayConfig({ ...DEFAULT_PLAY_CONFIG, moveTimeMs: Number.NaN }),
    ).not.toBeNull();
  });
});
