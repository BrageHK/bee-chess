import { beforeEach, describe, expect, it } from "vitest";
import { clearSavedGame, loadSavedGame, saveGame, type SavedGame } from "./savedGame";

const game: SavedGame = {
  white: { kind: "human" },
  black: { kind: "stockfish", elo: 1600, moveTimeMs: 100, debug: false },
  moves: ["e2e4", "e7e5"],
};

beforeEach(() => {
  sessionStorage.clear();
});

describe("loadSavedGame", () => {
  it("returns null when nothing is saved", () => {
    expect(loadSavedGame()).toBeNull();
  });

  it("round-trips a saved game", () => {
    saveGame(game);
    expect(loadSavedGame()).toEqual(game);
  });

  it("returns null for corrupt JSON instead of throwing", () => {
    sessionStorage.setItem("bee-chess:game", "{not json");
    expect(loadSavedGame()).toBeNull();
  });

  it("returns null for a value that doesn't look like a saved game", () => {
    sessionStorage.setItem("bee-chess:game", JSON.stringify({ foo: "bar" }));
    expect(loadSavedGame()).toBeNull();
  });
});

describe("clearSavedGame", () => {
  it("removes a saved game", () => {
    saveGame(game);
    clearSavedGame();
    expect(loadSavedGame()).toBeNull();
  });
});
