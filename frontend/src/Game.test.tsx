import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { Game } from "./Game";
import * as labClient from "./labClient";
import type { GameSnapshot } from "./labClient";

vi.mock("./labClient", async () => {
  const actual = await vi.importActual<typeof labClient>("./labClient");
  return {
    ...actual,
    subscribeToGameEvents: vi.fn().mockReturnValue(() => {}),
  };
});

/** A running game only -- these tests never need `finished`/`aborted`
 * (see `GameSnapshot`'s status-discriminated union in labClient.ts),
 * so `overrides` is restricted to the fields the "running" variant
 * actually has. */
function snapshot(overrides: Partial<Omit<GameSnapshot, "status">>): GameSnapshot & { status: "running" } {
  return {
    id: "game-1",
    fen: "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    moves: [],
    uci_log: [],
    white: { kind: "human" },
    black: { kind: "human" },
    experiment_id: null,
    time_control: { type: "move_time", move_time_ms: 100 },
    white_clock_ms: null,
    black_clock_ms: null,
    ...overrides,
    status: "running",
  };
}

/** Chessground toggles an `orientation-<color>` class on the board's
 * wrapper element -- the one DOM-visible signal of which way it's
 * currently facing (see @lichess-org/chessground/dist/wrap.js). It's
 * applied via a separate effect from the one that mounts the board
 * (see Chessground.tsx), so it can still be mid-flight even after the
 * component around it has otherwise settled -- always read it through
 * `waitFor`, never right after an unrelated `findBy*`. */
function boardOrientation(): "white" | "black" {
  const el = document.querySelector(".orientation-white, .orientation-black");
  expect(el).not.toBeNull();
  return el!.classList.contains("orientation-white") ? "white" : "black";
}

async function expectOrientation(color: "white" | "black") {
  await waitFor(() => expect(boardOrientation()).toBe(color));
}

const noop = () => {};

beforeEach(() => {
  vi.mocked(labClient.subscribeToGameEvents).mockReturnValue(() => {});
});

afterEach(() => {
  vi.restoreAllMocks();
});

describe("Game board rotation (#119)", () => {
  it("defaults to white-at-bottom when the human plays white", async () => {
    vi.spyOn(labClient, "getGame").mockResolvedValue(
      snapshot({ white: { kind: "human" }, black: { kind: "engine", name: "Bee", debug: false } }),
    );

    render(
      <Game source={{ kind: "resume", gameId: "game-1" }} onGameCreated={noop} onBackToSetup={noop} onOpenExperiment={noop} />,
    );

    await screen.findByRole("button", { name: /rotate board/i });
    await expectOrientation("white");
  });

  it("defaults to black-at-bottom when the human plays black", async () => {
    vi.spyOn(labClient, "getGame").mockResolvedValue(
      snapshot({ white: { kind: "engine", name: "Bee", debug: false }, black: { kind: "human" } }),
    );

    render(
      <Game source={{ kind: "resume", gameId: "game-1" }} onGameCreated={noop} onBackToSetup={noop} onOpenExperiment={noop} />,
    );

    await screen.findByRole("button", { name: /rotate board/i });
    await expectOrientation("black");
  });

  it("the Rotate board button flips the default orientation", async () => {
    vi.spyOn(labClient, "getGame").mockResolvedValue(
      snapshot({ white: { kind: "human" }, black: { kind: "engine", name: "Bee", debug: false } }),
    );
    const user = userEvent.setup();

    render(
      <Game source={{ kind: "resume", gameId: "game-1" }} onGameCreated={noop} onBackToSetup={noop} onOpenExperiment={noop} />,
    );

    await screen.findByRole("button", { name: /rotate board/i });
    await expectOrientation("white");

    await user.click(screen.getByRole("button", { name: /rotate board/i }));
    await expectOrientation("black");

    await user.click(screen.getByRole("button", { name: /rotate board/i }));
    await expectOrientation("white");
  });
});
