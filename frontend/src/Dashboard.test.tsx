import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import { Dashboard } from "./Dashboard";
import * as labClient from "./labClient";
import type { ExperimentSnapshot, GameSnapshot } from "./labClient";

vi.mock("./labClient", async () => {
  const actual = await vi.importActual<typeof labClient>("./labClient");
  return {
    ...actual,
    listGames: vi.fn(),
    listExperiments: vi.fn(),
  };
});

function game(overrides: Partial<GameSnapshot> = {}): GameSnapshot {
  return {
    id: "g1",
    fen: "start",
    moves: [],
    white: { kind: "human" },
    black: { kind: "human" },
    experiment_id: null,
    status: "running",
    ...overrides,
  } as GameSnapshot;
}

function experiment(overrides: Partial<ExperimentSnapshot> = {}): ExperimentSnapshot {
  return {
    id: "e1",
    status: "running",
    label_a: "A",
    label_b: "B",
    requested_games: 5,
    completed_games: 0,
    wins_a: 0,
    draws: 0,
    wins_b: 0,
    score_a: null,
    games: [],
    ...overrides,
  };
}

describe("Dashboard", () => {
  it("shows the empty-state message for both sections when nothing exists", async () => {
    vi.mocked(labClient.listGames).mockResolvedValue([]);
    vi.mocked(labClient.listExperiments).mockResolvedValue([]);

    render(
      <Dashboard onNewGame={() => {}} onNewExperiment={() => {}} onOpenGame={() => {}} onOpenExperiment={() => {}} />,
    );

    expect(await screen.findByText(/nothing running right now/i)).toBeInTheDocument();
    expect(screen.getByText(/no finished games or experiments yet/i)).toBeInTheDocument();
  });

  it("splits games and experiments into Running vs Past by status", async () => {
    vi.mocked(labClient.listGames).mockResolvedValue([
      game({ id: "running-game", status: "running" }),
      game({ id: "finished-game", status: "finished", result: "white_wins" } as Partial<GameSnapshot>),
    ]);
    vi.mocked(labClient.listExperiments).mockResolvedValue([
      experiment({ id: "running-exp", status: "running" }),
      experiment({ id: "done-exp", status: "completed" }),
    ]);

    render(
      <Dashboard onNewGame={() => {}} onNewExperiment={() => {}} onOpenGame={() => {}} onOpenExperiment={() => {}} />,
    );

    await waitFor(() => expect(screen.queryByText(/loading/i)).not.toBeInTheDocument());
    // Both "Running" and "Past" sections should have their own rows,
    // not fall into the empty-state message the Section-children-
    // length bug (two rendered lists per section, not one) used to
    // trigger regardless of contents.
    expect(screen.queryByText(/nothing running right now/i)).not.toBeInTheDocument();
    expect(screen.queryByText(/no finished games or experiments yet/i)).not.toBeInTheDocument();
  });

  it("clicking a game row calls onOpenGame with that game's id", async () => {
    vi.mocked(labClient.listGames).mockResolvedValue([game({ id: "g42" })]);
    vi.mocked(labClient.listExperiments).mockResolvedValue([]);
    const onOpenGame = vi.fn();
    const user = userEvent.setup();

    render(
      <Dashboard onNewGame={() => {}} onNewExperiment={() => {}} onOpenGame={onOpenGame} onOpenExperiment={() => {}} />,
    );

    await user.click(await screen.findByText(/Game — You vs You/i));

    expect(onOpenGame).toHaveBeenCalledWith("g42");
  });

  it("clicking an experiment row calls onOpenExperiment with that experiment's id", async () => {
    vi.mocked(labClient.listGames).mockResolvedValue([]);
    vi.mocked(labClient.listExperiments).mockResolvedValue([experiment({ id: "exp-42" })]);
    const onOpenExperiment = vi.fn();
    const user = userEvent.setup();

    render(
      <Dashboard
        onNewGame={() => {}}
        onNewExperiment={() => {}}
        onOpenGame={() => {}}
        onOpenExperiment={onOpenExperiment}
      />,
    );

    await user.click(await screen.findByText(/Experiment — A vs B/i));

    expect(onOpenExperiment).toHaveBeenCalledWith("exp-42");
  });

  it("New game and New experiment buttons call their handlers", async () => {
    vi.mocked(labClient.listGames).mockResolvedValue([]);
    vi.mocked(labClient.listExperiments).mockResolvedValue([]);
    const onNewGame = vi.fn();
    const onNewExperiment = vi.fn();
    const user = userEvent.setup();

    render(
      <Dashboard
        onNewGame={onNewGame}
        onNewExperiment={onNewExperiment}
        onOpenGame={() => {}}
        onOpenExperiment={() => {}}
      />,
    );

    await user.click(screen.getByRole("button", { name: /new game/i }));
    await user.click(screen.getByRole("button", { name: /new experiment/i }));

    expect(onNewGame).toHaveBeenCalled();
    expect(onNewExperiment).toHaveBeenCalled();
  });

  it("shows an error message if fetching the lists fails", async () => {
    vi.mocked(labClient.listGames).mockRejectedValue(new Error("network down"));
    vi.mocked(labClient.listExperiments).mockResolvedValue([]);

    render(
      <Dashboard onNewGame={() => {}} onNewExperiment={() => {}} onOpenGame={() => {}} onOpenExperiment={() => {}} />,
    );

    expect(await screen.findByText("network down")).toBeInTheDocument();
  });
});
