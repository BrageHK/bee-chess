import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import { ExperimentView } from "./ExperimentView";
import * as labClient from "./labClient";
import type { ExperimentSnapshot } from "./labClient";

vi.mock("./labClient", async () => {
  const actual = await vi.importActual<typeof labClient>("./labClient");
  return {
    ...actual,
    getExperiment: vi.fn(),
  };
});

function runningSnapshot(overrides: Partial<ExperimentSnapshot> = {}): ExperimentSnapshot {
  return {
    id: "exp-1",
    status: "running",
    label_a: "Baseline",
    label_b: "Candidate",
    requested_games: 2,
    completed_games: 0,
    wins_a: 0,
    draws: 0,
    wins_b: 0,
    score_a: null,
    games: [],
    ...overrides,
  };
}

describe("ExperimentView", () => {
  it("renders the current tally and progress once the snapshot loads", async () => {
    vi.mocked(labClient.getExperiment).mockResolvedValue(
      runningSnapshot({
        status: "completed",
        completed_games: 2,
        wins_a: 1,
        draws: 1,
        score_a: 0.75,
        games: [
          { game_id: "g1", variant_a_is_white: true, outcome: { status: "finished", result: "white_wins" } },
          { game_id: "g2", variant_a_is_white: false, outcome: { status: "finished", result: "draw" } },
        ],
      }),
    );

    render(<ExperimentView experimentId="exp-1" onOpenGame={() => {}} onBackToSetup={() => {}} />);

    expect(await screen.findByText("Completed")).toBeInTheDocument();
    expect(screen.getByText("2 / 2 games")).toBeInTheDocument();
    expect(screen.getByText("75%")).toBeInTheDocument();
    expect(screen.getByText(/white wins/i)).toBeInTheDocument();
    expect(screen.getByText(/^draw$/i)).toBeInTheDocument();
  });

  it("clicking a game row calls onOpenGame with that game's id", async () => {
    vi.mocked(labClient.getExperiment).mockResolvedValue(
      runningSnapshot({
        status: "completed",
        completed_games: 1,
        wins_a: 1,
        games: [{ game_id: "g1", variant_a_is_white: true, outcome: { status: "finished", result: "white_wins" } }],
      }),
    );
    const onOpenGame = vi.fn();
    const user = userEvent.setup();

    render(<ExperimentView experimentId="exp-1" onOpenGame={onOpenGame} onBackToSetup={() => {}} />);

    await user.click(await screen.findByText(/#1/));

    expect(onOpenGame).toHaveBeenCalledWith("g1");
  });

  it("keeps polling while the experiment is still running", async () => {
    vi.mocked(labClient.getExperiment)
      .mockResolvedValueOnce(runningSnapshot({ completed_games: 0 }))
      .mockResolvedValueOnce(runningSnapshot({ status: "completed", completed_games: 2, draws: 2, score_a: 0.5 }));

    render(<ExperimentView experimentId="exp-1" onOpenGame={() => {}} onBackToSetup={() => {}} />);

    await screen.findByText("0 / 2 games");
    await waitFor(() => expect(screen.getByText("2 / 2 games")).toBeInTheDocument(), { timeout: 3000 });
    expect(vi.mocked(labClient.getExperiment).mock.calls.length).toBeGreaterThanOrEqual(2);
  });

  it("shows a back-to-setup button and error when the fetch fails entirely", async () => {
    vi.mocked(labClient.getExperiment).mockRejectedValue(new Error("no such experiment"));
    const onBackToSetup = vi.fn();
    const user = userEvent.setup();

    render(<ExperimentView experimentId="missing" onOpenGame={() => {}} onBackToSetup={onBackToSetup} />);

    await user.click(await screen.findByRole("button", { name: /back to setup/i }));
    expect(onBackToSetup).toHaveBeenCalled();
  });
});
