import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import { ExperimentView } from "./ExperimentView";
import * as labClient from "./labClient";
import type { ExperimentGame } from "./labClient";
import { experimentSnapshotFixture } from "./testFixtures";

vi.mock("./labClient", async () => {
  const actual = await vi.importActual<typeof labClient>("./labClient");
  return {
    ...actual,
    getExperiment: vi.fn(),
  };
});

function game(overrides: Partial<ExperimentGame> = {}): ExperimentGame {
  return {
    game_id: "g1",
    variant_a_is_white: true,
    outcome: { status: "finished", result: "white_wins", reason: "checkmate" },
    started_at: "2026-01-01T00:00:00Z",
    finished_at: "2026-01-01T00:00:30Z",
    plies: 42,
    ...overrides,
  };
}

describe("ExperimentView", () => {
  it("renders the current tally and progress once the snapshot loads", async () => {
    vi.mocked(labClient.getExperiment).mockResolvedValue(
      experimentSnapshotFixture({
        status: "completed",
        requested_games: 2,
        completed_games: 2,
        wins_a: 1,
        draws: 1,
        score_a: 0.75,
        games: [
          game({ game_id: "g1", variant_a_is_white: true, outcome: { status: "finished", result: "white_wins", reason: "checkmate" } }),
          game({ game_id: "g2", variant_a_is_white: false, outcome: { status: "finished", result: "draw", reason: "stalemate" } }),
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

  it("renders the stats summary (avg duration, avg plies, games/hour)", async () => {
    vi.mocked(labClient.getExperiment).mockResolvedValue(
      experimentSnapshotFixture({
        status: "completed",
        completed_games: 1,
        wins_a: 1,
        games: [game()],
        stats: {
          avg_game_duration_ms: 30_000,
          avg_plies: 42,
          runtime_ms: 45_000,
          games_per_hour: 120,
          variant_a_search: {
            searches: 21, total_nodes: 210_000, avg_nodes: 10_000, avg_time_ms: 50,
            avg_depth: 8.5, max_depth: 11, effective_nps: 200_000, avg_eval_cp: 32,
          },
          variant_b_search: {
            searches: 21, total_nodes: 168_000, avg_nodes: 8_000, avg_time_ms: 50,
            avg_depth: 7, max_depth: 9, effective_nps: 160_000, avg_eval_cp: -15,
          },
          timeouts: 0,
        },
      }),
    );

    render(<ExperimentView experimentId="exp-1" onOpenGame={() => {}} onBackToSetup={() => {}} />);

    expect(await screen.findByText("30.0s")).toBeInTheDocument();
    expect(screen.getByText("45.0s")).toBeInTheDocument();
    expect(screen.getByText("42")).toBeInTheDocument();
    expect(screen.getByText("120.0")).toBeInTheDocument();
    expect(screen.getByText("8.5 / 11")).toBeInTheDocument();
    expect(screen.getByText("+0.32")).toBeInTheDocument();
  });

  it("shows a placeholder for stats that have no data yet", async () => {
    vi.mocked(labClient.getExperiment).mockResolvedValue(experimentSnapshotFixture());

    render(<ExperimentView experimentId="exp-1" onOpenGame={() => {}} onBackToSetup={() => {}} />);

    await screen.findByText(/0 \/ 20 games/);
    // avg duration / avg plies / games-per-hour / elo diff all render
    // "—" rather than a misleading 0 while nothing has settled yet.
    expect(screen.getAllByText("—").length).toBeGreaterThanOrEqual(4);
  });

  it("renders a positive Elo estimate with an explicit + sign", async () => {
    vi.mocked(labClient.getExperiment).mockResolvedValue(
      experimentSnapshotFixture({ score_a: 0.75, elo_diff_a: 190.85 }),
    );

    render(<ExperimentView experimentId="exp-1" onOpenGame={() => {}} onBackToSetup={() => {}} />);

    expect(await screen.findByText("+191")).toBeInTheDocument();
  });

  it("renders a negative Elo estimate without a double sign", async () => {
    vi.mocked(labClient.getExperiment).mockResolvedValue(
      experimentSnapshotFixture({ score_a: 0.25, elo_diff_a: -190.85 }),
    );

    render(<ExperimentView experimentId="exp-1" onOpenGame={() => {}} onBackToSetup={() => {}} />);

    expect(await screen.findByText("-191")).toBeInTheDocument();
  });

  it("shows a placeholder Elo estimate at a perfect score rather than a fake number", async () => {
    vi.mocked(labClient.getExperiment).mockResolvedValue(
      experimentSnapshotFixture({ status: "completed", completed_games: 3, wins_a: 3, score_a: 1.0, elo_diff_a: null }),
    );

    render(<ExperimentView experimentId="exp-1" onOpenGame={() => {}} onBackToSetup={() => {}} />);

    expect(await screen.findByText("100%")).toBeInTheDocument();
    expect(screen.getByText("Elo diff").previousSibling).toHaveTextContent("—");
  });

  it("clicking a game row calls onOpenGame with that game's id", async () => {
    vi.mocked(labClient.getExperiment).mockResolvedValue(
      experimentSnapshotFixture({
        status: "completed",
        completed_games: 1,
        wins_a: 1,
        games: [game({ game_id: "g1", variant_a_is_white: true, outcome: { status: "finished", result: "white_wins", reason: "checkmate" } })],
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
      .mockResolvedValueOnce(experimentSnapshotFixture({ completed_games: 0 }))
      .mockResolvedValueOnce(
        experimentSnapshotFixture({ status: "completed", completed_games: 2, draws: 2, score_a: 0.5 }),
      );

    render(<ExperimentView experimentId="exp-1" onOpenGame={() => {}} onBackToSetup={() => {}} />);

    await screen.findByText("0 / 20 games");
    await waitFor(() => expect(screen.getByText("2 / 20 games")).toBeInTheDocument(), { timeout: 3000 });
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
