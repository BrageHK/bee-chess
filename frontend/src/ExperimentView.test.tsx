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
            ...experimentSnapshotFixture().stats.variant_a_search,
            searches: 21, total_nodes: 210_000, avg_nodes: 10_000, avg_time_ms: 50,
            avg_depth: 8.5, max_depth: 11, effective_nps: 200_000, avg_eval_cp: 32,
            lmr_attempts: 100, lmr_fail_lows: 80, lmr_researches: 20, lmr_research_rate: 0.2,
            nmp_attempts: 50, nmp_cutoffs: 30, nmp_cutoff_rate: 0.6,
            delta_attempts: 40, delta_pruned: 10, delta_prune_rate: 0.25,
            time_management: null,
          },
          variant_b_search: {
            ...experimentSnapshotFixture().stats.variant_b_search,
            searches: 21, total_nodes: 168_000, avg_nodes: 8_000, avg_time_ms: 50,
            avg_depth: 7, max_depth: 9, effective_nps: 160_000, avg_eval_cp: -15,
            lmr_attempts: 0, lmr_fail_lows: 0, lmr_researches: 0, lmr_research_rate: null,
            nmp_attempts: 0, nmp_cutoffs: 0, nmp_cutoff_rate: null,
            delta_attempts: 0, delta_pruned: 0, delta_prune_rate: null,
            time_management: null,
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
    expect(screen.queryByText("Time management")).not.toBeInTheDocument();
  });

  it("renders the time management panel only when at least one variant has bee-tm telemetry", async () => {
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
            ...experimentSnapshotFixture().stats.variant_a_search,
            searches: 21, total_nodes: 210_000, avg_nodes: 10_000, avg_time_ms: 50,
            avg_depth: 8.5, max_depth: 11, effective_nps: 200_000, avg_eval_cp: 32,
            time_management: {
              searches_with_telemetry: 21,
              avg_soft_ms: 164,
              avg_hard_ms: 492,
              total_aborted_ms: 354,
              avg_aborted_ms: 16.9,
              max_aborted_ms: 354,
              searches_with_aborted_iteration: 1,
              avg_best_move_changes: 0.3,
              avg_score_delta_cp: -5,
            },
          },
          variant_b_search: {
            ...experimentSnapshotFixture().stats.variant_b_search,
            searches: 21, total_nodes: 168_000, avg_nodes: 8_000, avg_time_ms: 50,
            avg_depth: 7, max_depth: 9, effective_nps: 160_000, avg_eval_cp: -15,
            time_management: null,
          },
          timeouts: 0,
        },
      }),
    );

    render(<ExperimentView experimentId="exp-1" onOpenGame={() => {}} onBackToSetup={() => {}} />);

    expect(await screen.findByText("Time management")).toBeInTheDocument();
    expect(screen.getByText("164 ms")).toBeInTheDocument();
    expect(screen.getByText("492 ms")).toBeInTheDocument();
    expect(screen.getByText("no bee-tm telemetry")).toBeInTheDocument();
  });

  it("keeps setting-specific metrics in a collapsed advanced section", async () => {
    vi.mocked(labClient.getExperiment).mockResolvedValue(experimentSnapshotFixture());
    const user = userEvent.setup();

    render(<ExperimentView experimentId="exp-1" onOpenGame={() => {}} onBackToSetup={() => {}} />);

    const summary = await screen.findByText("Advanced metrics");
    const details = summary.closest("details");
    expect(details).not.toHaveAttribute("open");

    await user.click(summary);

    expect(details).toHaveAttribute("open");
    expect(screen.getByText("Late move reductions")).toBeInTheDocument();
    expect(screen.getByText("Null-move pruning")).toBeInTheDocument();
    expect(screen.getByText("Quiescence delta pruning")).toBeInTheDocument();
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
