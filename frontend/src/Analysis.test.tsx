import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { Analysis } from "./Analysis";
import * as client from "./analysisClient";
import { analysisReportFixture as report, analysisRunFixture as run, reviewMoveFixture as move } from "./analysisFixtures";

vi.mock("./analysisClient", () => ({ listAnalysisRuns: vi.fn(), getAnalysisReport: vi.fn(), getAnalysisMove: vi.fn() }));
vi.mock("./Chessground", () => ({ Chessground: () => <div>Review board</div> }));

beforeEach(() => {
  vi.mocked(client.listAnalysisRuns).mockResolvedValue([run]);
  vi.mocked(client.getAnalysisReport).mockResolvedValue(report);
  vi.mocked(client.getAnalysisMove).mockResolvedValue(move);
});

describe("Analysis dashboard", () => {
  it("shows run totals, strict mistake thresholds, SAN moves, and the weakest phase", async () => {
    render(<Analysis runId={1} moveId={null} onNavigate={() => {}} />);
    expect(await screen.findByText("Highest average loss:", { exact: false })).toHaveTextContent("middlegame (78.0cp)");
    expect(screen.getByText(">400cp blunders")).toBeInTheDocument();
    const table = screen.getByRole("table", { name: "Worst Bee moves" });
    expect(within(table).getByText("g4")).toBeInTheDocument();
    expect(within(table).getByText("e4")).toBeInTheDocument();
    expect(within(table).getByText("412")).toBeInTheDocument();
    expect(screen.getByText(/82% of game CPL/)).toBeInTheDocument();
  });

  it("filters the table and opens a review with the run and move IDs", async () => {
    const user = userEvent.setup();
    const navigate = vi.fn();
    render(<Analysis runId={1} moveId={null} onNavigate={navigate} />);
    await user.selectOptions(await screen.findByRole("combobox", { name: "Phase" }), "middlegame");
    await user.selectOptions(screen.getByRole("combobox", { name: "Loss threshold" }), "200");
    await user.click(screen.getByRole("checkbox", { name: "Only lost games" }));
    await waitFor(() => expect(client.getAnalysisReport).toHaveBeenLastCalledWith(1, expect.objectContaining({ phase: "middlegame", over_cp: 200, losses_only: true, offset: 0 })));
    await user.click(screen.getByRole("button", { name: "Review abc123 ply 0" }));
    expect(navigate).toHaveBeenCalledWith(1, 7);
  });

  it("restores a bookmarked review independently of the current table page", async () => {
    render(<Analysis runId={1} moveId={7} onNavigate={() => {}} />);
    expect(await screen.findByText("Review board")).toBeInTheDocument();
    expect(client.getAnalysisMove).toHaveBeenCalledWith(1, 7);
    expect(screen.getByRole("link", { name: /open original game/i })).toHaveAttribute("href", move.original_game_url);
  });

  it("handles empty catalogs and a failed request with refresh", async () => {
    vi.mocked(client.listAnalysisRuns).mockResolvedValue([]);
    const user = userEvent.setup();
    render(<Analysis runId={null} moveId={null} onNavigate={() => {}} />);
    expect(await screen.findByText("No analysis runs yet")).toBeInTheDocument();
    vi.mocked(client.listAnalysisRuns).mockRejectedValueOnce(new Error("Catalog unavailable"));
    await user.click(screen.getByRole("button", { name: "Refresh analysis" }));
    expect(await screen.findByRole("alert")).toHaveTextContent("Catalog unavailable");
    await user.click(screen.getByRole("button", { name: "Refresh analysis" }));
    await waitFor(() => expect(screen.queryByRole("alert")).not.toBeInTheDocument());
  });

  it("does not let a late response replace a newly selected run", async () => {
    let resolveOld!: (value: typeof report) => void;
    vi.mocked(client.getAnalysisReport).mockImplementation((id) => id === 1
      ? new Promise((resolve) => { resolveOld = resolve; })
      : Promise.resolve({ ...report, run: { ...run, id: 2 }, moves: [], matching_moves: 0 }));
    const { rerender } = render(<Analysis runId={1} moveId={null} onNavigate={() => {}} />);
    rerender(<Analysis runId={2} moveId={null} onNavigate={() => {}} />);
    await screen.findByText("No moves match these filters.");
    resolveOld(report);
    await waitFor(() => expect(screen.queryByRole("button", { name: "Review abc123 ply 0" })).not.toBeInTheDocument());
  });
});
