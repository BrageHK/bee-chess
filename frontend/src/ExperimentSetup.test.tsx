import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import { ExperimentSetup } from "./ExperimentSetup";
import * as labClient from "./labClient";

vi.mock("./labClient", async () => {
  const actual = await vi.importActual<typeof labClient>("./labClient");
  return {
    ...actual,
    getEngineOptions: vi.fn().mockResolvedValue([]),
    createExperiment: vi.fn(),
  };
});

describe("ExperimentSetup", () => {
  it("starts with Baseline/Candidate labels and calls onStarted with the new experiment's id", async () => {
    vi.mocked(labClient.createExperiment).mockResolvedValue({
      id: "exp-1",
      status: "running",
      label_a: "Baseline",
      label_b: "Candidate",
      requested_games: 20,
      completed_games: 0,
      wins_a: 0,
      draws: 0,
      wins_b: 0,
      score_a: null,
      games: [],
    });
    const onStarted = vi.fn();
    const user = userEvent.setup();

    render(<ExperimentSetup onStarted={onStarted} />);

    expect(screen.getByDisplayValue("Baseline")).toBeInTheDocument();
    expect(screen.getByDisplayValue("Candidate")).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: /run experiment/i }));

    expect(labClient.createExperiment).toHaveBeenCalledWith(
      expect.objectContaining({
        variantA: { label: "Baseline", options: {} },
        variantB: { label: "Candidate", options: {} },
        games: 20,
        moveTimeMs: 100,
      }),
    );
    expect(onStarted).toHaveBeenCalledWith("exp-1");
  });

  it("disables the submit button while a label is blank", async () => {
    const user = userEvent.setup();
    render(<ExperimentSetup onStarted={() => {}} />);

    const labelInputs = screen.getAllByDisplayValue(/Baseline|Candidate/);
    await user.clear(labelInputs[0]);

    expect(screen.getByRole("button", { name: /run experiment/i })).toBeDisabled();
    expect(screen.getByText(/both variants need a label/i)).toBeInTheDocument();
  });

  it("shows the server's error message if creating the experiment fails", async () => {
    vi.mocked(labClient.createExperiment).mockRejectedValue(new Error("unknown engine"));
    const user = userEvent.setup();

    render(<ExperimentSetup onStarted={() => {}} />);
    await user.click(screen.getByRole("button", { name: /run experiment/i }));

    expect(await screen.findByText("unknown engine")).toBeInTheDocument();
  });
});
