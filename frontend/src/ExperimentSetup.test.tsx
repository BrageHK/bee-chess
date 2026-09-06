import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import { ExperimentSetup } from "./ExperimentSetup";
import * as labClient from "./labClient";
import { experimentSnapshotFixture } from "./testFixtures";

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
    vi.mocked(labClient.createExperiment).mockResolvedValue(experimentSnapshotFixture());
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
        concurrency: 2,
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

  it("requires complete color-swapped game pairs", async () => {
    const user = userEvent.setup();
    render(<ExperimentSetup onStarted={() => {}} />);

    const games = screen.getByRole("spinbutton", { name: "Games" });
    await user.clear(games);
    await user.type(games, "3");

    expect(screen.getByRole("button", { name: /run experiment/i })).toBeDisabled();
    expect(screen.getByText(/positive even number/i)).toBeInTheDocument();
  });

  it("requires concurrency to fit within the requested game count", async () => {
    const user = userEvent.setup();
    render(<ExperimentSetup onStarted={() => {}} />);

    const concurrency = screen.getByRole("spinbutton", { name: "Concurrent games" });
    await user.clear(concurrency);
    await user.type(concurrency, "21");

    expect(screen.getByRole("button", { name: /run experiment/i })).toBeDisabled();
    expect(screen.getByText(/concurrency must be a whole number/i)).toBeInTheDocument();
  });

  it("shows the server's error message if creating the experiment fails", async () => {
    vi.mocked(labClient.createExperiment).mockRejectedValue(new Error("unknown engine"));
    const user = userEvent.setup();

    render(<ExperimentSetup onStarted={() => {}} />);
    await user.click(screen.getByRole("button", { name: /run experiment/i }));

    expect(await screen.findByText("unknown engine")).toBeInTheDocument();
  });
});
