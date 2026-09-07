import { fireEvent, render, screen } from "@testing-library/react";
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
        timeControl: { type: "move_time", move_time_ms: 100 },
      }),
    );
    expect(onStarted).toHaveBeenCalledWith("exp-1");
  });

  it("switching to Fischer sends initial/increment in milliseconds", async () => {
    vi.mocked(labClient.createExperiment).mockResolvedValue(experimentSnapshotFixture());
    const user = userEvent.setup();

    const { container } = render(<ExperimentSetup onStarted={() => {}} />);

    await user.selectOptions(screen.getByRole("combobox", { name: "Time control" }), "fischer");
    const initial = screen.getByRole("spinbutton", { name: "Initial (minutes)" });
    await user.clear(initial);
    await user.type(initial, "3");
    const increment = screen.getByRole("spinbutton", { name: "Increment (seconds)" });
    await user.clear(increment);
    await user.type(increment, "2");

    // `user.click`/a native `<button type="submit">` click don't
    // reliably trigger form submission in jsdom once a `<select>` in
    // the same form has been interacted with beforehand (a jsdom
    // quirk, not component behavior -- `fireEvent.submit` and
    // `form.requestSubmit()` both still work correctly and exercise
    // the exact same `onSubmit` handler).
    fireEvent.submit(container.querySelector("form")!);

    expect(labClient.createExperiment).toHaveBeenCalledWith(
      expect.objectContaining({
        timeControl: { type: "fischer", initial_ms: 180_000, increment_ms: 2_000 },
      }),
    );
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
