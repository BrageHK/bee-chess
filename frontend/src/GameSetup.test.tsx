import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import { GameSetup } from "./GameSetup";
import * as labClient from "./labClient";

vi.mock("./labClient", async () => {
  const actual = await vi.importActual<typeof labClient>("./labClient");
  return {
    ...actual,
    checkLabAvailable: vi.fn().mockResolvedValue(true),
    getEngineOptions: vi.fn(),
  };
});

/** `GameSetup`'s White slot defaults to Bee (see `defaultParticipant`
 * in its own initial state), so rendering the whole component -- there
 * is no exported way to reach `SlotPicker`/`EngineOptionsFields`
 * directly -- exercises Bee's options discovery without any
 * interaction needed to select it first. */
function BeeParticipantFields() {
  return (
    <GameSetup
      onStart={() => {
        /* not exercised here */
      }}
    />
  );
}

describe("GameSetup's Bee options discovery", () => {
  it("renders a discovered check option as a checkbox with its own name as the label", async () => {
    vi.mocked(labClient.getEngineOptions).mockResolvedValue([
      { type: "check", name: "UseTT", default: true },
    ]);

    render(<BeeParticipantFields />);

    // Bee is White's default slot (see defaultParticipant/GameSetup's
    // initial state), so its options load without any interaction.
    await waitFor(() => {
      expect(screen.getByRole("checkbox", { name: "UseTT" })).toBeChecked();
    });
  });

  it("renders a discovered combo option as a select with its values as options", async () => {
    vi.mocked(labClient.getEngineOptions).mockResolvedValue([
      {
        type: "combo",
        name: "Evaluator",
        default: "Positional",
        values: ["Positional", "Material", "Experimental"],
      },
    ]);

    render(<BeeParticipantFields />);

    const select = await screen.findByRole("combobox", { name: "Evaluator" });
    expect(select).toHaveValue("Positional");
    expect(screen.getByRole("option", { name: "Material" })).toBeInTheDocument();
    expect(screen.getByRole("option", { name: "Experimental" })).toBeInTheDocument();
  });

  it("toggling a discovered checkbox updates its value", async () => {
    vi.mocked(labClient.getEngineOptions).mockResolvedValue([
      { type: "check", name: "UseQuiescence", default: true },
    ]);
    const user = userEvent.setup();

    render(<BeeParticipantFields />);

    const checkbox = await screen.findByRole("checkbox", { name: "UseQuiescence" });
    expect(checkbox).toBeChecked();

    await user.click(checkbox);

    expect(checkbox).not.toBeChecked();
  });

  it("shows an error rather than crashing when option discovery fails", async () => {
    vi.mocked(labClient.getEngineOptions).mockRejectedValue(new Error("network error"));

    render(<BeeParticipantFields />);

    expect(await screen.findByText(/couldn't load bee's options/i)).toBeInTheDocument();
  });
});
