import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { useState } from "react";
import { describe, expect, it } from "vitest";
import { Checkbox } from "./Checkbox";

function ControlledCheckbox({ initial }: { initial: boolean }) {
  const [checked, setChecked] = useState(initial);
  return (
    <Checkbox label="Debug logging" checked={checked} onChange={(e) => setChecked(e.target.checked)} />
  );
}

describe("Checkbox", () => {
  it("stays controlled: toggling flips the bound state, not internal DOM state", async () => {
    const user = userEvent.setup();
    render(<ControlledCheckbox initial={false} />);
    const checkbox = screen.getByRole("checkbox", { name: "Debug logging" });

    expect(checkbox).not.toBeChecked();
    await user.click(checkbox);
    expect(checkbox).toBeChecked();
    await user.click(checkbox);
    expect(checkbox).not.toBeChecked();
  });

  it("clicking the label toggles the box, same as clicking the box itself", async () => {
    const user = userEvent.setup();
    render(<ControlledCheckbox initial={false} />);

    await user.click(screen.getByText("Debug logging"));

    expect(screen.getByRole("checkbox", { name: "Debug logging" })).toBeChecked();
  });
});
