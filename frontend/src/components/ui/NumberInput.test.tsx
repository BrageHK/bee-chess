import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { useState } from "react";
import { describe, expect, it } from "vitest";
import { NumberInput } from "./NumberInput";

/** A thin controlled wrapper -- `NumberInput` has no state of its
 * own, so exercising it means round-tripping through a parent, same
 * as `GameSetup` does. */
function ControlledNumberInput({ initial }: { initial: number }) {
  const [value, setValue] = useState(initial);
  return <NumberInput aria-label="value" value={value} onChange={setValue} />;
}

describe("NumberInput", () => {
  it("reports typed digits as a number", async () => {
    const user = userEvent.setup();
    render(<ControlledNumberInput initial={0} />);
    const input = screen.getByRole("spinbutton", { name: "value" });

    await user.clear(input);
    await user.type(input, "42");

    expect(input).toHaveValue(42);
  });

  it("stays on the field while it's empty, instead of snapping to 0/NaN", async () => {
    const user = userEvent.setup();
    render(<ControlledNumberInput initial={100} />);
    const input = screen.getByRole("spinbutton", { name: "value" });

    await user.clear(input);

    expect(input).toHaveValue(null);
  });

  it("doesn't clobber a leading minus sign while typing a negative number", async () => {
    const user = userEvent.setup();
    render(<ControlledNumberInput initial={0} />);
    const input = screen.getByRole("spinbutton", { name: "value" });

    await user.clear(input);
    await user.type(input, "-5");

    expect(input).toHaveValue(-5);
  });
});
