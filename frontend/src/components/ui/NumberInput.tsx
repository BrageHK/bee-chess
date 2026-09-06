import { useState } from "react";
import { Input, type InputProps } from "./Input";

export type NumberInputProps = Omit<InputProps, "type" | "value" | "onChange"> & {
  value: number;
  onChange: (value: number) => void;
};

/** A numeric input that always reports a `number` to `onChange`,
 * rather than the raw string `<input type="number">` normally hands
 * back.
 *
 * The field keeps its own draft string rather than deriving the
 * displayed text straight from `value`: while the user is mid-edit
 * (an empty field, a bare "-") there is no valid `number` to report
 * yet, so `onChange` isn't called and `value` doesn't change -- if
 * the input then re-rendered from `value` alone, that unchanged prop
 * would snap the field straight back to its last committed digits on
 * every keystroke. `value` is re-derived during render (the
 * "adjusting state when a prop changes" pattern) rather than synced
 * via an effect, so a sibling field resetting the form is reflected
 * in the same render instead of one tick later. */
export function NumberInput({ value, onChange, ...props }: NumberInputProps) {
  const [draft, setDraft] = useState(String(value));
  const [lastValue, setLastValue] = useState(value);

  if (value !== lastValue && Number(draft) !== value) {
    setLastValue(value);
    setDraft(String(value));
  }

  return (
    <Input
      type="number"
      value={draft}
      onChange={(e) => {
        const raw = e.target.value;
        setDraft(raw);
        if (raw === "" || raw === "-") return;
        const next = Number(raw);
        if (!Number.isNaN(next)) onChange(next);
      }}
      {...props}
    />
  );
}
