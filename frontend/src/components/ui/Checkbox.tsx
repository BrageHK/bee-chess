import type { InputHTMLAttributes, ReactNode } from "react";

export type CheckboxProps = Omit<InputHTMLAttributes<HTMLInputElement>, "type"> & {
  label: ReactNode;
};

/** Label and box are laid out with `items-start` and a fixed,
 * `shrink-0` box so a long/wrapping label can't squash or resize the
 * checkbox itself. */
export function Checkbox({ label, className = "", id, ...props }: CheckboxProps) {
  return (
    <label className={["flex items-start gap-2 text-sm text-text", className].join(" ")}>
      <input
        type="checkbox"
        id={id}
        className={[
          "mt-0.5 h-4 w-4 shrink-0 rounded-sm border border-border-strong bg-surface accent-accent",
          "focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent",
          "disabled:cursor-not-allowed disabled:opacity-50",
        ].join(" ")}
        {...props}
      />
      <span>{label}</span>
    </label>
  );
}
