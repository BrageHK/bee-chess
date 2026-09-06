import type { SelectHTMLAttributes } from "react";

export type SelectProps = SelectHTMLAttributes<HTMLSelectElement> & {
  invalid?: boolean;
};

/** Same footprint as `Input`/`NumberInput` so a row mixing selects
 * and text/number fields lines up. */
export function Select({ invalid = false, className = "", ...props }: SelectProps) {
  return (
    <select
      aria-invalid={invalid || undefined}
      className={[
        "h-9 w-full rounded-md border bg-surface px-3 text-sm text-text",
        invalid ? "border-danger" : "border-border",
        "focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent",
        "disabled:cursor-not-allowed disabled:opacity-50",
        className,
      ].join(" ")}
      {...props}
    />
  );
}
