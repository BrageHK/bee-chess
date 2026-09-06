import type { InputHTMLAttributes } from "react";

export type InputProps = InputHTMLAttributes<HTMLInputElement> & {
  invalid?: boolean;
};

/** Base text input. Same height/border/focus ring as `NumberInput`
 * and `Select`, so a form doesn't visibly mix control sizes. */
export function Input({ invalid = false, className = "", ...props }: InputProps) {
  return (
    <input
      aria-invalid={invalid || undefined}
      className={[
        "h-9 w-full rounded-md border bg-surface px-3 text-sm text-text",
        "placeholder:text-subtle",
        invalid ? "border-danger" : "border-border",
        "focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent",
        "disabled:cursor-not-allowed disabled:opacity-50",
        className,
      ].join(" ")}
      {...props}
    />
  );
}
