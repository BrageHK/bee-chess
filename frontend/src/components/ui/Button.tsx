import type { ButtonHTMLAttributes } from "react";

export type ButtonVariant = "primary" | "secondary" | "danger";

const VARIANT_CLASSES: Record<ButtonVariant, string> = {
  primary: "bg-accent text-white border-accent hover:bg-accent-hover hover:border-accent-hover",
  secondary: "bg-surface text-text border-border hover:bg-surface-hover",
  danger: "bg-surface text-danger border-danger hover:bg-danger hover:text-white",
};

export type ButtonProps = ButtonHTMLAttributes<HTMLButtonElement> & {
  variant?: ButtonVariant;
};

/** The one button in Bee Lab. No component should hand-roll its own
 * `<button className=...>` -- see the design-system milestone. */
export function Button({ variant = "secondary", className = "", ...props }: ButtonProps) {
  return (
    <button
      className={[
        "inline-flex h-9 items-center justify-center gap-2 rounded-md border px-3",
        "text-sm font-medium transition-colors",
        "focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent",
        "disabled:cursor-not-allowed disabled:opacity-50 disabled:hover:bg-surface",
        VARIANT_CLASSES[variant],
        className,
      ].join(" ")}
      {...props}
    />
  );
}
