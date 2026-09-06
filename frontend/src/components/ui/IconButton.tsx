import type { ButtonHTMLAttributes } from "react";

export type IconButtonProps = ButtonHTMLAttributes<HTMLButtonElement> & {
  /** Required since an icon-only button has no visible label. */
  "aria-label": string;
};

/** A square, icon-only button -- same states as `Button`, sized for a
 * single glyph (e.g. toolbar actions) rather than text. */
export function IconButton({ className = "", ...props }: IconButtonProps) {
  return (
    <button
      className={[
        "inline-flex h-9 w-9 items-center justify-center rounded-md border border-border",
        "bg-surface text-text transition-colors hover:bg-surface-hover",
        "focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent",
        "disabled:cursor-not-allowed disabled:opacity-50 disabled:hover:bg-surface",
        className,
      ].join(" ")}
      {...props}
    />
  );
}
