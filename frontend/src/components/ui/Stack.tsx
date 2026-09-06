import type { HTMLAttributes } from "react";

export type SpacingProps = HTMLAttributes<HTMLDivElement> & {
  /** Gap, in the design system's spacing scale (4px increments). */
  gap?: 1 | 2 | 3 | 4 | 6 | 8;
  align?: "start" | "center" | "end" | "stretch";
};

const GAP_CLASSES: Record<NonNullable<SpacingProps["gap"]>, string> = {
  1: "gap-1",
  2: "gap-2",
  3: "gap-3",
  4: "gap-4",
  6: "gap-6",
  8: "gap-8",
};

const ALIGN_CLASSES: Record<NonNullable<SpacingProps["align"]>, string> = {
  start: "items-start",
  center: "items-center",
  end: "items-end",
  stretch: "items-stretch",
};

/** A vertical flex layout with a spacing-scale gap -- the replacement
 * for ad-hoc `style={{ display: "grid", gap: N }}` scattered across
 * feature components. */
export function Stack({ gap = 4, align = "stretch", className = "", ...props }: SpacingProps) {
  return (
    <div
      className={["flex flex-col", GAP_CLASSES[gap], ALIGN_CLASSES[align], className].join(" ")}
      {...props}
    />
  );
}

/** Same as `Stack`, laid out horizontally; wraps by default since
 * Bee Lab's rows (slot pickers, board + eval bars) need to reflow on
 * narrow viewports rather than force a horizontal scroll. */
export function Inline({ gap = 4, align = "center", className = "", ...props }: SpacingProps) {
  return (
    <div
      className={["flex flex-wrap", GAP_CLASSES[gap], ALIGN_CLASSES[align], className].join(" ")}
      {...props}
    />
  );
}
