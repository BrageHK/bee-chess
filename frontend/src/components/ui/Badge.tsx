import type { ReactNode } from "react";

export type BadgeTone = "neutral" | "success" | "warning" | "danger" | "accent";

const TONE_CLASSES: Record<BadgeTone, string> = {
  neutral: "bg-surface-subtle text-muted border-border",
  success: "bg-surface-subtle text-success border-border",
  warning: "bg-surface-subtle text-warning border-border",
  danger: "bg-surface-subtle text-danger border-border",
  accent: "bg-surface-subtle text-accent border-border",
};

/** A small status label, e.g. connection/game state ("Running",
 * "Disconnected", "Error"). Deliberately flat -- no pill shape, no
 * saturated fill -- see the design-system milestone's "boring
 * primitives" rule. */
export function Badge({ tone = "neutral", children }: { tone?: BadgeTone; children: ReactNode }) {
  return (
    <span
      className={[
        "inline-flex items-center gap-1.5 rounded-sm border px-2 py-0.5 text-xs font-medium",
        TONE_CLASSES[tone],
      ].join(" ")}
    >
      {children}
    </span>
  );
}
