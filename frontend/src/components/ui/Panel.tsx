import type { HTMLAttributes, ReactNode } from "react";

export type PanelProps = HTMLAttributes<HTMLDivElement>;

/** A bordered surface -- the one container Bee Lab's screens are
 * built from (game config, engine cards, log views, ...). */
export function Panel({ className = "", ...props }: PanelProps) {
  return (
    <div
      className={["rounded-md border border-border bg-surface", className].join(" ")}
      {...props}
    />
  );
}

export function PanelHeader({ children, className = "" }: { children: ReactNode; className?: string }) {
  return (
    <div
      className={[
        "border-b border-border px-4 py-2 text-sm font-medium text-text",
        className,
      ].join(" ")}
    >
      {children}
    </div>
  );
}

export function PanelBody({ children, className = "" }: { children: ReactNode; className?: string }) {
  return <div className={["p-4", className].join(" ")}>{children}</div>;
}
