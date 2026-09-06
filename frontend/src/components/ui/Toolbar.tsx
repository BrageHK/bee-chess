import type { ReactNode } from "react";

/** A header/toolbar strip: title (or other leading content) on the
 * left, actions/status on the right, with the border Bee Lab uses to
 * separate page regions everywhere else. */
export function Toolbar({ children }: { children: ReactNode }) {
  return (
    <div className="flex items-center justify-between gap-4 border-b border-border pb-4">
      {children}
    </div>
  );
}
