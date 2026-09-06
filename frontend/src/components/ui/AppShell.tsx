import type { ReactNode } from "react";

/** The page's outer frame: a centered, max-width column that every
 * screen (setup, game) renders inside, so the app's overall width and
 * horizontal padding are defined in exactly one place. */
export function AppShell({ children }: { children: ReactNode }) {
  return (
    <div className="mx-auto flex min-h-svh w-full max-w-5xl flex-col gap-4 px-6 py-6">
      {children}
    </div>
  );
}
