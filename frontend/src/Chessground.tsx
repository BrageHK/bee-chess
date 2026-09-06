import { useEffect, useRef } from "react";
import { Chessground as cg } from "@lichess-org/chessground";
import type { Api } from "@lichess-org/chessground/api";
import type { Config } from "@lichess-org/chessground/config";

/** Fixed board size -- matches `EvalBar`'s own fixed `HEIGHT` (see
 * EvalBar.tsx), since the two sit side by side and are meant to line
 * up.
 *
 * A responsive (shrink-on-narrow-viewport) version of this was tried
 * here and reverted: `aspect-square` sizing on a flex item inside
 * Game.tsx's row rendered a full-height, near-zero-width board
 * instead of scaling down cleanly, and it wasn't practical to debug
 * further without a browser inspector attached to this session. Back
 * to a fixed size until that's revisited with real visual tooling. */
const SIZE_PX = 480;

export function Chessground({ config }: { config: Config }) {
  const ref = useRef<HTMLDivElement>(null);
  const apiRef = useRef<Api | null>(null);
  const lastFenRef = useRef(config.fen);

  // Chessground's `set` API cannot reconfigure `viewOnly` or
  // `drawable.visible`. In particular, a view-only instance is created
  // without any board/document pointer handlers, so merely setting
  // `viewOnly: false` after the initial Lab snapshot arrives leaves the
  // pieces impossible to move. Recreate the instance when either of
  // those construction-time options changes.
  //
  // config/config.fen are deliberately excluded from the deps below --
  // the second effect (api.set) is what applies every other change,
  // including a new fen; recreating on every fen change would also
  // destroy drawn arrows on every 500ms Lab poll (see that effect).
  useEffect(() => {
    if (ref.current) {
      apiRef.current = cg(ref.current, config);
      lastFenRef.current = config.fen;
    }
    return () => {
      apiRef.current?.destroy();
      apiRef.current = null;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [config.viewOnly, config.drawable?.visible]);

  useEffect(() => {
    const api = apiRef.current;
    if (!api) return;

    if (config.fen === lastFenRef.current) {
      // Chessground clears drawable.shapes whenever `set` contains a FEN,
      // even if it is identical to the current position. Lab polls produce
      // a fresh config every 500 ms, so omit an unchanged FEN to preserve
      // right-click arrows between authoritative snapshot refreshes.
      const update = { ...config };
      delete update.fen;
      api.set(update);
      return;
    }

    api.set(config);
    lastFenRef.current = config.fen;
  }, [config]);

  return <div ref={ref} style={{ width: SIZE_PX, height: SIZE_PX }} />;
}
