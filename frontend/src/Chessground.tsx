import { useEffect, useRef } from "react";
import { Chessground as cg } from "@lichess-org/chessground";
import type { Api } from "@lichess-org/chessground/api";
import type { Config } from "@lichess-org/chessground/config";

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
  useEffect(() => {
    if (ref.current) {
      apiRef.current = cg(ref.current, config);
      lastFenRef.current = config.fen;
    }
    return () => {
      apiRef.current?.destroy();
      apiRef.current = null;
    };
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

  return <div ref={ref} style={{ width: 480, height: 480 }} />;
}
