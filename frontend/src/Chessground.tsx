import { useEffect, useRef } from "react";
import { Chessground as cg } from "@lichess-org/chessground";
import type { Api } from "@lichess-org/chessground/api";
import type { Config } from "@lichess-org/chessground/config";

export function Chessground({ config }: { config: Config }) {
  const ref = useRef<HTMLDivElement>(null);
  const apiRef = useRef<Api | null>(null);

  // Chessground's `set` API cannot reconfigure `viewOnly` or
  // `drawable.visible`. In particular, a view-only instance is created
  // without any board/document pointer handlers, so merely setting
  // `viewOnly: false` after the initial Lab snapshot arrives leaves the
  // pieces impossible to move. Recreate the instance when either of
  // those construction-time options changes.
  useEffect(() => {
    if (ref.current) apiRef.current = cg(ref.current, config);
    return () => { apiRef.current?.destroy(); apiRef.current = null; };
  }, [config.viewOnly, config.drawable?.visible]);

  useEffect(() => { apiRef.current?.set(config); }, [config]);

  return <div ref={ref} style={{ width: 480, height: 480 }} />;
}
