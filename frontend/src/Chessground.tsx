import { useEffect, useRef } from "react";
import { Chessground as cg } from "@lichess-org/chessground";
import type { Api } from "@lichess-org/chessground/api";
import type { Config } from "@lichess-org/chessground/config";

export function Chessground({ config }: { config: Config }) {
  const ref = useRef<HTMLDivElement>(null);
  const apiRef = useRef<Api | null>(null);

  useEffect(() => {
    if (ref.current && !apiRef.current) apiRef.current = cg(ref.current, config);
    return () => { apiRef.current?.destroy(); apiRef.current = null; };
  }, []);

  useEffect(() => { apiRef.current?.set(config); }, [config]);

  return <div ref={ref} style={{ width: 480, height: 480 }} />;
}
