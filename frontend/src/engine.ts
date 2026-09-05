/**
 * `UciLogLine`: one line of raw UCI traffic, for the log/eval/search-
 * stats panels (`UciLogPanel`, `EvalBar`, `SearchStatsPanel`).
 *
 * This module used to also own `UciClient`, a direct-WebSocket UCI
 * client the frontend used to drive an engine process itself against
 * `bridge/server.py`. Per #69/67b, Bee Lab is now authoritative for
 * game state and drives every engine-controlled side itself
 * (`lab/src/game.rs`'s `run_engine_loop`) -- the frontend
 * (`Game.tsx`, `labClient.ts`) only ever talks to Lab's HTTP+WebSocket
 * API now, never to an engine process directly, so that client (and
 * `createBotClient`/`checkBotAvailable`/`BotKind`, its supporting
 * pieces) had no remaining callers and was removed. `UciLogLine`
 * itself survives because `Game.tsx`'s `logSubscribeFor` still
 * produces this exact shape (from Lab's `GameEvent::Uci` stream
 * instead of a direct WebSocket) for those three panels to consume
 * unchanged.
 */
export interface UciLogLine {
  direction: "sent" | "received";
  text: string;
  timestamp: number;
}
