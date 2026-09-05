/**
 * Client for the Bee Lab HTTP+WebSocket API (`lab/`, see #69/67b) --
 * the authoritative-game-state boundary Game.tsx talks to instead of
 * computing position/turn/legality/result itself.
 *
 * Per ADR 0001/#67: the frontend never spawns an engine process itself
 * or decides whether a move happened -- it asks Lab to create a game,
 * renders whatever snapshot Lab reports, and (for a human's move) asks
 * Lab to apply it, trusting Lab's answer either way. `GET /api/games/:id`
 * (via `getGame`) is the resync mechanism a client can always fall back
 * on; `subscribeToGameEvents`'s WebSocket only exists to avoid needing
 * to poll for the live UCI log/eval telemetry panels.
 */

const LAB_BASE_URL = "http://localhost:8080";
const LAB_WS_BASE_URL = "ws://localhost:8080";

export type Color = "white" | "black";

/** Mirrors `game::GameStatus`'s JSON shape exactly (`#[serde(flatten)]`
 * puts `result`/`reason` as siblings of `status`, not nested). */
export type GameStatus =
  | { status: "running" }
  | { status: "finished"; result: "white_wins" | "black_wins" | "draw" }
  | { status: "aborted"; reason: string };

/** Mirrors `game::ParticipantInfo`'s JSON shape exactly -- who plays
 * one side of a game, and enough to reconstruct that side's UI (is it
 * a human who can drag pieces? which engine's eval bar is this?) from
 * the snapshot alone, without a client needing to have remembered
 * `Participant` configuration on its own. This is what lets a page
 * refresh resume a game by persisting only its id (see `App.tsx`). */
export type ParticipantInfo = { kind: "human" } | { kind: "engine"; name: string; debug: boolean };

/** Mirrors `game::GameSnapshot`'s JSON shape exactly -- the complete,
 * self-sufficient resync payload `GET /api/games/:id` returns. */
export type GameSnapshot = {
  id: string;
  fen: string;
  moves: string[];
  white: ParticipantInfo;
  black: ParticipantInfo;
} & GameStatus;

/** One side's requested participant for `createGame` -- mirrors
 * `api::ParticipantRequest`'s untagged JSON shape: either a bare engine
 * name, or an object with `setoption`s/debug. */
export type ParticipantRequest =
  | string
  | { engine: string; options?: Record<string, string | number | boolean>; debug?: boolean };

export interface CreateGameRequest {
  white?: ParticipantRequest;
  black?: ParticipantRequest;
  moveTimeMs?: number;
}

class LabError extends Error {}

async function parseJsonOrThrow<T>(response: Response, what: string): Promise<T> {
  if (!response.ok) {
    const body = await response.json().catch(() => null);
    const detail = body && typeof body === "object" && "error" in body ? String(body.error) : response.statusText;
    throw new LabError(`${what} failed (${response.status}): ${detail}`);
  }
  return response.json() as Promise<T>;
}

/** `POST /api/games`. `moveTimeMs` becomes `move_time_ms`; omitted
 * fields aren't sent at all, matching every field's `#[serde(default)]`
 * on the Rust side. */
export async function createGame(request: CreateGameRequest = {}): Promise<GameSnapshot> {
  const body: Record<string, unknown> = {};
  if (request.white !== undefined) body.white = request.white;
  if (request.black !== undefined) body.black = request.black;
  if (request.moveTimeMs !== undefined) body.move_time_ms = request.moveTimeMs;

  const response = await fetch(`${LAB_BASE_URL}/api/games`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(body),
  });
  return parseJsonOrThrow(response, "create game");
}

/** `GET /api/games/:id` -- the authoritative resync mechanism. */
export async function getGame(id: string): Promise<GameSnapshot> {
  const response = await fetch(`${LAB_BASE_URL}/api/games/${id}`);
  return parseJsonOrThrow(response, "get game");
}

/**
 * Briefly checks whether Bee Lab itself is reachable at all --
 * enough to warn "Lab doesn't seem to be running" on the setup screen
 * before the user configures a whole game around it, rather than only
 * failing once `createGame` is called. Lab `require()`s both Stockfish
 * and Bee at startup (see `lab/src/main.rs`) and refuses to start
 * without them, so unlike the old per-bot-port bridge probe this used
 * to replace, a single "is Lab up" check covers every engine Lab
 * currently supports -- there's no per-engine reachability question
 * to ask separately anymore.
 */
export async function checkLabAvailable(timeoutMs = 1500): Promise<boolean> {
  try {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), timeoutMs);
    try {
      const response = await fetch(LAB_BASE_URL, { signal: controller.signal });
      return response.ok;
    } finally {
      clearTimeout(timer);
    }
  } catch {
    return false;
  }
}

/** `POST /api/games/:id/moves`. Rejects (does not silently ignore) if
 * Lab refuses the move for any reason -- illegal, game not running, or
 * the game doesn't exist -- so a caller can't mistake a rejected move
 * for an applied one. */
export async function postMove(id: string, uci: string): Promise<GameSnapshot> {
  const response = await fetch(`${LAB_BASE_URL}/api/games/${id}/moves`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ uci }),
  });
  return parseJsonOrThrow(response, "apply move");
}

/** Mirrors `api::GameEventWire`'s JSON shape exactly. */
export type GameEvent =
  | { type: "uci"; color: Color; direction: "sent" | "received"; line: string }
  | { type: "updated"; snapshot: GameSnapshot };

/**
 * Subscribes to `id`'s live event stream (`GET /ws/games/:id`).
 * Returns an unsubscribe function that closes the socket. Events are
 * transient telemetry, not replayed history (see this module's and
 * the lab server's own docs) -- a caller that needs the current
 * authoritative state should call `getGame`, not wait for an
 * `"updated"` event to arrive.
 */
export function subscribeToGameEvents(id: string, onEvent: (event: GameEvent) => void): () => void {
  const ws = new WebSocket(`${LAB_WS_BASE_URL}/ws/games/${id}`);
  ws.onmessage = (e: MessageEvent<string>) => {
    try {
      onEvent(JSON.parse(e.data) as GameEvent);
    } catch {
      // Malformed frame -- ignore rather than crash the subscriber;
      // this is telemetry, not the authoritative resync path.
    }
  };
  return () => ws.close();
}

export { LabError };
