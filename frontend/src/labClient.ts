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

/** Overridable via `VITE_LAB_PORT` (e.g. `.env.local`, or
 * `VITE_LAB_PORT=8081 npm run dev`) for whenever :8080 -- Bee Lab's
 * own default (`lab/src/main.rs`'s `DEFAULT_PORT`) -- is already taken
 * by something else on the machine (Docker Desktop commonly claims it).
 * Must match whatever port Bee Lab was actually started with (its own
 * `PORT` env var) -- the two aren't linked automatically. */
const LAB_PORT = import.meta.env.VITE_LAB_PORT ?? "8080";
const LAB_BASE_URL = `http://localhost:${LAB_PORT}`;
const LAB_WS_BASE_URL = `ws://localhost:${LAB_PORT}`;

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
  uci_log: Array<{ color: Color; direction: "sent" | "received"; line: string }>;
  white: ParticipantInfo;
  black: ParticipantInfo;
  /** The experiment that created this game, if any -- `null` for an
   * ordinary game started from the setup screen. Mirrors
   * `lab::game::GameSnapshot::experiment_id` exactly; lets a game
   * viewer link back to its experiment regardless of how the game was
   * reached (the dashboard, an experiment's own game list, or a
   * bookmarked link). */
  experiment_id: string | null;
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

/** Mirrors `lab::uci_process::UciOption`'s JSON shape exactly (a
 * `#[serde(tag = "type", rename_all = "lowercase")]` enum) -- UCI's
 * own generic option vocabulary, not anything engine-specific. This is
 * the whole point: a config UI renders whichever of these an engine
 * happens to advertise (`check` -> a checkbox, `spin` -> a bounded
 * number field, `combo` -> a dropdown, `string` -> free text) without
 * needing to know the option's name ahead of time. See
 * `GameSetup.tsx`'s `EngineOptionsFields`. */
export type EngineOption =
  | { type: "check"; name: string; default: boolean }
  | { type: "spin"; name: string; default: number; min: number; max: number }
  | { type: "combo"; name: string; default: string; values: string[] }
  | { type: "string"; name: string; default: string };

/** One side of an experiment being requested -- mirrors `api::
 * ExperimentVariantRequest`'s JSON shape. No `engine` field here
 * (unlike `ParticipantRequest`): v1 experiments are Bee-vs-Bee only
 * (see `lab::experiment`'s module docs), so `CreateExperimentRequest`
 * names the engine once for the whole experiment, not per variant. */
export interface ExperimentVariantRequest {
  label: string;
  options?: Record<string, string | number | boolean>;
}

export interface CreateExperimentRequest {
  /** Defaults to `"bee"` server-side if omitted -- see `api::
   * CreateExperimentRequest`'s docs. */
  engine?: string;
  variantA: ExperimentVariantRequest;
  variantB: ExperimentVariantRequest;
  games: number;
  concurrency?: number;
  moveTimeMs?: number;
  debug?: boolean;
}

/** Mirrors `lab::experiment::GameOutcome`'s JSON shape exactly (a
 * `#[serde(tag = "status", rename_all = "snake_case")]` enum). */
export type GameOutcome =
  | { status: "pending" }
  | { status: "finished"; result: "white_wins" | "black_wins" | "draw" }
  | { status: "aborted" };

/** Mirrors `lab::experiment::ExperimentGame`'s JSON shape exactly. */
export interface ExperimentGame {
  game_id: string;
  variant_a_is_white: boolean;
  outcome: GameOutcome;
  started_at: string;
  /** `null` while the game is still running. */
  finished_at: string | null;
  /** `null` for a still-running game, or one that aborted before a
   * final snapshot was available -- see the Rust type's own docs. */
  plies: number | null;
}

/** Mirrors `lab::experiment::ExperimentMetadata`'s JSON shape exactly
 * -- enough about how/when an experiment ran to make its numbers
 * interpretable again later. */
export interface ExperimentMetadata {
  lab_git_commit: string;
  variant_a_argv: string[];
  variant_b_argv: string[];
  started_at: string;
  finished_at: string | null;
}

/** Mirrors `lab::experiment::ExperimentStats`'s JSON shape exactly.
 * Every average is `null` (not `0`) with no settled games yet, same
 * reasoning as `ExperimentSnapshot.score_a`. */
export interface ExperimentStats {
  avg_game_duration_ms: number | null;
  avg_plies: number | null;
  runtime_ms: number;
  games_per_hour: number | null;
  variant_a_search: ExperimentSearchStats;
  variant_b_search: ExperimentSearchStats;
}

export interface ExperimentSearchStats {
  searches: number;
  total_nodes: number;
  avg_nodes: number | null;
  avg_time_ms: number | null;
  avg_depth: number | null;
  max_depth: number | null;
  effective_nps: number | null;
  avg_eval_cp: number | null;
}

/** Mirrors `lab::experiment::ExperimentSnapshot`'s JSON shape exactly
 * -- the complete, self-sufficient resync payload
 * `GET /api/experiments/:id` returns, same "authoritative snapshot"
 * philosophy as `GameSnapshot`. Field names stay the server's own
 * snake_case (no camelCase re-mapping) to match how this module
 * already treats `GameSnapshot`/`ExperimentGame` -- one direct mirror
 * of the wire shape, not a second naming convention layered on top. */
export interface ExperimentSnapshot {
  id: string;
  status: "running" | "completed";
  label_a: string;
  label_b: string;
  requested_games: number;
  concurrency: number;
  completed_games: number;
  wins_a: number;
  draws: number;
  wins_b: number;
  score_a: number | null;
  /** A's estimated Elo advantage over B, derived from `score_a` --
   * `null` whenever `score_a` is, plus at a perfect 0%/100% score
   * (see `lab::experiment::elo_diff_from_score`'s docs on why those
   * two can't produce a finite number). A point estimate only -- no
   * confidence interval yet. */
  elo_diff_a: number | null;
  games: ExperimentGame[];
  metadata: ExperimentMetadata;
  stats: ExperimentStats;
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

/** `GET /api/games` -- every game the server currently knows about,
 * newest first. Powers the dashboard's running/past game lists (it
 * filters this one list client-side by `status` rather than Lab
 * offering separate endpoints per status). */
export async function listGames(): Promise<GameSnapshot[]> {
  const response = await fetch(`${LAB_BASE_URL}/api/games`);
  return parseJsonOrThrow(response, "list games");
}

/** `GET /api/engines/:name/options` -- the UCI options `name` (e.g.
 * `"bee"`) advertises during its own handshake. See `EngineOption`'s
 * docs for why the frontend renders this generically rather than
 * hardcoding any option's name. */
export async function getEngineOptions(name: string): Promise<EngineOption[]> {
  const response = await fetch(`${LAB_BASE_URL}/api/engines/${name}/options`);
  return parseJsonOrThrow(response, `get ${name} options`);
}

/** `POST /api/experiments`. Returns immediately with the experiment's
 * id (status `"running"`, no games yet) -- Lab runs it to completion
 * in the background; see `getExperiment` for the resync mechanism
 * that follows its progress. */
export async function createExperiment(request: CreateExperimentRequest): Promise<ExperimentSnapshot> {
  const body: Record<string, unknown> = {
    variant_a: request.variantA,
    variant_b: request.variantB,
    games: request.games,
  };
  if (request.engine !== undefined) body.engine = request.engine;
  if (request.concurrency !== undefined) body.concurrency = request.concurrency;
  if (request.moveTimeMs !== undefined) body.move_time_ms = request.moveTimeMs;
  if (request.debug !== undefined) body.debug = request.debug;

  const response = await fetch(`${LAB_BASE_URL}/api/experiments`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(body),
  });
  return parseJsonOrThrow(response, "create experiment");
}

/** `GET /api/experiments/:id` -- the authoritative resync mechanism
 * for an experiment's progress. No live WebSocket stream exists for
 * experiments (unlike games); a caller that wants to follow progress
 * polls this, the same fallback `Game.tsx` already relies on for a
 * game's own snapshot. */
export async function getExperiment(id: string): Promise<ExperimentSnapshot> {
  const response = await fetch(`${LAB_BASE_URL}/api/experiments/${id}`);
  return parseJsonOrThrow(response, "get experiment");
}

/** `GET /api/experiments` -- every experiment the server currently
 * knows about, newest first. Same "one list, filter client-side by
 * status" reasoning as `listGames`. */
export async function listExperiments(): Promise<ExperimentSnapshot[]> {
  const response = await fetch(`${LAB_BASE_URL}/api/experiments`);
  return parseJsonOrThrow(response, "list experiments");
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
