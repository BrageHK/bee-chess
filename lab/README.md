# Bee Lab

A Rust server that serves the compiled frontend and is authoritative
for game state: position, move list, clocks, turn, legality, and
result. It owns and supervises the Stockfish/Bee subprocesses for
every game itself, rather than the browser talking to an engine
process directly. See [#67](https://github.com/BrageHK/bee-chess/issues/67)
for the full architecture and [#69](https://github.com/BrageHK/bee-chess/issues/69)
for the authoritative-game-state migration this crate completed.

This is a **separate crate** from `engine/` (the competition `bee`
binary) on purpose: `bee` stays runnable standalone with zero web/
orchestration dependencies. `lab/` is where all of that -- process
supervision, HTTP/WebSocket, static file serving, game state, and
(eventually) a real model registry -- lives instead.

## What it does

- Serves `frontend/dist/` as static files on one port.
- `POST /api/games` creates a game -- `white`/`black` each name an
  engine (`"stockfish"`, `"bee"`, or an object with `setoption`s/debug,
  see `api::CreateGameRequest`) or are omitted for a human-controlled
  side. Lab spawns and drives any engine-controlled side itself
  (`game::run_engine_loop`), asking it for a move and applying it
  through the exact same path a human's move goes through.
- `GET /api/games/:id` returns a complete, self-sufficient snapshot
  (fen, moves, status, and which participant plays each side) -- the
  authoritative resync mechanism a client can always fall back on,
  including after a page refresh (persist only the game id, e.g. in
  the URL).
- `POST /api/games/:id/moves` applies a human's move.
- `GET /ws/games/:id` streams live events for an already-loaded
  snapshot to stay fresh: raw UCI traffic per side
  (`GameEvent::Uci` -- everything a direct engine connection would
  have seen, id/option/info/bestmove, all of it) and the new snapshot
  whenever a move is applied or the game ends (`GameEvent::Updated`).
  Events are transient telemetry, never replayed -- a client that
  connects late or misses some only needs its next `GET`.
- If an engine process dies -- wrong binary, crashes on startup,
  whatever -- before ever producing a reply, the game is marked
  `aborted` with the reason rather than hanging forever.

## What it deliberately does *not* do yet

- **No Bee-Mamba.** The Python/PyTorch engine isn't served here; it
  stays on the legacy `bridge/` for now (that's now all `bridge/`
  does -- see #89). Its fate (ported here too, or handled entirely
  differently once [#66](https://github.com/BrageHK/bee-chess/issues/66)'s model-integration design lands) is a
  follow-up decision.
- **No real engine/model registry.** Stockfish and Bee's paths are
  resolved directly in `main.rs` into a stopgap `EngineRegistry`; the
  real descriptor-based version (ids, options, model references) is
  [#70](https://github.com/BrageHK/bee-chess/issues/70) (67c).
- **No concurrent games.** One game at a time in practice, though the
  API is already game-ID-shaped so this doesn't need an API reshape
  later -- see [#71](https://github.com/BrageHK/bee-chess/issues/71) (67d).

## Running it

Needs Stockfish and Bee built, and the frontend built (not just
`npm run dev`'d -- this serves the static `dist/` output, it doesn't
proxy Vite):

```bash
./scripts/build-stockfish.sh   # first run only, downloads NNUE nets
./scripts/build-bee.sh
npm --prefix frontend run build

cargo run -p bee-lab
# -> bee-lab listening on http://127.0.0.1:8080
```

Override the port with `PORT=<n>` (and, if running the frontend dev
server separately, `VITE_LAB_PORT` on the frontend side to match --
`./scripts/dev.sh` wires this up automatically, including picking a
free port itself if `:8080` is already taken). Note that `bee`'s build
output lives at the repo-root `target/release/bee`, not
`engine/target/release/bee` -- `engine/` and `lab/` are both members
of the root Cargo workspace (see `/Cargo.toml`), so Cargo shares one
`target/` directory across every member.

## Testing

```bash
cd lab && cargo test
```

`api`'s tests cover the HTTP+WebSocket surface end to end (via axum's
`oneshot` for plain requests, a real bound server + `tokio-tungstenite`
client for WebSocket upgrades) against fake engine processes (`sh`
one-liners), so they don't depend on real Stockfish/Bee binaries being
built in the test environment.
