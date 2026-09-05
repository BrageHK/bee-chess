# Bee Lab

A Rust server that serves the compiled frontend and relays UCI traffic
between the browser and engine subprocesses over WebSocket, replacing
`bridge/` (Python) for that purpose. See [#67](https://github.com/BrageHK/bee-chess/issues/67)
for the full architecture this is working toward, and [#68](https://github.com/BrageHK/bee-chess/issues/68)
for this slice's specific scope.

This is a **separate crate** from `engine/` (the competition `bee`
binary) on purpose: `bee` stays runnable standalone with zero web/
orchestration dependencies. `lab/` is where all of that -- process
supervision, HTTP/WebSocket, static file serving, eventually game state
and a model registry -- lives instead.

## What this slice does (67a)

- Serves `frontend/dist/` as static files on one port.
- Spawns a fresh Stockfish or Bee subprocess per WebSocket connection at
  `/ws/stockfish` and `/ws/bee`, relaying raw UCI lines to/from its
  stdin/stdout, unmodified.
- If the engine process dies -- wrong binary, crashes on startup,
  whatever -- before ever sending a UCI reply, the socket is actively
  closed with an `info string engine process exited: <reason>` line
  first, so the browser sees why instead of hanging forever waiting for
  a reply that will never come. This mirrors `bridge/server.py`'s
  `watch_for_exit` exactly.

## What this slice deliberately does *not* do yet

- **No authoritative game state.** The frontend still owns position,
  move list, clocks, and turn-taking, exactly as it does against the
  Python bridge today -- this is a straight relay. That's [#69](https://github.com/BrageHK/bee-chess/issues/69) (67b).
- **No Bee-Mamba.** The Python/PyTorch engine isn't served here; it
  stays on the old bridge for now. Its fate (ported here too, or
  handled entirely differently once [#66](https://github.com/BrageHK/bee-chess/issues/66)'s model-integration
  design lands) is a follow-up decision.
- **No engine/model registry.** Stockfish and Bee's paths are resolved
  directly in `main.rs`; that's [#70](https://github.com/BrageHK/bee-chess/issues/70) (67c).

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

Override the port with `PORT=<n>`. Note that `bee`'s build output lives
at the repo-root `target/release/bee`, not `engine/target/release/bee`
-- `engine/` and `lab/` are both members of the root Cargo workspace
(see `/Cargo.toml`), so Cargo shares one `target/` directory across
every member.

## Testing

```bash
cd lab && cargo test
```

`uci_relay`'s tests spin up a real axum server on an ephemeral port and
connect to it with a real WebSocket client (`tokio-tungstenite`), so
they exercise the actual upgrade path -- including the crash-visibility
behavior (a process that fails to spawn at all, and one that spawns and
exits immediately with a stderr message) -- rather than only unit-testing
`relay` in isolation.
