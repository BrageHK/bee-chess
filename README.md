# bee-chess

This is a crazy Transformer based chess engine with cool stuff.

Bee Chess is one monorepo with three products plus one development tool.
The competition engine is written in Rust, training and dataset
generation is owned by Python, and a React client is used for
development/visualization only. `bridge/` is a small development-only
tool that lets that client talk to engine processes over WebSockets. See
[`docs/adr/0001-v1-engine-architecture.md`](docs/adr/0001-v1-engine-architecture.md)
for the full architecture decision.

## Repository layout

```text
bee-chess/
├── docs/adr/          Architecture decision records
├── engine/            Rust UCI engine (competition hot path)
├── training/          Python training and dataset generation
├── bridge/            Development-only WebSocket <-> UCI bridge
├── frontend/          React development/visualization client
├── scripts/           Repo-level setup/dev/check/test entry points
└── .github/workflows  CI
```

## Getting started

```bash
./scripts/setup.sh  # one-time: creates every subproject's environment
./scripts/dev.sh    # build the engines, start the bridge and the UI
./scripts/check.sh  # everything CI checks, run locally
./scripts/test.sh   # just the test suites
```

Each subproject also works with its own native tooling directly:

```bash
# Engine
cd engine && cargo run --release --bin bee

# Training
cd training && uv sync && uv run pytest

# Bridge (development WebSocket <-> UCI adapter)
cd bridge && uv sync && uv run python server.py

# Frontend
cd frontend && npm install && npm run build
```

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for development rules and branch
naming conventions.

## Engine arena (Stockfish vs Bee)

```bash
git submodule update --init --recursive
./scripts/dev.sh
```

This builds Stockfish and Bee (first run only takes a few minutes for
Stockfish), starts the WebSocket bridge, and opens the frontend dev
server, which plays the two engines against each other.
