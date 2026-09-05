# bee-chess

This is a crazy Transformer based chess engine with cool stuff.

Bee Chess is one monorepo with three products plus development tooling.
The competition engine is written in Rust, training and dataset
generation is owned by Python, and a React client is used for
development/visualization only. See
[`docs/adr/0001-v1-engine-architecture.md`](docs/adr/0001-v1-engine-architecture.md)
for the full architecture decision.

Two things currently connect the frontend to engine processes over
WebSockets: `bridge/` (Python, the original, still what `./scripts/dev.sh`
uses) and `lab/` (Rust, newer -- see #67/#68). `lab/` is being built out
to eventually replace `bridge/` and become authoritative for game state
too, not just a dumb relay; until that migration is further along, both
exist and either works standalone. See [`lab/README.md`](lab/README.md).

## Repository layout

```text
bee-chess/
├── docs/adr/          Architecture decision records
├── chess/             Canonical chess-domain crate (Position/Move/legality/FEN/Zobrist), shared by engine/ and lab/
├── engine/            Rust UCI engine (competition hot path)
├── training/          Python training and dataset generation
├── bridge/            Development-only WebSocket <-> UCI bridge (Python)
├── lab/               Development/orchestration server (Rust), replacing bridge/ -- see #67
├── frontend/          React development/visualization client
├── scripts/           Repo-level setup/dev/check/test entry points
└── .github/workflows  CI
```

`engine/` and `lab/` both depend on `chess/` (`bee-chess-core`) for chess
rules, rather than each having their own -- see `chess/src/lib.rs`'s
docs for why that matters once `lab/` starts validating moves
server-side (#69).

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

# Bridge (development WebSocket <-> UCI adapter, Python)
cd bridge && uv sync && uv run python server.py

# Lab (development/orchestration server, Rust -- see lab/README.md)
cargo run -p bee-lab

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
