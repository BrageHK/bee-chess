# bee-chess

[![Lichess Stats](https://lichess-readme-stats.vercel.app/api?username=beechessmagnus)](https://lichess.org/@/beechessmagnus)

This is a crazy Transformer based chess engine with cool stuff.

Bee Chess is one monorepo with three products plus development tooling.
The competition engine is written in Rust, training and dataset
generation is owned by Python, and a React client is used for
development/visualization only. See
[`docs/adr/0001-v1-engine-architecture.md`](docs/adr/0001-v1-engine-architecture.md)
for the full architecture decision.

The frontend talks to `lab/` (Rust -- see #67), which serves the UI,
owns authoritative game state (position, moves, status, and which
engine or human plays each side -- see #69), and supervises Stockfish/
Bee as subprocesses. `bridge/` (Python) was the original, dumber
WebSocket relay this replaced; as of #89, it only serves Bee-Mamba
now -- Stockfish/Bee's relay routes were removed once nothing used
them (`./scripts/dev.sh` starts `lab/`, not `bridge/`). `bridge/` goes
away entirely once Bee-Mamba's real integration (#66/#70) lands
through `lab/` instead. See [`lab/README.md`](lab/README.md).

## Repository layout

```text
bee-chess/
├── docs/adr/          Architecture decision records
├── chess/             Canonical chess-domain crate (Position/Move/legality/FEN/Zobrist), shared by engine/ and lab/
├── engine/            Rust UCI engine (competition hot path)
├── training/          Python training and dataset generation
├── bridge/            Legacy WebSocket <-> UCI bridge (Python), superseded by lab/ -- see #67/#69
├── lab/               Orchestration server (Rust): serves the UI, owns game state, supervises engines
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
./scripts/dev.sh    # build the engines, start Bee Lab and the UI
./scripts/check.sh  # everything CI checks, run locally
./scripts/test.sh   # just the test suites
```

`./scripts/dev.sh` starts Bee Lab on `:8080` by default, and picks the
next free port automatically if something else already has it (Docker
Desktop commonly does) -- no flags needed either way. Set `LAB_PORT`
yourself only to force a specific port:

```bash
LAB_PORT=8081 ./scripts/dev.sh
```

Each subproject also works with its own native tooling directly:

```bash
# Engine
cargo run --release -p bee-engine --bin bee

# Training
cd training && uv sync && uv run pytest

# Lab (orchestration server, Rust -- see lab/README.md)
cargo run -p bee-lab
# or, if :8080 is taken:
PORT=8081 cargo run -p bee-lab

# Frontend (talks to Lab on :8080 by default -- see labClient.ts)
cd frontend && npm install && npm run dev
# or, matching a non-default Lab port:
VITE_LAB_PORT=8081 npm run dev
```

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for development rules and branch
naming conventions.

## Engine arena (Stockfish vs Bee)

```bash
git submodule update --init --recursive
./scripts/dev.sh
```

This builds Stockfish, Bee, and Bee Lab (first run only takes a few
minutes for Stockfish), starts Lab, and opens the frontend dev server --
pick Stockfish vs Bee on the setup screen and Lab plays them against
each other, driving both engine processes itself.
