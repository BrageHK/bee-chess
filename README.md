# bee-chess

This is a crazy Transformer based chess engine with cool stuff.

Bee Chess is a standard-chess UCI engine. The competition engine is
written in Rust, training and dataset generation is owned by Python, and
a React client is used for development/visualization only. See
[`docs/adr/0001-v1-engine-architecture.md`](docs/adr/0001-v1-engine-architecture.md)
for the full architecture decision.

## Repository layout

```text
bee-chess/
├── docs/adr/        Architecture decision records
├── engine/           Rust UCI engine (competition hot path)
├── training/         Python training and dataset generation
├── frontend/         React development/visualization client
└── .github/workflows CI
```

## Getting started

```bash
# Engine
cd engine && cargo run --release --bin bee

# Training
cd training && uv sync && uv run pytest

# Frontend
cd frontend && npm install && npm run build
```

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for development rules and branch
naming conventions.

## Engine arena (Stockfish vs Bee)

```bash
git submodule update --init --recursive
./scripts/dev.sh
./scripts/build-stockfish.sh
./scripts/build-bee.sh
uv run --with websockets python bridge/server.py
npm --prefix frontend run dev
```
