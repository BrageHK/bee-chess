# Development rules

- `main` must always build and pass tests.
- Every behavioral change gets a test where practical.
- Search hot-path code must not perform network I/O.
- UCI strings must not leak below the UCI adapter.
- Python-generated model/data formats must be versioned.
- Performance-sensitive changes should include benchmark results.
- Strength claims require game-testing results, not intuition alone.

See [`docs/adr/0001-v1-engine-architecture.md`](docs/adr/0001-v1-engine-architecture.md)
for the v1 architecture these rules assume.

## Local checks

`./scripts/check.sh` runs everything below in one go (after
`./scripts/setup.sh` once). To run a single subproject's checks
directly:

Rust (`engine/`):

```bash
cargo fmt --check
cargo clippy --all-targets -- -D warnings
cargo test
```

Python (`training/`):

```bash
uv sync
uv run pytest
uv run ruff check .
```

Python (`bridge/`, legacy -- now only serves Bee-Mamba, see #89;
`training/`'s environment is unrelated and much heavier, so this is a
separate `uv`-managed environment on purpose):

```bash
uv sync
uv run ruff check .
```

Frontend (`frontend/`):

```bash
npm install
npm run lint
npm run build
```

Rust, Python (`training/`), and frontend checks run in CI on every pull
request and must pass before merging. `bridge/` has no CI job yet since
it has no behavior to regression-test beyond linting.

## Branch naming

```text
feat/core-legal-moves
feat/uci-state-machine
feat/search-alpha-beta
feat/training-schema
feat/frontend-shell
```

Use `feat/`, `fix/`, `test/`, `chore/`, or `docs/` prefixes as appropriate.

## Merging

- Open a PR against `main`; direct pushes to `main` are not allowed.
- CI must pass.
- Squash merge, then delete the branch.
