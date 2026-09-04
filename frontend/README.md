# Bee Chess frontend

Development/visualization client. Per
[`docs/adr/0001-v1-engine-architecture.md`](../docs/adr/0001-v1-engine-architecture.md),
this is not part of the competition hot path — it talks to a separate lab
service, never directly to the engine process.

This is currently the Vite + React + TypeScript starter scaffold. The
engine visualization shell (mocked telemetry, board, PV display, etc.)
lands in a follow-up PR (`feat/frontend-shell`).

## Commands

```bash
npm install
npm run lint
npm run build
```
