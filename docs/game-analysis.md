# Offline Stockfish analysis

`bee-games analyze` reads the local catalog, replays every ply with
`bee-chess-core`, and stores both players' analysis. It requires a local
Stockfish executable with its default NNUE networks available. It makes no
network requests. The engine/search binary has no new dependencies.

```sh
cargo run --release -p bee-games -- analyze \
  --player beechessjohan \
  --stockfish external/stockfish/src/stockfish \
  --nodes 100000 --limit 100 --top 20
```

The catalog defaults to `data/games/catalog.sqlite3`; set `BEE_GAMES_DB` to
use another SQLite file. `--player` is required, matches case-insensitively,
and can be repeated for aliases. A game matching both sides is rejected
because the existing game rollup represents one Bee color; select one identity
when analyzing games between two Bee accounts.

Repeat the exact command to resume. The output explicitly counts analyzed
games, plies, skipped completed games, rejected games, and Stockfish searches.
`--limit` selects the newest matching games **before** checking completion,
so restarting a limited pass does not silently select another batch. Omit it
to cover all matching catalog games.

Each game's entire SAN sequence is checked before any searches. Nonstandard
variants, PGN setup positions, missing moves, and unfinished games are rejected
with a reason. Valid games continue; any rejection makes the command exit
nonzero. Engine failures/timeouts stop the command. Each successful game writes
all move records and its `GameAnalysis` completion marker in a single
transaction. Partial legacy records are replaced when their game completes.
Restarting recomputes an interrupted game but skips completed games. Correcting
a game's moves, variant, player names, or raw PGN invalidates its prior analysis.

## Reproducibility and score conventions

Analysis data version **2** fixes these conventions:

- `go nodes N`, default 100,000, with `Threads=1`, `Hash=16`, `MultiPV=1`,
  full strength, pondering off, and Syzygy probing off. A 60-second deadline
  detects an unresponsive search; it is not a movetime budget.
- `ucinewgame` before **every position** clears hash/search state. The full
  `position startpos moves ...` history preserves repetition and rule clocks.
  An N-ply game needs N+1 searches; adjacent plies share one position result.
- The run stores Stockfish's UCI name, executable SHA-256, node budget,
  normalized Bee identities, fixed options, platform, and analysis version.
  Identical configuration resumes the same run; changed configuration gets
  another run. Use the same binary with its default bundled networks for a
  reproducible pass; arbitrary external/custom NNUE files are not supported.
- `fen_before`, `played_move`, `best_move`, `mover_color`, `is_bee`, and
  `pv` describe the position before each move. Moves and PV use UCI notation;
  `ply` is zero-based. Best move and every PV move are checked for legality.
- Both evaluations are from the **mover's** perspective: `eval_before_cp`
  is the root score and `eval_after_cp` is the next root score negated.
  `centipawn_loss = max(0, before - after)` when both are centipawn scores.
  These are finite-search estimates, so negative differences are clamped to
  zero and neighboring searches can disagree. The score, best move and PV
  come together from the **last exact primary PV**. Stockfish can finish its
  node budget during an unfinished iteration and emit a bound and a different
  `bestmove`; that bounded result is ignored. Upper/lower bounds and secondary
  PVs are never treated as exact scores.
- Mate scores remain separate. `mate_before` / `mate_after` are signed
  **plies** to mate in their respective positions, positive for the mover
  winning and negative for losing. `mate_after = 0` means the move delivered
  checkmate. UCI mate-in-N full moves converts to `2*N-1` plies when winning
  and `2*N` when losing, then the after score is negated. If either evaluation
  is mate, CPL is NULL and the move is excluded from ACPL and CP severity
  counts. The report counts these moves separately.
- Endgame is total phase material <= 8 across both sides (knight/bishop=1,
  rook=2, queen=4; pawns/kings=0). Otherwise plies 0–19 are opening and
  later plies middlegame. Phase uses the position **before** the move.
- Bee's game summaries use exclusive CPL buckets: inaccuracies 50–99,
  mistakes 100–199, blunders >=200. Global/phase ACPL is weighted by the
  number of moves with CP scores, not an average of game averages. The
  separate `>200cp` column is strictly greater than 200.

The database migration preserves existing analysis and marks newly added
provenance/PV fields NULL for legacy rows. Runs remain separate.

## Inspect stored results

Reports read SQLite without starting Stockfish:

```sh
cargo run -p bee-games -- analysis report --run 1 --top 50
cargo run -p bee-games -- analysis report --run 1 --phase middlegame --top 50
```

Each report prints Bee/opponent totals and phase ACPL, large errors, mate
counts, and the highest-CPL Bee moves with FEN and Stockfish PV. Only completed
games from that run are included. Mate transitions can be queried separately:

```sql
-- Moves allowing forced mate or losing a previously forced win.
SELECT game_id, ply, fen_before, played_move, best_move,
       mate_before, mate_after, pv
FROM move_analysis
WHERE analysis_run_id = :run AND is_bee = 1
  AND ((mate_before IS NULL AND mate_after < 0)
       OR (mate_before > 0 AND (mate_after IS NULL OR mate_after < 0)));
```

This is the offline analyzer slice following the analysis storage PR. The Lab dashboard, pattern mining,
and ExperienceBook v2 remain subsequent work. Comparing Bee's own evaluation
with Stockfish also requires importing Bee's search telemetry: catalog game
results and ordinary PGN do not establish what Bee thought during a move.

Stockfish protocol background: [official developer documentation](https://official-stockfish.github.io/docs/stockfish-wiki/Developers.html).

## Validation

```sh
cargo fmt --all --check
cargo clippy -p bee-game-catalog -p bee-games --all-targets -- -D warnings
cargo test -p bee-game-catalog -p bee-games
cargo test -p bee-game-catalog real_stockfish_is_reproducible_and_restart_does_no_searches -- --ignored
```

The last test uses the local Stockfish binary (override with
`BEE_TEST_STOCKFISH`), verifies mate handling, compares independent fresh
passes, and checks that restarting performs zero searches. The normal tests
also reopen a 100-game fixture catalog and assert zero recomputation; fixtures
are synthetic and are not evidence about Bee's playing strength.
