# Regression diagnostic runner

`bee-games regressions diagnose` compares current Bee searches with reviewed
catalog failures. It is offline, reads fixtures directly, and does not open the
catalog DB. No engine evaluation weights or search behavior are changed.

```sh
cargo build --release -p bee-engine -p bee-games
target/release/bee-games regressions diagnose \
  --output data/games/regression-diagnostics-v1
# Identical restart: completed searches and recorded timeouts are reused.
target/release/bee-games regressions diagnose \
  --output data/games/regression-diagnostics-v1
```

Defaults:

| Setting | Value |
|---|---|
| Bee | `target/release/bee` (`--bee`) |
| Stockfish | `external/stockfish/src/stockfish` (`--stockfish`) |
| Corpus | `regressions` (`--fixtures`) |
| Depths | `4,6,8,10,12` (`--depths`, strictly increasing) |
| Variants | `baseline,no-lmr,no-null-move,no-see,no-king-safety` (`--variants`) |
| Stockfish budget | 1,000,000 nodes per root search (`--reference-nodes`) |
| Acceptance tolerance | 50cp inclusive (`--acceptable-cp`) |
| Bee deadline | 30 seconds per search (`--timeout-seconds`) |
| Concurrent fixtures | 4 (`--jobs`) |

Bee's current UCI adapter **does not support `go nodes`**; it silently ignores
that token. These runs therefore use `go depth`, recording actual nodes and
completed depth. Each search starts a fresh Bee process, disables opening books
and tablebases, explicitly sets all search/evaluation feature switches, and
replays the full `position startpos moves ...` history. `bee-chess-core` validates
the fixture FEN, historical moves, and every returned PV. A FEN alone cannot
preserve repetition history.

These are cold, direct fixed-depth searches. The competition engine normally
uses iterative deepening with time management and may retain a game TT. A
different result here is evidence about current Bee under the recorded
conditions, not a reproduction of the unidentified historical binary or clock.
Depth ablations also consume different node counts; compare the recorded nodes
before attributing a change to selectivity rather than additional work.

The existing options are changed one at a time: `UseLMR=false`,
`UseNullMove=false`, `UseSee=false`, or `UseKingSafety=false`. The latter is an
ablation, not a proposed replacement evaluator. To compare future modified
weights, build a separate Bee executable and use `--bee` with a new output
directory and otherwise identical settings. Binaries must retain Bee's UCI
option and score contracts.

## Acceptance and score conventions

For each fixture the runner searches Stockfish normally, then restricts another
search **from the same root and full history** to each distinct Bee choice. It
also checks the historical Bee move and recorded Stockfish candidate against
this new reference. Both scores are from the mover's perspective; no post-move
search or score negation enters regret. The existing analyzer's Stockfish
adapter resets between searches and uses one thread, 16 MiB hash, MultiPV 1,
full strength, no tablebases, and the last exact primary PV.

Matching the new reference move has zero regret without a second search.
Other CP-valued moves are provisionally acceptable when
`max(0, reference_cp - restricted_cp) <= acceptable_cp`. This admits equivalent
alternatives to the recorded candidate. If a restricted search scores more
than the tolerance **above** the normal root score, it is marked
`reference_inconsistent` and left unassessed: the reference needs more work.
Node-limited Stockfish is an acceptance proxy; a human must still check the
fixture's `expected_property` before claiming a feature fixed the position.

Mate values remain separate from centipawns. A move preserving a winning mate
is acceptable; losing a winning mate or entering a forced loss is unacceptable.
Two different forced-loss mate distances are unassessed. A discovered escape
from a reference forced loss is inconsistent with that reference and needs
review. All stored mate distances are **signed plies**: Bee already reports
plies in its UCI mate field, whereas Stockfish's full-move distances are
converted by the existing adapter. Exact best-move matches always have zero
regret, including mate-valued matches.

## Results, resume, and interpretation

The output directory contains a versioned `manifest.json`, one checkpoint JSON
per fixture, and `summary.json`. The manifest records both executable SHA-256s,
all fixture hashes, platform, effective options, budgets, acceptance semantics,
and concurrency. Changing any recorded configuration requires a new output
directory. Record the Bee source commit alongside reports when building it;
the runner identifies the executable by its content, not by guessing which
commit produced an existing file.

Each fixture checkpoint contains the complete input/provenance, the fresh
Stockfish reference, cached restricted-move judgments/PVs, and every Bee sample:
variant, requested depth, status, best move, mover score, actual nodes/depth,
elapsed time, PV, and LMR/null-move/SEE/delta counters. Checkpoints are written
atomically after each search. A process lock prevents simultaneous writes to
one run; it is released on exit, including crashes.

Restarting skips all checkpointed samples and judgments. A search interrupted
before its checkpoint can be repeated. A deadline is stored as `timed_out`
with no accepted move, score, or invented node count; it is not a completed
failure. Increase the deadline in a **new** output directory for deeper coverage.
Startup failures, malformed protocol, illegal PVs, and wrong completed depths
are errors, not timeouts. Valid checkpoints survive an error.

The summary reports acceptance/completion/unassessed/timeout counts by bucket,
category, variant, and depth. Each position includes the first acceptable depth,
all acceptable depths, deepest completed result, search recoveries, later
regressions, and same-depth toggle recoveries with both node counts. First
success does not imply stable success at larger budgets.

Historical prioritization is explicit and remains comparable to the catalog:

- `preventable`: stored pre-move score >= −200cp (the corpus already requires
  historical regret >200cp).
- `already_losing`: stored pre-move score < −200cp.
- State labels: winning >+200cp, losing <−200cp, equal otherwise. The report
  places equal→losing, winning→equal, and winning→losing before other transitions
  within the preventable bucket.

`reference_bucket` and `reference_state_transition` show how the deeper reference
changes this assessment; they are null for mate-valued comparisons. Review
these before prioritizing engine work. Distinct FEN count is shown alongside
fixture count; repeated boards with different histories are not independent
motifs.

Interpret completed budget/toggle recoveries as search evidence. Persistent
mismatches are candidates for deeper search, evaluation inspection, or both;
timeouts and inconsistent references leave the cause unclear. The runner does
not automatically label chess motifs or claim an evaluation defect.

```sh
# First acceptable depth and deepest completed result for every configuration:
jq '.positions[] | {fixture, bucket, reference_bucket, state_transition, variants}' \
  data/games/regression-diagnostics-v1/summary.json
# Review one position, including Bee and Stockfish PVs:
jq . data/games/regression-diagnostics-v1/HDjhNtQm-35.json
```

The 13 king-safety fixtures have provisional manual `review.subcategory`
annotations: 11 ignored attacks, one failure to evacuate, and one king move into
danger. The existing review notes explain the visible continuations. No case is
labeled a missed forced mate without evidence of one; these labels do not
distinguish search from evaluation by themselves.
