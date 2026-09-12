# Catalog regression positions

These versioned fixtures preserve reviewed positions that motivate future Bee
work. They are grouped into `tactical/`, `endgame/`, and `positional/`. Each JSON
file contains the original FEN, full UCI history, Bee's historical move,
Stockfish's candidate and both same-root PVs, measured regret, a manual motif
label, and an expected property to review when changing the engine.

The category is a provisional interpretation of the position and continuations,
not an automated classification or proof of a search/evaluation defect. The
catalog does not identify the historical Bee binary or preserve its search
telemetry. Reproduce with the current Bee build before assigning an engine
cause. Already-lost positions are identified separately.

Use `position startpos moves <history_uci>` when running a candidate engine;
FEN alone loses repetition history. Apply a fixed node budget and record the
engine commit, options, chosen move and PV. Evaluate the `expected_property`;
matching the recorded Stockfish move exactly is not required when an equivalent
move meets that property. Compare with deeper same-root Stockfish searches
before treating a changed engine choice as a confirmed improvement.

Schema version 1:

- `id`, `category`, `fen`, `history_uci`, `bee_move`, `stockfish_candidate`.
- `expected_property`: human-readable behavior sought in a future engine.
- `review`: explanation, confidence, and whether the root was already below
  -200cp. These are review candidates, not claims of an engine fix.
- `source`: game/ply, result, original-game link, run and method version,
  fixed node budget, Stockfish name and canonical configuration.
- `analysis`: both root scores, regret, best PV, and played continuation.

`cargo test -p bee-game-catalog regression_corpus` checks history/FEN agreement,
move/PV legality, provenance and score consistency using `bee-chess-core`.
It validates the fixtures without asserting that today's Bee already solves
these positions. The corresponding catalog report records the complete
30-game review, including uncertain entries that are not corpus fixtures.
