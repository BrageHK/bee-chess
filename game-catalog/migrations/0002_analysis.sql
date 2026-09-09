-- Schema version 2: post-game move/game analysis (Stockfish-backed, but
-- this schema itself knows nothing about Stockfish specifically -- see
-- `analysis_runs.engine`). PR-scoped deliberately narrow: this only adds
-- the persistence layer (tables + GameCatalog methods to read/write
-- them); nothing in this crate actually runs an analyzer yet. A later
-- PR's `bee-games analyze` populates these tables incrementally.
--
-- One `analysis_runs` row per distinct analysis configuration that has
-- ever been run (engine, node budget, MultiPV, schema version) --
-- `move_analysis`/`game_analysis` rows are always scoped to the run that
-- produced them, so results from different configurations (or a
-- different Stockfish version) never get silently mixed together, and
-- re-running the same config just adds rows under the same
-- `analysis_run_id` (or a caller can create a fresh run to keep old
-- results around for comparison -- see `record_analysis_run`'s docs).
CREATE TABLE analysis_runs (
    id                   INTEGER PRIMARY KEY AUTOINCREMENT,
    engine               TEXT NOT NULL,
    nodes_per_position   INTEGER,
    multipv              INTEGER,
    schema_version       INTEGER NOT NULL,
    created_at           INTEGER NOT NULL
);

-- One row per ply Bee (or, once an all-players analysis policy exists,
-- either side) played in a game, scoped to the `analysis_runs` row that
-- produced it. `fen_before` is stored directly (not re-derived from
-- `games.moves` on every read) since a move's analysis is meaningless
-- without the exact position it was played from, and re-deriving it via
-- SAN replay on every query would be needless recomputation for a table
-- that exists specifically to be queried repeatedly (see the design's
-- "problem miner" queries). `played_move`/`best_move` are UCI strings
-- (e.g. "e2e4"), not `chess::Move`'s packed bits -- unlike the `.book`
-- binary format, this table is meant to be inspected directly (SQL
-- queries, a future Lab UI), so a human-readable move notation is worth
-- more here than the packed representation's compactness.
CREATE TABLE move_analysis (
    id                   INTEGER PRIMARY KEY AUTOINCREMENT,
    analysis_run_id      INTEGER NOT NULL REFERENCES analysis_runs (id),
    game_id              TEXT NOT NULL REFERENCES games (id),
    ply                  INTEGER NOT NULL,
    fen_before           TEXT NOT NULL,
    played_move          TEXT NOT NULL,
    best_move            TEXT,
    eval_before_cp       INTEGER,
    eval_after_cp        INTEGER,
    centipawn_loss       INTEGER,
    mate_before          INTEGER,
    mate_after           INTEGER,
    phase                TEXT NOT NULL
);

-- The "problem miner" queries (worst moves, filter by phase/loss
-- threshold) all narrow by these.
CREATE INDEX move_analysis_run_game ON move_analysis (analysis_run_id, game_id);
CREATE INDEX move_analysis_run_phase_loss ON move_analysis (analysis_run_id, phase, centipawn_loss);

-- One row per (game, analysis run): the game-level rollup a Lab
-- dashboard actually wants first ("avg CPL", "how many blunders") --
-- computed once by the analyzer and stored, rather than aggregated from
-- `move_analysis` on every dashboard load.
CREATE TABLE game_analysis (
    analysis_run_id       INTEGER NOT NULL REFERENCES analysis_runs (id),
    game_id               TEXT NOT NULL REFERENCES games (id),
    bee_color             TEXT NOT NULL,
    avg_centipawn_loss    REAL,
    worst_move_cp_loss    INTEGER,
    inaccuracies          INTEGER NOT NULL,
    mistakes              INTEGER NOT NULL,
    blunders              INTEGER NOT NULL,
    opening_avg_loss      REAL,
    middlegame_avg_loss   REAL,
    endgame_avg_loss      REAL,
    PRIMARY KEY (analysis_run_id, game_id)
);
