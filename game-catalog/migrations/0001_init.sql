-- Schema version 1.
--
-- Deliberately one row per game, not one row per ply -- see the
-- `game-catalog` crate's docs for why. `moves` holds the game's move
-- sequence (SAN, space-separated, as Lichess's `moves` field gives it) and
-- `pgn` the full PGN when the importer requested it; positions are derived
-- from `moves` during iteration rather than exploded into rows here. That
-- can change later if a real query need shows up for it.
CREATE TABLE games (
    id             TEXT PRIMARY KEY,
    source         TEXT NOT NULL,
    played_at      INTEGER,
    white          TEXT,
    black          TEXT,
    white_rating   INTEGER,
    black_rating   INTEGER,
    result         TEXT,
    termination    TEXT,
    time_control   TEXT,
    rated          INTEGER,
    variant        TEXT,
    moves          TEXT,
    raw_pgn        TEXT,
    imported_at    INTEGER NOT NULL
);

-- The common filters (`GameFilter`) all narrow by these -- see `filter.rs`.
CREATE INDEX games_source_played_at ON games (source, played_at);
CREATE INDEX games_white ON games (white);
CREATE INDEX games_black ON games (black);

-- Incremental sync (`GameCatalog::latest_imported_at`) needs "the newest
-- `played_at` this source/player pair has imported so far" without a full
-- table scan; one row per (source, player) rather than deriving it from
-- `games` by a MAX(played_at) query on every sync, since a player can be
-- either `white` or `black` across their own games and this makes the
-- "what's the watermark for player X" question O(1) instead of an OR-scan.
CREATE TABLE sync_state (
    source         TEXT NOT NULL,
    player         TEXT NOT NULL,
    latest_played_at INTEGER NOT NULL,
    PRIMARY KEY (source, player)
);
