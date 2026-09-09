//! Builds an `ExperienceBook` artifact from a
//! [`GameCatalog`]: replays a player's own games move by move, records
//! what they played (and how it turned out) at every position where it
//! was their move within the first `max_ply` plies, and reduces that
//! into scored, filtered candidates ready to write.
//!
//! Two data shapes on purpose (see the design this followed): this
//! module's `ExperienceStats`/`PositionExperience`/`MoveExperience` are
//! the rich builder-side aggregate (raw W/D/L counts per move, from the
//! player's perspective regardless of which color they had); `write`ing
//! reduces that down to `format::BookEntry`/`BookCandidate`, the small
//! runtime shape a consuming `ExperienceBook` actually needs. Bee's own
//! `Engine`/`OpeningBook` never sees this module or its formulas at
//! all -- see the crate docs.

use std::collections::HashMap;

use bee_book_format::{book_position_key, BookCandidate, BookEntry};
use bee_chess_core::{Color, Move, Position};

use crate::error::Result;
use crate::filter::GameFilter;
use crate::game::GameRecord;
use crate::GameCatalog;

use super::san;

/// Tunable knobs for one build run -- see `BuildReport` for what a given
/// choice of these actually produced, and this module's docs for why
/// re-running with the same catalog and the same `BuildConfig` must
/// produce byte-identical output.
#[derive(Debug, Clone)]
pub struct BuildConfig {
    /// Only positions within the first `max_ply` plies of a game are
    /// considered -- this is an opening book, not a full-game learner.
    pub max_ply: u32,
    /// A move needs at least this many observed games before it's
    /// eligible to appear in the book at all. Filters out "2/2 = best
    /// move ever" noise from a handful of games rather than trusting
    /// small samples.
    pub min_games: u32,
    /// The shrinkage prior's weight, in "games" -- see
    /// `shrunken_score_per_mille`'s docs. Higher pulls a low-sample
    /// move's score harder toward `prior_score_per_mille`.
    pub prior_games: u32,
    /// The shrinkage prior's score (per mille, i.e. out of 1000 -- 500
    /// is an even/50% prior).
    pub prior_score_per_mille: u32,
}

impl Default for BuildConfig {
    fn default() -> Self {
        Self {
            max_ply: 20,
            min_games: 5,
            prior_games: 10,
            prior_score_per_mille: 500,
        }
    }
}

/// What a build run actually did, for the manifest (`book::manifest`) --
/// separate from the artifact itself, since none of this is needed at
/// runtime to consult the book.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BuildReport {
    pub games_considered: u64,
    pub games_skipped_unresolvable: u64,
    pub positions: u64,
}

/// Raw W/D/L for one candidate move at one position, from the player's
/// own perspective (a draw always counts as a draw regardless of
/// color; win/loss are already normalized to "did the player win").
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
struct MoveExperience {
    games: u32,
    wins: u32,
    draws: u32,
    losses: u32,
}

impl MoveExperience {
    fn record(&mut self, outcome: Outcome) {
        self.games += 1;
        match outcome {
            Outcome::Win => self.wins += 1,
            Outcome::Draw => self.draws += 1,
            Outcome::Loss => self.losses += 1,
        }
    }

    /// A conservative score in per-mille (0..=1000), shrunk toward
    /// `config`'s prior so a tiny sample can't look better than it is:
    ///
    /// ```text
    /// adjusted = (wins + 0.5*draws + prior_games*prior_score) / (games + prior_games)
    /// ```
    ///
    /// All-integer arithmetic (scaled by 1000, then by 2 to keep the
    /// draw's `0.5` exact) so this is reproducible bit-for-bit across
    /// platforms -- see this module's determinism requirement.
    fn shrunken_score_per_mille(&self, config: &BuildConfig) -> u32 {
        // Everything scaled by a common factor of 2 (to keep the draw's
        // 0.5 exact) with a single division at the very end, so nothing
        // truncates before the final result -- see the doc comment above.
        let numerator = 2000 * u64::from(self.wins)
            + 1000 * u64::from(self.draws)
            + 2 * u64::from(config.prior_games) * u64::from(config.prior_score_per_mille);
        let denominator = 2 * u64::from(self.games + config.prior_games);
        (numerator / denominator) as u32
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum Outcome {
    Win,
    Draw,
    Loss,
}

/// Every candidate move ever observed at one position, keyed by the
/// move's packed representation (`Move` is `Copy`+`Eq` but not `Hash`,
/// so this indexes by its `(from, to, flag)` triple instead -- see
/// `MoveKey`).
#[derive(Debug, Clone, Default)]
struct PositionExperience {
    moves: HashMap<MoveKey, (Move, MoveExperience)>,
}

/// A hashable stand-in for `Move`, since `Move` doesn't derive `Hash`
/// (nothing in `chess/` has needed it before now) and this module has
/// no reason to add that to a core, widely-used type just for its own
/// internal `HashMap` key.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct MoveKey(u8, u8, u8);

fn move_key(mv: Move) -> MoveKey {
    MoveKey(mv.from().index(), mv.to().index(), flag_index(mv.flag()))
}

fn flag_index(flag: bee_chess_core::MoveFlag) -> u8 {
    use bee_chess_core::MoveFlag::*;
    match flag {
        Quiet => 0,
        DoublePawnPush => 1,
        EnPassant => 2,
        CastleKingside => 3,
        CastleQueenside => 4,
        PromoteKnight => 5,
        PromoteBishop => 6,
        PromoteRook => 7,
        PromoteQueen => 8,
    }
}

/// The full builder-side aggregate across every game considered.
#[derive(Debug, Clone, Default)]
struct ExperienceStats {
    positions: HashMap<u64, PositionExperience>,
}

impl ExperienceStats {
    fn record(&mut self, key: u64, mv: Move, outcome: Outcome) {
        self.positions
            .entry(key)
            .or_default()
            .moves
            .entry(move_key(mv))
            .or_insert_with(|| (mv, MoveExperience::default()))
            .1
            .record(outcome);
    }
}

/// Builds an `ExperienceBook` artifact (as ready-to-write `BookEntry`s,
/// sorted by key) from `players`' games in `catalog`, treating every
/// name in `players` as the same learning identity -- e.g. Bee's games
/// played under two different Lichess accounts are pooled into one
/// book, rather than needing a separate build (and a separate merge
/// step) per account. Each name is matched the same way
/// `GameFilter::player` matches a single one -- as either side.
///
/// A game gets counted at most once even if, in principle, more than
/// one name in `players` could match it (an account playing itself is
/// not a real scenario this needs to handle correctly, but `players`
/// containing the same name twice, or two names that both happen to
/// match one game, must not double-count that game's outcome).
pub fn build(
    catalog: &GameCatalog,
    players: &[&str],
    config: &BuildConfig,
) -> Result<(Vec<BookEntry>, BuildReport)> {
    let mut stats = ExperienceStats::default();
    let mut games_considered = 0u64;
    let mut games_skipped_unresolvable = 0u64;
    let mut seen_game_ids = std::collections::HashSet::new();

    for &player in players {
        let games = catalog.games(&GameFilter::all().player(player))?;
        for game in &games {
            if !seen_game_ids.insert(game.id.clone()) {
                continue;
            }
            match record_game(&mut stats, game, player, config) {
                Ok(()) => games_considered += 1,
                Err(_) => games_skipped_unresolvable += 1,
            }
        }
    }

    let entries = to_entries(&stats, config);
    let report = BuildReport {
        games_considered,
        games_skipped_unresolvable,
        positions: entries.len() as u64,
    };

    Ok((entries, report))
}

/// Replays one game from the start position, recording every position
/// within `max_ply` where `player` was to move. A game whose move list
/// can't be fully resolved (a SAN token this resolver can't match --
/// see `book::san`) is skipped **in full**, not partially recorded: a
/// resolver failure partway through means every position after it in
/// that game was reached via an unverified move sequence, so trusting
/// them would risk recording experience against the wrong position
/// entirely.
fn record_game(
    stats: &mut ExperienceStats,
    game: &GameRecord,
    player: &str,
    config: &BuildConfig,
) -> std::result::Result<(), san::SanError> {
    let Some(outcome) = player_outcome(game, player) else {
        return Ok(()); // No result recorded (e.g. still "*") -- nothing to learn from.
    };
    let Some(player_color) = player_color(game, player) else {
        return Ok(());
    };

    let mut position = Position::startpos();
    for (ply, token) in game.plies().into_iter().enumerate() {
        if ply as u32 >= config.max_ply {
            break;
        }
        let mv = san::resolve(&position, token)?;
        if position.side_to_move() == player_color {
            let key = book_position_key(&position);
            stats.record(key, mv, outcome);
        }
        position.make_move(mv);
    }

    Ok(())
}

fn player_color(game: &GameRecord, player: &str) -> Option<Color> {
    if game.white.as_deref() == Some(player) {
        Some(Color::White)
    } else if game.black.as_deref() == Some(player) {
        Some(Color::Black)
    } else {
        None
    }
}

fn player_outcome(game: &GameRecord, player: &str) -> Option<Outcome> {
    let color = player_color(game, player)?;
    match game.result.as_deref()? {
        "1-0" => Some(if color == Color::White {
            Outcome::Win
        } else {
            Outcome::Loss
        }),
        "0-1" => Some(if color == Color::Black {
            Outcome::Win
        } else {
            Outcome::Loss
        }),
        "1/2-1/2" => Some(Outcome::Draw),
        _ => None, // "*" or anything else unfinished/unrecognized.
    }
}

/// Reduces the builder-side aggregate to sorted, filtered, scored
/// `BookEntry`s -- see this module's docs on why these are two
/// different shapes. Deterministic: positions sorted by key ascending,
/// candidates within a position sorted by weight descending then by
/// encoded move (a stable tiebreak, since two candidates can tie on
/// weight and a `HashMap`'s iteration order is not itself stable across
/// runs).
fn to_entries(stats: &ExperienceStats, config: &BuildConfig) -> Vec<BookEntry> {
    let mut entries: Vec<BookEntry> = stats
        .positions
        .iter()
        .filter_map(|(&key, position)| {
            let mut candidates: Vec<BookCandidate> = position
                .moves
                .values()
                .filter(|(_, exp)| exp.games >= config.min_games)
                .map(|(mv, exp)| {
                    let score = exp.shrunken_score_per_mille(config);
                    BookCandidate {
                        mv: *mv,
                        weight: score, // v1: weight is just the score -- see module docs.
                        games: exp.games,
                        score_per_mille: score as u16,
                    }
                })
                .collect();

            if candidates.is_empty() {
                return None;
            }

            candidates.sort_by(|a, b| {
                b.weight
                    .cmp(&a.weight)
                    .then_with(|| encoded_sort_key(a.mv).cmp(&encoded_sort_key(b.mv)))
            });

            Some(BookEntry { key, candidates })
        })
        .collect();

    entries.sort_by_key(|entry| entry.key);
    entries
}

/// A deterministic tiebreak key for two candidates with equal weight --
/// arbitrary but stable, unlike hash-map iteration order.
fn encoded_sort_key(mv: Move) -> (u8, u8, u8) {
    (mv.from().index(), mv.to().index(), flag_index(mv.flag()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::GameCatalog;

    fn game(id: &str, white: &str, black: &str, result: &str, moves: &str) -> GameRecord {
        GameRecord {
            id: id.to_string(),
            source: "lichess".to_string(),
            played_at: Some(0),
            white: Some(white.to_string()),
            black: Some(black.to_string()),
            white_rating: None,
            black_rating: None,
            result: Some(result.to_string()),
            termination: None,
            time_control: None,
            rated: Some(true),
            variant: Some("standard".to_string()),
            moves: Some(moves.to_string()),
            raw_pgn: None,
            imported_at: 0,
        }
    }

    #[test]
    fn records_a_win_as_bee_playing_white() {
        let catalog = GameCatalog::open_in_memory().unwrap();
        for i in 0..5 {
            catalog
                .upsert_game(&game(
                    &format!("g{i}"),
                    "Bee",
                    "opp",
                    "1-0",
                    "e4 e5 Nf3 Nc6",
                ))
                .unwrap();
        }

        let (entries, report) = build(&catalog, &["Bee"], &BuildConfig::default()).unwrap();
        assert_eq!(report.games_considered, 5);
        assert_eq!(report.games_skipped_unresolvable, 0);

        let startpos_key = book_position_key(&Position::startpos());
        let entry = entries.iter().find(|e| e.key == startpos_key).unwrap();
        assert_eq!(entry.candidates.len(), 1);
        assert_eq!(entry.candidates[0].games, 5);
        // All wins: score should be well above the 500 prior.
        assert!(entry.candidates[0].score_per_mille > 500);
    }

    #[test]
    fn below_min_games_is_filtered_out_entirely() {
        let catalog = GameCatalog::open_in_memory().unwrap();
        for i in 0..3 {
            catalog
                .upsert_game(&game(&format!("g{i}"), "Bee", "opp", "1-0", "e4 e5"))
                .unwrap();
        }

        let config = BuildConfig {
            min_games: 5,
            ..BuildConfig::default()
        };
        let (entries, _) = build(&catalog, &["Bee"], &config).unwrap();
        assert!(entries.is_empty());
    }

    #[test]
    fn only_positions_where_the_player_is_to_move_are_recorded() {
        let catalog = GameCatalog::open_in_memory().unwrap();
        for i in 0..5 {
            catalog
                .upsert_game(&game(
                    &format!("g{i}"),
                    "opp",
                    "Bee",
                    "0-1",
                    "e4 e5 Nf3 Nc6",
                ))
                .unwrap();
        }

        let (entries, _) = build(&catalog, &["Bee"], &BuildConfig::default()).unwrap();
        // Bee is Black here: it moved after "e4" (playing e5) and after
        // "Nf3" (playing Nc6) -- two distinct positions, not four.
        assert_eq!(entries.len(), 2);
    }

    #[test]
    fn respects_max_ply() {
        let catalog = GameCatalog::open_in_memory().unwrap();
        for i in 0..5 {
            catalog
                .upsert_game(&game(
                    &format!("g{i}"),
                    "Bee",
                    "opp",
                    "1-0",
                    "e4 e5 Nf3 Nc6",
                ))
                .unwrap();
        }

        let config = BuildConfig {
            max_ply: 1,
            min_games: 1,
            ..BuildConfig::default()
        };
        let (entries, _) = build(&catalog, &["Bee"], &config).unwrap();
        // Only ply 0 (the position before "e4") is within max_ply=1.
        assert_eq!(entries.len(), 1);
    }

    #[test]
    fn unresolvable_games_are_skipped_in_full_not_partially_recorded() {
        let catalog = GameCatalog::open_in_memory().unwrap();
        catalog
            .upsert_game(&game("bad", "Bee", "opp", "1-0", "e4 ??? Nf3"))
            .unwrap();
        for i in 0..5 {
            catalog
                .upsert_game(&game(&format!("g{i}"), "Bee", "opp", "1-0", "e4 e5"))
                .unwrap();
        }

        let (_, report) = build(&catalog, &["Bee"], &BuildConfig::default()).unwrap();
        assert_eq!(report.games_considered, 5);
        assert_eq!(report.games_skipped_unresolvable, 1);
    }

    #[test]
    fn building_twice_from_the_same_catalog_is_byte_identical() {
        let catalog = GameCatalog::open_in_memory().unwrap();
        for i in 0..8 {
            let (result, moves) = if i % 2 == 0 {
                ("1-0", "e4 e5 Nf3 Nc6")
            } else {
                ("0-1", "d4 d5 Nf3 Nf6")
            };
            catalog
                .upsert_game(&game(&format!("g{i}"), "Bee", "opp", result, moves))
                .unwrap();
        }

        let config = BuildConfig::default();
        let (entries_a, _) = build(&catalog, &["Bee"], &config).unwrap();
        let (entries_b, _) = build(&catalog, &["Bee"], &config).unwrap();

        let mut bytes_a = Vec::new();
        let mut bytes_b = Vec::new();
        bee_book_format::write(&entries_a, &mut bytes_a).unwrap();
        bee_book_format::write(&entries_b, &mut bytes_b).unwrap();
        assert_eq!(bytes_a, bytes_b);
    }

    #[test]
    fn shrinkage_pulls_a_perfect_small_sample_toward_the_prior() {
        let mut exp = MoveExperience::default();
        exp.record(Outcome::Win);
        exp.record(Outcome::Win);
        let config = BuildConfig::default(); // prior_games=10, prior_score=500

        let score = exp.shrunken_score_per_mille(&config);
        // 2 wins / 2 games is a naive 1000; shrinkage against a
        // 10-game 500 prior must pull it well below that.
        assert!(score < 1000);
        assert!(score > 500);
    }

    #[test]
    fn multiple_player_names_pool_into_one_identity() {
        // Bee's own games played under two different accounts: build
        // with both names should see every game from either account,
        // as if they were one player.
        let catalog = GameCatalog::open_in_memory().unwrap();
        for i in 0..3 {
            catalog
                .upsert_game(&game(
                    &format!("johan{i}"),
                    "BeeJohan",
                    "opp",
                    "1-0",
                    "e4 e5",
                ))
                .unwrap();
        }
        for i in 0..3 {
            catalog
                .upsert_game(&game(
                    &format!("magnus{i}"),
                    "BeeMagnus",
                    "opp",
                    "1-0",
                    "e4 e5",
                ))
                .unwrap();
        }

        let config = BuildConfig {
            min_games: 5,
            ..BuildConfig::default()
        };
        let (entries, report) = build(&catalog, &["BeeJohan", "BeeMagnus"], &config).unwrap();

        assert_eq!(report.games_considered, 6);
        let startpos_key = book_position_key(&Position::startpos());
        let entry = entries.iter().find(|e| e.key == startpos_key).unwrap();
        // Pooled: 6 games total, clearing the min_games=5 threshold
        // that neither account alone (3 games each) would clear.
        assert_eq!(entry.candidates[0].games, 6);
    }

    #[test]
    fn a_game_matching_more_than_one_requested_name_is_not_double_counted() {
        // Contrived (a real account never plays itself), but the
        // contract must hold regardless: the same game id must
        // contribute its outcome at most once.
        let catalog = GameCatalog::open_in_memory().unwrap();
        catalog
            .upsert_game(&game("g1", "Bee", "Bee", "1-0", "e4 e5"))
            .unwrap();

        let (_, report) = build(&catalog, &["Bee", "Bee"], &BuildConfig::default()).unwrap();
        assert_eq!(report.games_considered, 1);
    }
}
