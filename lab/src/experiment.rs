//! A/B experiments: run N paired games between two engine
//! configurations ("variant A" vs "variant B") and tally the result --
//! the actual point of the whole UCI-option-discovery milestone (#104,
//! #105, #106): change one `setoption` on Bee, run an experiment, find
//! out whether it actually won more games.
//!
//! v1 is deliberately Bee-vs-Bee only (both variants use the same
//! `EngineSpec`, just different `options`) -- see the design-system
//! milestone's discussion for why: Stockfish's Elo/`UCI_LimitStrength`
//! semantics and asymmetric engines add real complexity that has
//! nothing to do with "did this Bee change help." Nothing here
//! actually enforces that (an `EngineVariant` carries its own full
//! `EngineConfig`, spec included), so lifting the restriction later --
//! if there's ever a real reason to A/B two different engines -- needs
//! no redesign, only a decision to allow it in the API layer.
//!
//! Games run through the exact same `GameStore`/`run_engine_loop` a
//! normal `POST /api/games` game does -- an experiment is an
//! orchestrator over ordinary Lab games, not a second game engine.
//! That's also why an experiment's games are inspectable at their own
//! ordinary `/api/games/:id` and `/ws/games/:id` -- see
//! `ExperimentSnapshot::games`.
//!
//! Games run sequentially, not concurrently: this keeps the
//! implementation (and its state machine) simple and correct for v1,
//! at the cost of an experiment taking roughly `requested_games *
//! move_time_ms * average_game_length` wall-clock time. Worth
//! revisiting once that's actually felt to be too slow -- not before.
//!
//! Games are paired by color and opening: game `2k` has variant A playing
//! White, game `2k+1` has variant A playing Black, and both begin from the
//! same short opening line. Successive pairs rotate through a small built-in
//! suite so an experiment does not merely replay two start-position games.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use chrono::{DateTime, Utc};
use serde::Serialize;
use uuid::Uuid;

/// Short, deliberately unsurprising opening lines in UCI notation. Every
/// line has an even number of plies, leaving White to move in each seeded
/// position and making both games in a color-swapped pair directly comparable.
const OPENING_SUITE: &[&[&str]] = &[
    &[],
    &["e2e4", "e7e5", "g1f3", "b8c6"],
    &["d2d4", "d7d5", "c2c4", "e7e6"],
    &["c2c4", "e7e5", "b1c3", "g8f6"],
    &["g1f3", "d7d5", "g2g3", "g8f6"],
    &["e2e4", "c7c5", "g1f3", "d7d6"],
    &["e2e4", "e7e6", "d2d4", "d7d5"],
    &["d2d4", "g8f6", "c2c4", "g7g6"],
    &["e2e4", "c7c6", "d2d4", "d7d5"],
];

fn opening_moves_for_game(game_index: u32) -> &'static [&'static str] {
    let pair_index = game_index as usize / 2;
    OPENING_SUITE[pair_index % OPENING_SUITE.len()]
}

use crate::game::{EngineConfig, EngineSlots, GameId, GameResult, GameStatus, GameStore};

/// The git commit `bee-lab` itself was built from -- embedded at
/// compile time by `build.rs` (`"unknown"` if that couldn't determine
/// one, e.g. a packaged binary with no `.git` around). Recorded on
/// every `ExperimentMetadata` so a result can always be traced back
/// to exactly which build of Lab (and, since this workspace builds
/// them together, `bee`) produced it -- see that type's docs.
pub const LAB_GIT_COMMIT: &str = env!("BEE_LAB_GIT_COMMIT");

/// Opaque experiment identifier, serialized as a plain string over the
/// API -- same shape as `GameId`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize)]
#[serde(transparent)]
pub struct ExperimentId(Uuid);

impl ExperimentId {
    fn new() -> Self {
        ExperimentId(Uuid::new_v4())
    }
}

impl std::fmt::Display for ExperimentId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl std::str::FromStr for ExperimentId {
    type Err = uuid::Error;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Ok(ExperimentId(Uuid::parse_str(s)?))
    }
}

/// One side of an experiment: a human-readable label ("Baseline",
/// "PassedPawns") plus the full engine configuration that label means.
#[derive(Debug, Clone)]
pub struct EngineVariant {
    pub label: String,
    pub config: EngineConfig,
}

/// One experiment's fixed setup, given at creation and never mutated
/// afterward -- see `Experiment` for the mutable progress tracked
/// alongside it.
#[derive(Debug, Clone)]
pub struct ExperimentSpec {
    pub variant_a: EngineVariant,
    pub variant_b: EngineVariant,
    pub requested_games: u32,
    pub move_time_ms: u64,
}

/// Enough about how/when an experiment ran to make its numbers
/// interpretable again later -- a score by itself doesn't say which
/// build of Bee produced it, when, or with what exact `argv`. Recorded
/// once at creation (`lab_git_commit`, `variant_a_argv`/
/// `variant_b_argv`, `started_at`) plus once more when the experiment
/// finishes (`finished_at`) -- see `Experiment::finish`.
///
/// The opening policy is the fixed built-in `OPENING_SUITE`; each game's
/// ordinary move list records the exact line it received.
#[derive(Debug, Clone, Serialize)]
pub struct ExperimentMetadata {
    pub lab_git_commit: String,
    /// The exact `argv` each variant's engine process was spawned
    /// with -- e.g. `["/path/to/bee"]` -- not just its label, since
    /// two variants can (and in the intended usage, do) share the
    /// same binary and differ only in `setoption`s (see `EngineConfig`).
    /// The `setoption`s themselves are already visible on the
    /// snapshot's `label_a`/`label_b` config the frontend rendered
    /// them from, so aren't duplicated here.
    pub variant_a_argv: Vec<String>,
    pub variant_b_argv: Vec<String>,
    pub started_at: DateTime<Utc>,
    /// `None` while the experiment is still running -- see
    /// `Experiment::finish`.
    pub finished_at: Option<DateTime<Utc>>,
}

impl ExperimentMetadata {
    fn new(spec: &ExperimentSpec) -> Self {
        ExperimentMetadata {
            lab_git_commit: LAB_GIT_COMMIT.to_string(),
            variant_a_argv: spec.variant_a.config.spec.argv.clone(),
            variant_b_argv: spec.variant_b.config.spec.argv.clone(),
            started_at: Utc::now(),
            finished_at: None,
        }
    }
}

/// One game this experiment ran or is running, in the order it was
/// started -- enough for a client to link into that game's own
/// ordinary `/api/games/:id` (and live `/ws/games/:id`) view, plus
/// know which variant played which color without needing to
/// cross-reference anything else.
#[derive(Debug, Clone, Serialize)]
pub struct ExperimentGame {
    pub game_id: GameId,
    /// `true` if variant A played White in this game (`false` means B
    /// did) -- see the module docs' pairing scheme.
    pub variant_a_is_white: bool,
    pub outcome: GameOutcome,
    pub started_at: DateTime<Utc>,
    /// `None` while the game is still running -- set alongside
    /// `outcome` once `run_experiment` sees it settle. Together with
    /// `started_at`, this is what `ExperimentStats` derives average
    /// game duration from, rather than re-deriving it from
    /// `ExperimentMetadata`'s experiment-wide `started_at`/
    /// `finished_at` (which span every game plus the gaps between
    /// them, not any one game's own duration).
    pub finished_at: Option<DateTime<Utc>>,
    /// Plies played, i.e. `GameSnapshot::moves.len()` at the point the
    /// game settled -- `None` for a game that's still running or that
    /// aborted before recording a final snapshot was possible. Stored
    /// here rather than re-fetched from `GameStore` at stats time so
    /// `ExperimentStats` doesn't depend on the underlying game still
    /// existing in memory (there's no persistence yet -- see
    /// `GameStore`'s docs).
    pub plies: Option<usize>,
}

/// How one experiment game ended, if it has -- deliberately distinct
/// from a bare `Option<GameResult>`: `Aborted` (an engine crashed,
/// played an illegal move, etc.) is *also* a game that's done running,
/// just one with no chess result to tally, and `Experiment::status`
/// needs to tell "still running" and "done, no result" apart to ever
/// report `Completed` for an experiment where a game aborted (see
/// `run_experiment`'s docs on why an aborted game isn't retried).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum GameOutcome {
    Pending,
    Finished { result: GameResult },
    Aborted,
}

impl GameOutcome {
    const fn is_settled(self) -> bool {
        !matches!(self, GameOutcome::Pending)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ExperimentStatus {
    Running,
    Completed,
}

/// One A/B experiment's full state: its fixed spec plus every game it
/// has run or is running, in order. `ExperimentStore` holds these
/// behind a mutex the same way `GameStore` holds `Game`s -- see that
/// type's docs for why a plain mutex is the right amount of
/// infrastructure here.
#[derive(Debug, Clone)]
pub struct Experiment {
    pub id: ExperimentId,
    pub spec: ExperimentSpec,
    pub games: Vec<ExperimentGame>,
    pub metadata: ExperimentMetadata,
    /// Monotonically increasing creation order, used only by
    /// `ExperimentStore::list` to sort newest-first -- an
    /// `ExperimentId` is a random `Uuid::new_v4`, so it carries no
    /// creation-order information of its own to sort by instead. Same
    /// reasoning as `game::Game::created_seq`.
    created_seq: u64,
}

/// Source for `Experiment::created_seq` -- see `game::next_game_seq`'s
/// docs; same reasoning, a separate counter since experiments and
/// games are ordered independently of each other.
static NEXT_EXPERIMENT_SEQ: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

impl Experiment {
    fn new(id: ExperimentId, spec: ExperimentSpec) -> Self {
        let metadata = ExperimentMetadata::new(&spec);
        Experiment {
            id,
            spec,
            games: Vec::new(),
            metadata,
            created_seq: NEXT_EXPERIMENT_SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed),
        }
    }

    /// `Completed` once every requested game has *ended* -- a real
    /// chess result or an abort, either one settles a game (see
    /// `GameOutcome::is_settled`) -- not once every game has a
    /// `GameResult` specifically; an experiment where a game aborted
    /// must still be able to report `Completed` once there's nothing
    /// left to run, rather than reporting `Running` forever.
    fn status(&self) -> ExperimentStatus {
        if self.games.len() as u32 >= self.spec.requested_games
            && self.games.iter().all(|g| g.outcome.is_settled())
        {
            ExperimentStatus::Completed
        } else {
            ExperimentStatus::Running
        }
    }

    /// Tallies completed games from variant A's perspective: wins,
    /// draws, and losses (i.e. variant B's wins), regardless of which
    /// color A played in any given game -- see `ExperimentGame::
    /// variant_a_is_white`. An aborted game contributes to none of the
    /// three (see `run_experiment`'s docs); `wins_a + draws + wins_b`
    /// therefore only ever equals the number of games that reached a
    /// real chess result, which may be less than `requested_games`.
    fn tally(&self) -> (u32, u32, u32) {
        let mut wins_a = 0;
        let mut draws = 0;
        let mut wins_b = 0;
        for game in &self.games {
            let GameOutcome::Finished { result } = game.outcome else {
                continue;
            };
            let a_won = match (result, game.variant_a_is_white) {
                (GameResult::WhiteWins, true) | (GameResult::BlackWins, false) => Some(true),
                (GameResult::WhiteWins, false) | (GameResult::BlackWins, true) => Some(false),
                (GameResult::Draw, _) => None,
            };
            match a_won {
                Some(true) => wins_a += 1,
                Some(false) => wins_b += 1,
                None => draws += 1,
            }
        }
        (wins_a, draws, wins_b)
    }
}

/// A complete, self-sufficient snapshot of one experiment -- the shape
/// `GET /api/experiments/:id` returns. Same "authoritative resync"
/// philosophy as `GameSnapshot`: a client renders this directly, with
/// no need to have tracked anything about the experiment itself.
#[derive(Debug, Clone, Serialize)]
pub struct ExperimentSnapshot {
    pub id: ExperimentId,
    pub status: ExperimentStatus,
    pub label_a: String,
    pub label_b: String,
    pub requested_games: u32,
    pub completed_games: u32,
    pub wins_a: u32,
    pub draws: u32,
    pub wins_b: u32,
    /// Variant A's score as a fraction of completed games (a win = 1,
    /// a draw = 0.5, a loss = 0), the standard chess-engine-testing
    /// scoring convention -- `None` until at least one game has
    /// finished, rather than a misleading `0.0`.
    pub score_a: Option<f64>,
    /// A's estimated Elo advantage over B, derived from `score_a` via
    /// the standard chess-engine-testing formula (see
    /// `elo_diff_from_score`) -- `None` whenever `score_a` is (no
    /// games settled yet), plus at the two scores that formula can't
    /// produce a finite number for (a perfect 0% or 100% score: see
    /// that function's docs). This is a point estimate only -- no
    /// confidence interval yet (a later PR in this same
    /// strength-measurement sequence), so read a small-sample number
    /// here with real skepticism, the same way you would a "51% score
    /// after 3 games."
    pub elo_diff_a: Option<f64>,
    pub games: Vec<ExperimentGame>,
    pub metadata: ExperimentMetadata,
    pub stats: ExperimentStats,
}

/// The standard chess-engine-testing Elo-difference-from-score
/// formula: `-400 * log10(1/score - 1)`. `score` is variant A's score
/// as a fraction of games played (0.0 = A lost every game, 1.0 = A
/// won every game, 0.5 = even) -- see `score_a`'s own docs.
///
/// Returns `None` at the two scores this formula is undefined for:
/// `0.0` (would need `-infinity`, i.e. "A is infinitely weaker,"
/// which isn't a real Elo number) and `1.0` (`+infinity`, same
/// problem in the other direction). A small number of games producing
/// a perfect score is exactly the situation where "we can't yet put a
/// number on how much stronger" is the honest answer, not "here's a
/// gigantic Elo estimate" -- the same `None`-not-a-misleading-number
/// stance `score_a`/`ExperimentStats`'s averages already take.
fn elo_diff_from_score(score: f64) -> Option<f64> {
    if score <= 0.0 || score >= 1.0 {
        return None;
    }
    Some(-400.0 * ((1.0 / score) - 1.0).log10())
}

/// Summary numbers a human skimming an experiment actually wants,
/// beyond the raw win/draw/loss tally: how long games take, how far
/// they go, and how fast the whole run is moving -- the first slice
/// of the strength-measurement milestone (before an estimated Elo
/// delta, confidence interval, or SPRT, which build on this same
/// `ExperimentSnapshot` in later PRs).
///
/// Every average here is `None` rather than `0.0`/`NaN` when there's
/// nothing to average yet (no game has settled), for the same reason
/// `score_a` is `Option` -- a `0.0` would misleadingly claim "instant
/// games," not "no data yet."
#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct ExperimentStats {
    /// Mean wall-clock duration of a settled game (one with both
    /// `started_at` and `finished_at` -- i.e. `Finished` or `Aborted`,
    /// not `Pending`), in milliseconds. Aborted games are included:
    /// an abort still took real wall-clock time, even though it
    /// contributes no chess result to `wins_a`/`draws`/`wins_b`.
    pub avg_game_duration_ms: Option<f64>,
    /// Mean plies played per settled game with a known ply count --
    /// see `ExperimentGame::plies`'s docs on why an aborted game may
    /// have `None` here instead of a real count.
    pub avg_plies: Option<f64>,
    /// Wall-clock time since the experiment started: `finished_at -
    /// started_at` once complete, or `now - started_at` while still
    /// running -- a `None` here (rather than "N/A" for a running
    /// experiment) would be a strictly less useful answer to "how
    /// long has this been going," so this is never `None`.
    pub runtime_ms: i64,
    /// Settled games (see `avg_game_duration_ms`) per hour of
    /// `runtime_ms` so far -- `None` if no game has settled yet or
    /// `runtime_ms` rounds to zero (can't meaningfully divide by it).
    pub games_per_hour: Option<f64>,
}

impl ExperimentStats {
    fn compute(experiment: &Experiment) -> Self {
        let settled: Vec<&ExperimentGame> = experiment
            .games
            .iter()
            .filter(|g| g.outcome.is_settled())
            .collect();

        let durations_ms: Vec<i64> = settled
            .iter()
            .filter_map(|g| {
                g.finished_at
                    .map(|finished| (finished - g.started_at).num_milliseconds())
            })
            .collect();
        let avg_game_duration_ms = mean(&durations_ms);

        let plies: Vec<usize> = settled.iter().filter_map(|g| g.plies).collect();
        let avg_plies = mean(&plies.iter().map(|&p| p as i64).collect::<Vec<_>>());

        let runtime_ms = (experiment.metadata.finished_at.unwrap_or_else(Utc::now)
            - experiment.metadata.started_at)
            .num_milliseconds();

        let games_per_hour = if settled.is_empty() || runtime_ms <= 0 {
            None
        } else {
            let hours = runtime_ms as f64 / 3_600_000.0;
            Some(settled.len() as f64 / hours)
        };

        ExperimentStats {
            avg_game_duration_ms,
            avg_plies,
            runtime_ms,
            games_per_hour,
        }
    }
}

/// Arithmetic mean of `values`, or `None` for an empty slice -- shared
/// by `ExperimentStats::compute`'s two averages rather than each
/// hand-rolling "empty means None, else sum/len."
fn mean(values: &[i64]) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    Some(values.iter().sum::<i64>() as f64 / values.len() as f64)
}

impl From<&Experiment> for ExperimentSnapshot {
    fn from(experiment: &Experiment) -> Self {
        let (wins_a, draws, wins_b) = experiment.tally();
        let completed_games = wins_a + draws + wins_b;
        let score_a = if completed_games == 0 {
            None
        } else {
            Some((f64::from(wins_a) + 0.5 * f64::from(draws)) / f64::from(completed_games))
        };
        let elo_diff_a = score_a.and_then(elo_diff_from_score);
        ExperimentSnapshot {
            id: experiment.id,
            status: experiment.status(),
            label_a: experiment.spec.variant_a.label.clone(),
            label_b: experiment.spec.variant_b.label.clone(),
            requested_games: experiment.spec.requested_games,
            completed_games,
            wins_a,
            draws,
            wins_b,
            score_a,
            elo_diff_a,
            games: experiment.games.clone(),
            metadata: experiment.metadata.clone(),
            stats: ExperimentStats::compute(experiment),
        }
    }
}

/// In-memory store of every experiment the server currently knows
/// about -- same shape/reasoning as `GameStore`.
#[derive(Clone, Default)]
pub struct ExperimentStore {
    experiments: Arc<Mutex<HashMap<ExperimentId, Experiment>>>,
}

impl ExperimentStore {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn create(&self, spec: ExperimentSpec) -> ExperimentSnapshot {
        let id = ExperimentId::new();
        let experiment = Experiment::new(id, spec);
        let snapshot = ExperimentSnapshot::from(&experiment);
        self.experiments
            .lock()
            .expect("experiment store mutex poisoned")
            .insert(id, experiment);
        snapshot
    }

    /// Every experiment the server currently knows about, newest
    /// first -- the shape `GET /api/experiments` returns for the
    /// dashboard's running/past experiment lists. Same point-in-time,
    /// no-persistence caveat as `GameStore::list`.
    pub fn list(&self) -> Vec<ExperimentSnapshot> {
        let experiments = self
            .experiments
            .lock()
            .expect("experiment store mutex poisoned");
        let mut ordered: Vec<&Experiment> = experiments.values().collect();
        ordered.sort_by_key(|experiment| std::cmp::Reverse(experiment.created_seq));
        ordered
            .iter()
            .map(|experiment| ExperimentSnapshot::from(*experiment))
            .collect()
    }

    pub fn snapshot(&self, id: ExperimentId) -> Option<ExperimentSnapshot> {
        self.experiments
            .lock()
            .expect("experiment store mutex poisoned")
            .get(&id)
            .map(ExperimentSnapshot::from)
    }

    /// Records a newly-started game against `id`'s experiment, with no
    /// result yet -- called once per game, right after `GameStore::
    /// create` returns its id, before that game's `run_engine_loop`
    /// starts (see `run_experiment`).
    fn record_game_started(&self, id: ExperimentId, game_id: GameId, variant_a_is_white: bool) {
        let mut experiments = self
            .experiments
            .lock()
            .expect("experiment store mutex poisoned");
        if let Some(experiment) = experiments.get_mut(&id) {
            experiment.games.push(ExperimentGame {
                game_id,
                variant_a_is_white,
                outcome: GameOutcome::Pending,
                started_at: Utc::now(),
                finished_at: None,
                plies: None,
            });
        }
    }

    /// Records `game_id`'s final outcome (a real chess result or an
    /// abort -- see `GameOutcome`), its ply count (`None` if a final
    /// snapshot wasn't available -- see `ExperimentGame::plies`'s
    /// docs), and its finish time, against `id`'s experiment. Silently
    /// does nothing if either the experiment or that particular game
    /// entry isn't found -- mirrors `GameStore::abort`'s "the caller
    /// shouldn't panic over state that vanished" stance; there's no
    /// persistence yet to make that likely, but nothing here should
    /// crash the orchestration task over it either.
    fn record_game_outcome(
        &self,
        id: ExperimentId,
        game_id: GameId,
        outcome: GameOutcome,
        plies: Option<usize>,
    ) {
        let mut experiments = self
            .experiments
            .lock()
            .expect("experiment store mutex poisoned");
        let Some(experiment) = experiments.get_mut(&id) else {
            return;
        };
        if let Some(game) = experiment.games.iter_mut().find(|g| g.game_id == game_id) {
            game.outcome = outcome;
            game.finished_at = Some(Utc::now());
            game.plies = plies;
        }
    }

    /// Records `id`'s experiment as finished right now -- called once,
    /// by `run_experiment`, after its last game has settled. A no-op
    /// if the experiment doesn't exist or `finished_at` is already
    /// set, so calling it twice (there's no real path to that today,
    /// but nothing here should assume there never will be) can't
    /// silently overwrite an earlier, more accurate finish time.
    fn finish(&self, id: ExperimentId) {
        let mut experiments = self
            .experiments
            .lock()
            .expect("experiment store mutex poisoned");
        let Some(experiment) = experiments.get_mut(&id) else {
            return;
        };
        if experiment.metadata.finished_at.is_none() {
            experiment.metadata.finished_at = Some(Utc::now());
        }
    }
}

/// Runs `id`'s experiment to completion: `spec.requested_games` games,
/// alternating which variant plays White each game (see the module
/// docs' pairing scheme), each played out via the ordinary
/// `GameStore`/`run_engine_loop` machinery. Intended to be
/// `tokio::spawn`ed once, right after `ExperimentStore::create` --
/// like `run_engine_loop`, this returns once there's nothing left to
/// do (every requested game finished), so the caller doesn't need to
/// `.await` it directly.
///
/// A game that ends in `Aborted` (an engine crashed, failed to spawn,
/// etc.) rather than a real chess result is not retried and not
/// counted in either variant's win/draw/loss tally -- it still occupies
/// one of `requested_games`' slots and is visible in
/// `ExperimentGame`/the underlying game's own snapshot, so an aborted
/// game is never silently hidden, just excluded from `wins_a`/`draws`/
/// `wins_b` (which only ever sum to `completed_games`, not
/// `requested_games`).
pub async fn run_experiment(
    store: GameStore,
    experiments: ExperimentStore,
    id: ExperimentId,
    spec: ExperimentSpec,
) {
    for game_index in 0..spec.requested_games {
        // Even-indexed games: A plays White. Odd-indexed: A plays Black.
        // Both games in each pair receive the same opening line.
        let variant_a_is_white = game_index % 2 == 0;
        let (white, black) = if variant_a_is_white {
            (&spec.variant_a, &spec.variant_b)
        } else {
            (&spec.variant_b, &spec.variant_a)
        };

        let white_info = engine_participant_info(white);
        let black_info = engine_participant_info(black);
        let snapshot = store.create_for_experiment(white_info, black_info, id);
        experiments.record_game_started(id, snapshot.id, variant_a_is_white);

        for &mv in opening_moves_for_game(game_index) {
            if let Err(err) = store.apply_move(snapshot.id, mv) {
                store.abort(
                    snapshot.id,
                    format!("built-in opening contains illegal move {mv}: {err:?}"),
                );
                break;
            }
        }

        let slots = EngineSlots {
            white: Some(white.config.clone()),
            black: Some(black.config.clone()),
        };
        crate::game::run_engine_loop(store.clone(), snapshot.id, slots, spec.move_time_ms).await;

        // `run_engine_loop` only returns once the game reached a
        // terminal status (Finished or Aborted) -- see its own docs --
        // so re-fetching the snapshot here always sees one of those,
        // never `Running`. If the game vanished entirely (no
        // persistence yet -- see GameStore's docs), there's nothing to
        // record and nothing more this loop iteration can do.
        if let Some(final_snapshot) = store.snapshot(snapshot.id) {
            let outcome = match final_snapshot.status {
                GameStatus::Finished { result } => GameOutcome::Finished { result },
                GameStatus::Aborted { .. } => GameOutcome::Aborted,
                GameStatus::Running => {
                    unreachable!("run_engine_loop only returns once the game is no longer Running")
                }
            };
            experiments.record_game_outcome(
                id,
                snapshot.id,
                outcome,
                Some(final_snapshot.moves.len()),
            );
        }
    }

    experiments.finish(id);
}

fn engine_participant_info(variant: &EngineVariant) -> crate::game::ParticipantInfo {
    crate::game::ParticipantInfo::Engine {
        name: variant.label.clone(),
        debug: variant.config.debug,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::game::{EngineSpec, Game, ParticipantInfo};

    fn fake_bee_spec() -> EngineSpec {
        EngineSpec {
            argv: vec![
                "sh".to_string(),
                "-c".to_string(),
                r#"
                while read -r line; do
                    case "$line" in
                        uci) echo "uciok" ;;
                        isready) echo "readyok" ;;
                        go*) echo "bestmove e2e4" ;;
                    esac
                done
                "#
                .to_string(),
            ],
            cwd: std::env::temp_dir(),
        }
    }

    fn fake_spec(requested_games: u32) -> ExperimentSpec {
        ExperimentSpec {
            variant_a: EngineVariant {
                label: "Baseline".to_string(),
                config: EngineConfig {
                    spec: fake_bee_spec(),
                    options: Vec::new(),
                    debug: false,
                },
            },
            variant_b: EngineVariant {
                label: "Candidate".to_string(),
                config: EngineConfig {
                    spec: fake_bee_spec(),
                    options: Vec::new(),
                    debug: false,
                },
            },
            requested_games,
            move_time_ms: 5,
        }
    }

    #[test]
    fn color_swapped_pairs_use_the_same_opening_and_pairs_rotate() {
        assert_eq!(opening_moves_for_game(0), opening_moves_for_game(1));
        assert_eq!(opening_moves_for_game(2), opening_moves_for_game(3));
        assert_ne!(opening_moves_for_game(0), opening_moves_for_game(2));
        assert_eq!(
            opening_moves_for_game((OPENING_SUITE.len() * 2) as u32),
            OPENING_SUITE[0]
        );
    }

    #[test]
    fn every_built_in_opening_is_legal_and_non_terminal() {
        for opening in OPENING_SUITE {
            let mut game = Game::new(ParticipantInfo::Human, ParticipantInfo::Human);
            for &mv in *opening {
                game.apply_move(mv).unwrap_or_else(|err| {
                    panic!("illegal move {mv} in opening {opening:?}: {err:?}")
                });
            }
            assert_eq!(game.status(), &GameStatus::Running, "opening {opening:?}");
        }
    }

    #[test]
    fn create_returns_a_running_snapshot_with_no_games_yet() {
        let store = ExperimentStore::new();
        let snapshot = store.create(fake_spec(4));

        assert_eq!(snapshot.status, ExperimentStatus::Running);
        assert_eq!(snapshot.requested_games, 4);
        assert_eq!(snapshot.completed_games, 0);
        assert_eq!(snapshot.score_a, None);
        assert_eq!(snapshot.elo_diff_a, None);
        assert!(snapshot.games.is_empty());
    }

    #[test]
    fn elo_diff_from_score_is_zero_at_an_even_score() {
        assert_eq!(elo_diff_from_score(0.5), Some(0.0));
    }

    #[test]
    fn elo_diff_from_score_matches_the_standard_reference_values() {
        // Widely published reference points for this exact formula
        // (e.g. the CCRL/CEGT Elo-from-score tables) -- cross-checking
        // against them, not just re-deriving the same formula the
        // implementation uses, is what actually catches a sign error
        // or an off-by-something in the log/ratio.
        let at_60_percent = elo_diff_from_score(0.60).unwrap();
        assert!((at_60_percent - 70.44).abs() < 0.01, "got {at_60_percent}");

        let at_75_percent = elo_diff_from_score(0.75).unwrap();
        assert!((at_75_percent - 190.85).abs() < 0.01, "got {at_75_percent}");

        // Symmetric: A scoring 25% (losing 3 of 4, roughly) should be
        // exactly as far behind as A scoring 75% is ahead.
        let at_25_percent = elo_diff_from_score(0.25).unwrap();
        assert!(
            (at_25_percent + at_75_percent).abs() < 1e-9,
            "should be exact negatives of each other"
        );
    }

    #[test]
    fn elo_diff_from_score_is_none_at_a_perfect_score_either_direction() {
        assert_eq!(
            elo_diff_from_score(0.0),
            None,
            "can't estimate finite Elo from an all-loss record"
        );
        assert_eq!(
            elo_diff_from_score(1.0),
            None,
            "can't estimate finite Elo from an all-win record"
        );
    }

    #[test]
    fn snapshot_elo_diff_tracks_score_a_including_a_perfect_score() {
        let store = ExperimentStore::new();
        let created = store.create(fake_spec(1));
        let game_id: GameId = "11111111-1111-1111-1111-111111111111".parse().unwrap();

        store.record_game_started(created.id, game_id, true);
        store.record_game_outcome(
            created.id,
            game_id,
            GameOutcome::Finished {
                result: GameResult::WhiteWins,
            },
            Some(30),
        );

        let snapshot = store.snapshot(created.id).unwrap();
        assert_eq!(
            snapshot.score_a,
            Some(1.0),
            "A played White and White won -- a perfect score so far"
        );
        assert_eq!(
            snapshot.elo_diff_a, None,
            "a perfect score from one game is exactly the 'can't estimate yet' case, not a real Elo number"
        );
    }

    #[test]
    fn create_records_reproducibility_metadata_with_no_finish_time_yet() {
        let store = ExperimentStore::new();
        let before = Utc::now();
        let snapshot = store.create(fake_spec(1));
        let after = Utc::now();

        assert_eq!(snapshot.metadata.lab_git_commit, LAB_GIT_COMMIT);
        assert_eq!(snapshot.metadata.variant_a_argv, fake_bee_spec().argv);
        assert_eq!(snapshot.metadata.variant_b_argv, fake_bee_spec().argv);
        assert!(snapshot.metadata.started_at >= before && snapshot.metadata.started_at <= after);
        assert_eq!(snapshot.metadata.finished_at, None);
    }

    #[test]
    fn stats_are_none_with_no_settled_games_yet() {
        let store = ExperimentStore::new();
        let created = store.create(fake_spec(1));

        assert_eq!(created.stats.avg_game_duration_ms, None);
        assert_eq!(created.stats.avg_plies, None);
        assert_eq!(created.stats.games_per_hour, None);
        assert!(created.stats.runtime_ms >= 0);
    }

    #[test]
    fn avg_plies_averages_across_settled_games_only() {
        let store = ExperimentStore::new();
        let created = store.create(fake_spec(3));
        let a: GameId = "11111111-1111-1111-1111-111111111111".parse().unwrap();
        let b: GameId = "22222222-2222-2222-2222-222222222222".parse().unwrap();
        let c: GameId = "33333333-3333-3333-3333-333333333333".parse().unwrap();

        store.record_game_started(created.id, a, true);
        store.record_game_outcome(
            created.id,
            a,
            GameOutcome::Finished {
                result: GameResult::WhiteWins,
            },
            Some(40),
        );
        store.record_game_started(created.id, b, false);
        store.record_game_outcome(
            created.id,
            b,
            GameOutcome::Finished {
                result: GameResult::Draw,
            },
            Some(80),
        );
        // A pending (still-running) game must not drag the average
        // down/up with a phantom zero -- only settled games with a
        // known ply count count.
        store.record_game_started(created.id, c, true);

        let snapshot = store.snapshot(created.id).unwrap();
        assert_eq!(snapshot.stats.avg_plies, Some(60.0));
    }

    #[test]
    fn avg_plies_excludes_an_aborted_game_with_no_known_ply_count() {
        let store = ExperimentStore::new();
        let created = store.create(fake_spec(2));
        let finished: GameId = "11111111-1111-1111-1111-111111111111".parse().unwrap();
        let aborted: GameId = "22222222-2222-2222-2222-222222222222".parse().unwrap();

        store.record_game_started(created.id, finished, true);
        store.record_game_outcome(
            created.id,
            finished,
            GameOutcome::Finished {
                result: GameResult::WhiteWins,
            },
            Some(50),
        );
        store.record_game_started(created.id, aborted, false);
        store.record_game_outcome(created.id, aborted, GameOutcome::Aborted, None);

        let snapshot = store.snapshot(created.id).unwrap();
        assert_eq!(
            snapshot.stats.avg_plies,
            Some(50.0),
            "the aborted game's None ply count shouldn't count as 0"
        );
    }

    #[test]
    fn game_duration_and_runtime_are_recorded_in_real_wall_clock_time() {
        let store = ExperimentStore::new();
        let created = store.create(fake_spec(1));
        let game_id: GameId = "11111111-1111-1111-1111-111111111111".parse().unwrap();

        store.record_game_started(created.id, game_id, true);
        std::thread::sleep(std::time::Duration::from_millis(5));
        store.record_game_outcome(
            created.id,
            game_id,
            GameOutcome::Finished {
                result: GameResult::Draw,
            },
            Some(10),
        );

        let snapshot = store.snapshot(created.id).unwrap();
        let avg_duration = snapshot
            .stats
            .avg_game_duration_ms
            .expect("one settled game should produce a duration");
        assert!(
            avg_duration >= 5.0,
            "should reflect the real ~5ms sleep, got {avg_duration}"
        );
        assert!(snapshot.stats.runtime_ms >= 5);
        let expected_games_per_hour = 3_600_000.0 / snapshot.stats.runtime_ms as f64;
        let actual_games_per_hour = snapshot
            .stats
            .games_per_hour
            .expect("one settled game should give a rate");
        assert!(
            (actual_games_per_hour - expected_games_per_hour).abs() < 1e-6,
            "expected ~{expected_games_per_hour}, got {actual_games_per_hour}"
        );
    }

    #[test]
    fn list_returns_every_experiment_newest_first() {
        let store = ExperimentStore::new();
        let first = store.create(fake_spec(1));
        let second = store.create(fake_spec(1));
        let third = store.create(fake_spec(1));

        let listed: Vec<ExperimentId> = store.list().into_iter().map(|e| e.id).collect();

        assert_eq!(listed, vec![third.id, second.id, first.id]);
    }

    #[test]
    fn list_is_empty_for_a_fresh_store() {
        let store = ExperimentStore::new();
        assert!(store.list().is_empty());
    }

    #[test]
    fn snapshot_of_unknown_id_is_none() {
        let store = ExperimentStore::new();
        let bogus: ExperimentId = "00000000-0000-0000-0000-000000000000".parse().unwrap();
        assert!(store.snapshot(bogus).is_none());
    }

    #[test]
    fn record_game_started_then_outcome_updates_the_snapshot() {
        let store = ExperimentStore::new();
        let created = store.create(fake_spec(2));
        let game_id: GameId = "11111111-1111-1111-1111-111111111111".parse().unwrap();

        store.record_game_started(created.id, game_id, true);
        let mid = store.snapshot(created.id).unwrap();
        assert_eq!(mid.games.len(), 1);
        assert_eq!(mid.games[0].outcome, GameOutcome::Pending);
        assert_eq!(mid.completed_games, 0);

        store.record_game_outcome(
            created.id,
            game_id,
            GameOutcome::Finished {
                result: GameResult::WhiteWins,
            },
            Some(42),
        );
        let after = store.snapshot(created.id).unwrap();
        assert_eq!(
            after.games[0].outcome,
            GameOutcome::Finished {
                result: GameResult::WhiteWins
            }
        );
        assert_eq!(after.games[0].plies, Some(42));
        assert!(after.games[0].finished_at.is_some());
        assert_eq!(after.wins_a, 1, "A played White in this game and White won");
        assert_eq!(after.completed_games, 1);
        assert_eq!(after.score_a, Some(1.0));
    }

    #[test]
    fn tally_credits_the_right_variant_regardless_of_color() {
        let store = ExperimentStore::new();
        let created = store.create(fake_spec(2));
        let game_a_white: GameId = "11111111-1111-1111-1111-111111111111".parse().unwrap();
        let game_a_black: GameId = "22222222-2222-2222-2222-222222222222".parse().unwrap();

        // Game 1: A is White, White wins -- A wins.
        store.record_game_started(created.id, game_a_white, true);
        store.record_game_outcome(
            created.id,
            game_a_white,
            GameOutcome::Finished {
                result: GameResult::WhiteWins,
            },
            Some(30),
        );
        // Game 2: A is Black, White wins -- B wins (B was White).
        store.record_game_started(created.id, game_a_black, false);
        store.record_game_outcome(
            created.id,
            game_a_black,
            GameOutcome::Finished {
                result: GameResult::WhiteWins,
            },
            Some(50),
        );

        let snapshot = store.snapshot(created.id).unwrap();
        assert_eq!(snapshot.wins_a, 1);
        assert_eq!(snapshot.wins_b, 1);
        assert_eq!(snapshot.draws, 0);
        assert_eq!(snapshot.score_a, Some(0.5));
        assert_eq!(
            snapshot.elo_diff_a,
            Some(0.0),
            "an even score through the real snapshot path"
        );
    }

    #[test]
    fn a_draw_counts_toward_neither_variants_wins() {
        let store = ExperimentStore::new();
        let created = store.create(fake_spec(1));
        let game_id: GameId = "11111111-1111-1111-1111-111111111111".parse().unwrap();

        store.record_game_started(created.id, game_id, true);
        store.record_game_outcome(
            created.id,
            game_id,
            GameOutcome::Finished {
                result: GameResult::Draw,
            },
            Some(80),
        );

        let snapshot = store.snapshot(created.id).unwrap();
        assert_eq!(snapshot.wins_a, 0);
        assert_eq!(snapshot.wins_b, 0);
        assert_eq!(snapshot.draws, 1);
        assert_eq!(snapshot.score_a, Some(0.5));
    }

    #[test]
    fn an_aborted_game_is_settled_but_tallies_toward_neither_variant() {
        let store = ExperimentStore::new();
        let created = store.create(fake_spec(1));
        let game_id: GameId = "11111111-1111-1111-1111-111111111111".parse().unwrap();

        store.record_game_started(created.id, game_id, true);
        store.record_game_outcome(created.id, game_id, GameOutcome::Aborted, None);

        let snapshot = store.snapshot(created.id).unwrap();
        assert_eq!(snapshot.games[0].outcome, GameOutcome::Aborted);
        assert_eq!(snapshot.games[0].plies, None);
        assert_eq!(snapshot.wins_a, 0);
        assert_eq!(snapshot.wins_b, 0);
        assert_eq!(snapshot.draws, 0);
        assert_eq!(snapshot.completed_games, 0, "an abort isn't a chess result");
        assert_eq!(
            snapshot.status,
            ExperimentStatus::Completed,
            "the only requested game has ended (aborted), so there's nothing left to run"
        );
    }

    #[test]
    fn experiment_is_completed_once_every_requested_game_has_settled() {
        let store = ExperimentStore::new();
        let created = store.create(fake_spec(1));
        let game_id: GameId = "11111111-1111-1111-1111-111111111111".parse().unwrap();

        store.record_game_started(created.id, game_id, true);
        assert_eq!(
            store.snapshot(created.id).unwrap().status,
            ExperimentStatus::Running,
            "started but not yet settled"
        );

        store.record_game_outcome(
            created.id,
            game_id,
            GameOutcome::Finished {
                result: GameResult::Draw,
            },
            Some(60),
        );
        assert_eq!(
            store.snapshot(created.id).unwrap().status,
            ExperimentStatus::Completed
        );
    }

    #[tokio::test]
    async fn run_experiment_plays_the_requested_number_of_games_and_alternates_colors() {
        // The fake engine always replies "bestmove e2e4" -- legal only
        // on the game's first move, so every one of these games aborts
        // (an illegal move) rather than reaching a real chess result.
        // That's fine for what this test checks: run_experiment still
        // must run exactly `requested_games` games, in order, alternating
        // colors, and return once every one of them has *ended* -- an
        // abort settles a game just as much as a real result does (see
        // GameOutcome's docs), it just contributes nothing to
        // wins_a/draws/wins_b. Result tallying itself is covered by the
        // record_game_outcome-based tests above.
        let game_store = GameStore::new();
        let experiments = ExperimentStore::new();
        let created = experiments.create(fake_spec(3));

        run_experiment(game_store, experiments.clone(), created.id, fake_spec(3)).await;

        let snapshot = experiments.snapshot(created.id).unwrap();
        assert_eq!(snapshot.games.len(), 3, "should have started all 3 games");
        assert_eq!(
            snapshot
                .games
                .iter()
                .filter(|g| g.variant_a_is_white)
                .count(),
            2,
            "games 0 and 2 (of 3) should have A playing White"
        );
        assert!(
            snapshot
                .games
                .iter()
                .all(|g| g.outcome == GameOutcome::Aborted),
            "every game should have aborted on its second (illegal) move: {:?}",
            snapshot.games
        );
        assert_eq!(
            snapshot.status,
            ExperimentStatus::Completed,
            "every game settled (by aborting), so there's nothing left to run"
        );
    }

    #[tokio::test]
    async fn run_experiment_tags_every_game_it_creates_with_the_experiment_id() {
        let game_store = GameStore::new();
        let experiments = ExperimentStore::new();
        let created = experiments.create(fake_spec(2));

        run_experiment(
            game_store.clone(),
            experiments.clone(),
            created.id,
            fake_spec(2),
        )
        .await;

        let snapshot = experiments.snapshot(created.id).unwrap();
        assert_eq!(snapshot.games.len(), 2);
        for game in &snapshot.games {
            let game_snapshot = game_store
                .snapshot(game.game_id)
                .expect("game should still exist in the game store");
            assert_eq!(
                game_snapshot.experiment_id,
                Some(created.id),
                "a game run_experiment created should link back to its own experiment"
            );
        }
    }

    #[tokio::test]
    async fn run_experiment_sets_finished_at_once_every_game_has_settled() {
        let game_store = GameStore::new();
        let experiments = ExperimentStore::new();
        let created = experiments.create(fake_spec(1));
        assert_eq!(created.metadata.finished_at, None);

        let before = Utc::now();
        run_experiment(game_store, experiments.clone(), created.id, fake_spec(1)).await;
        let after = Utc::now();

        let snapshot = experiments.snapshot(created.id).unwrap();
        let finished_at = snapshot
            .metadata
            .finished_at
            .expect("finished_at should be set once run_experiment returns");
        assert!(finished_at >= before && finished_at <= after);
        assert!(finished_at >= snapshot.metadata.started_at);
    }
}
