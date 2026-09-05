//! Authoritative game state: `Game`, its lifecycle, an in-memory store
//! keyed by `GameId`, and (slice 69b) the automatic engine-vs-engine
//! play loop -- see #69 (67b).
//!
//! 69a's game-state model and HTTP surface (`POST /api/games`,
//! `GET /api/games/:id`) are unchanged: moves are applied via
//! `Game::apply_move`, validated against `bee_chess_core`'s legal move
//! generator -- the same canonical chess-rules implementation
//! `bee-engine` itself uses (see `chess/src/lib.rs`'s docs for why that
//! sharing matters: a server that disagreed with Bee about what's legal
//! would be its own class of bug).
//!
//! 69b adds `run_engine_loop`: spawned as a background task per game
//! that has one or two engine-driven sides, it repeatedly asks whichever
//! engine is on move for a `bestmove` (via `uci_process::UciProcess`)
//! and applies it through the exact same `Game::apply_move` a human's
//! `POST /api/games/:id/moves` call goes through -- there is no second,
//! engine-only code path for legality/status. A side with no
//! `EngineSpec` is a human slot: the loop simply waits (polling
//! `GameStore` until that side's move shows up via the API) rather than
//! trying to move for them.
//!
//! 69c-1a adds a per-game live event stream (`GameEvent`, `GameStore::
//! subscribe`), so a WebSocket client (69c-1a's `api::game_events_ws`)
//! can mirror the exact raw UCI traffic a direct browser connection to
//! the engine used to see (`GameEvent::Uci`), plus `GameEvent::Updated`
//! whenever the authoritative snapshot changes (a move played, or the
//! game reaching a terminal status) -- deliberately *not* a structured
//! `SearchInfo` type parsed server-side: the frontend already has a
//! working raw-UCI-line parser (`uciInfo.ts`), so there is no reason to
//! duplicate that parsing in Rust just because the lines now originate
//! from a server-owned process instead of a directly-browser-connected
//! one. The snapshot returned by `GET /api/games/:id` stays the
//! authoritative resync mechanism regardless of events -- a client that
//! missed some events (or never subscribed at all) is never wrong about
//! the game's real state, only possibly a little stale on live search
//! telemetry, which is expected to matter far less.
//!
//! The API is deliberately game-ID-shaped even though only one
//! concurrent game is supported for now (`GameStore` is just a
//! `HashMap`, nothing stops it holding more) -- see #69's "avoids an
//! API reshape in #67d" note.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use bee_chess_core::{Color, PieceKind, Position, Square};
use serde::Serialize;
use tokio::sync::broadcast;
use uuid::Uuid;

use crate::uci_process::{UciDirection, UciProcess};
use crate::uci_relay::EngineSpec;

/// How many events a slow (or absent) subscriber can lag behind before
/// the broadcast channel starts dropping its oldest ones. Deliberately
/// generous for the volume one game's UCI traffic realistically
/// produces (a handful of `info` lines a second at most); this only
/// matters at all for a subscriber that's connected but not reading,
/// and dropped events are just transient telemetry (see this module's
/// docs) -- never the authoritative snapshot -- so a lagging
/// subscriber missing some history is an acceptable, self-healing
/// (on their next receive) condition, not a correctness bug.
const EVENT_CHANNEL_CAPACITY: usize = 256;

/// A live event about one game, broadcast to every current subscriber
/// (see `GameStore::subscribe`). Not persisted or replayed -- a client
/// that connects late (or misses some) just relies on its next
/// `GET /api/games/:id` for the authoritative picture; see the module
/// docs for why events and the snapshot have deliberately different
/// jobs.
#[derive(Debug, Clone)]
pub enum GameEvent {
    /// One raw line of UCI traffic to/from one side's engine process --
    /// mirrors exactly what a direct browser connection to that engine
    /// would have seen (id/uciok/isready/readyok/info/bestmove, all of
    /// it, not just the lines this crate's own code acts on).
    Uci {
        color: Color,
        direction: UciDirection,
        line: String,
    },
    /// The authoritative snapshot changed -- a move was applied
    /// (human or engine) or the game reached a terminal status.
    /// Carries the new snapshot directly so a subscriber doesn't need
    /// a separate `GET` just to find out what changed.
    Updated(GameSnapshot),
}

/// Opaque game identifier, serialized as a plain string over the API.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize)]
#[serde(transparent)]
pub struct GameId(Uuid);

impl GameId {
    fn new() -> Self {
        GameId(Uuid::new_v4())
    }
}

impl std::fmt::Display for GameId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl std::str::FromStr for GameId {
    type Err = uuid::Error;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Ok(GameId(Uuid::parse_str(s)?))
    }
}

/// A game's current lifecycle state. `Finished`/`Aborted` carry enough
/// to explain themselves in a snapshot without the client needing to
/// infer anything from the position alone (e.g. "no legal moves" is
/// ambiguous between checkmate and stalemate; `GameResult` isn't).
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum GameStatus {
    Running,
    Finished {
        result: GameResult,
    },
    /// An engine failed, or some other condition stopped the game
    /// short of a real chess result. Distinct from `Finished` so a
    /// client never mistakes an error state for a legitimate
    /// win/loss/draw. Set via `Game::abort` -- see 69b's engine loop.
    Aborted {
        reason: String,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum GameResult {
    WhiteWins,
    BlackWins,
    Draw,
}

/// One game: the authoritative position, its full move list (UCI long
/// algebraic, e.g. `"e2e4"`/`"e7e8q"`), and lifecycle status.
///
/// Deliberately does not story any engine/process handles yet -- 69b
/// adds those. This slice only needs `Game` to be a correct state
/// machine over moves, provable independently of anything about
/// engines.
#[derive(Debug, Clone)]
pub struct Game {
    pub id: GameId,
    position: Position,
    moves: Vec<String>,
    status: GameStatus,
}

/// Why `Game::apply_move` refused a move -- carries enough detail for
/// an API caller to show something better than a bare 400.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ApplyMoveError {
    GameNotRunning,
    NotAWellFormedMove,
    IllegalMove,
}

impl Game {
    /// Starts a new game from the standard starting position.
    pub fn new() -> Self {
        Game {
            id: GameId::new(),
            position: Position::startpos(),
            moves: Vec::new(),
            status: GameStatus::Running,
        }
    }

    /// Test-only: starts a game from an arbitrary position, so
    /// terminal-status handling (checkmate/stalemate) can be tested
    /// directly against a hand-picked FEN instead of needing a real
    /// legal move sequence that happens to reach one.
    #[cfg(test)]
    fn from_position(position: Position) -> Self {
        let mut game = Game {
            id: GameId::new(),
            position,
            moves: Vec::new(),
            status: GameStatus::Running,
        };
        game.update_status_after_move();
        game
    }

    pub fn status(&self) -> &GameStatus {
        &self.status
    }

    pub fn fen(&self) -> String {
        self.position.to_fen()
    }

    pub fn moves(&self) -> &[String] {
        &self.moves
    }

    /// Whose turn it currently is, independent of `status` -- callers
    /// (the engine loop) still need this to know which side to ask for
    /// a move even after checking `status` is `Running` themselves.
    pub fn side_to_move(&self) -> Color {
        self.position.side_to_move()
    }

    /// Marks the game aborted with `reason` -- called when the engine
    /// loop can't continue (a process failed to spawn, died mid-game,
    /// or replied with something that isn't actually legal here despite
    /// being the engine's own choice, which would itself be a serious
    /// bug worth surfacing distinctly from a normal chess result).
    /// A no-op if the game already has a terminal status, since an
    /// already-finished/aborted game shouldn't be re-aborted out from
    /// under whatever ended it first.
    pub fn abort(&mut self, reason: impl Into<String>) {
        if self.status == GameStatus::Running {
            self.status = GameStatus::Aborted {
                reason: reason.into(),
            };
        }
    }

    /// Applies `uci` (e.g. `"e2e4"`, `"e7e8q"`) to the current
    /// position if the game is running and the move is legal, updating
    /// status if the game just ended. Mirrors
    /// `bee_engine::engine::Engine::apply_move`'s matching approach
    /// (parse into from/to/promotion, then find the one legal move
    /// that matches) since that's the proven way to turn UCI move text
    /// into an actual board move without `Move` itself needing to
    /// carry protocol-text concerns.
    pub fn apply_move(&mut self, uci: &str) -> Result<(), ApplyMoveError> {
        if self.status != GameStatus::Running {
            return Err(ApplyMoveError::GameNotRunning);
        }

        let (from, to, promotion) =
            parse_uci_move(uci).ok_or(ApplyMoveError::NotAWellFormedMove)?;

        let matching_move = self
            .position
            .generate_legal_moves()
            .into_iter()
            .find(|mv| {
                mv.from() == from && mv.to() == to && mv.flag().promotion_kind() == promotion
            })
            .ok_or(ApplyMoveError::IllegalMove)?;

        self.position.make_move(matching_move);
        self.moves.push(uci.to_string());
        self.update_status_after_move();
        Ok(())
    }

    fn update_status_after_move(&mut self) {
        if !self.position.generate_legal_moves().is_empty() {
            return;
        }
        self.status = GameStatus::Finished {
            result: if self.position.in_check() {
                // The side to move is checkmated -- the *other* side won.
                if self.position.side_to_move() == bee_chess_core::Color::White {
                    GameResult::BlackWins
                } else {
                    GameResult::WhiteWins
                }
            } else {
                GameResult::Draw // stalemate
            },
        };
    }
}

impl Default for Game {
    fn default() -> Self {
        Self::new()
    }
}

/// Parses a UCI move token (`"e2e4"` or `"e7e8q"`) into (from, to,
/// promotion), not checking legality -- same shape as
/// `bee_engine::uci::UciMove::parse`, kept small and local here rather
/// than shared, since this is the only place in `lab/` that needs it.
fn parse_uci_move(s: &str) -> Option<(Square, Square, Option<PieceKind>)> {
    let (from, to, promotion) = match s.len() {
        4 => (&s[0..2], &s[2..4], None),
        5 => (&s[0..2], &s[2..4], Some(&s[4..5])),
        _ => return None,
    };

    let from = from.parse().ok()?;
    let to = to.parse().ok()?;
    let promotion = match promotion {
        Some(letter) => Some(match letter {
            "q" => PieceKind::Queen,
            "r" => PieceKind::Rook,
            "b" => PieceKind::Bishop,
            "n" => PieceKind::Knight,
            _ => return None,
        }),
        None => None,
    };

    Some((from, to, promotion))
}

/// A complete, self-sufficient snapshot of one game -- the shape
/// `GET /api/games/:id` returns. The primary resync mechanism per #69:
/// a client that just (re)connected renders this directly, with no
/// need to have seen any prior event.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct GameSnapshot {
    pub id: GameId,
    pub fen: String,
    pub moves: Vec<String>,
    #[serde(flatten)]
    pub status: GameStatus,
}

impl From<&Game> for GameSnapshot {
    fn from(game: &Game) -> Self {
        GameSnapshot {
            id: game.id,
            fen: game.fen(),
            moves: game.moves().to_vec(),
            status: game.status().clone(),
        }
    }
}

/// In-memory store of every game the server currently knows about.
/// `Arc<Mutex<..>>` rather than anything fancier: game count and
/// request volume are both tiny for this development/orchestration
/// server (see #67's module docs), so a plain mutex is simplest and
/// correct; revisit only if it's ever shown to matter.
///
/// Event channels (`events`) live in their own map, separate from
/// `games`, since a channel needs to be cheaply cloneable out to a
/// WebSocket handler independent of (and without holding) the games
/// mutex, and outlives no particular lock scope -- a `broadcast::
/// Sender` is itself just a cheap `Arc`-backed handle.
#[derive(Clone, Default)]
pub struct GameStore {
    games: Arc<Mutex<HashMap<GameId, Game>>>,
    events: Arc<Mutex<HashMap<GameId, broadcast::Sender<GameEvent>>>>,
}

impl GameStore {
    pub fn new() -> Self {
        Self::default()
    }

    /// Creates a new game and returns its snapshot.
    pub fn create(&self) -> GameSnapshot {
        let game = Game::new();
        let snapshot = GameSnapshot::from(&game);
        let id = game.id;
        self.games
            .lock()
            .expect("game store mutex poisoned")
            .insert(id, game);
        let (sender, _) = broadcast::channel(EVENT_CHANNEL_CAPACITY);
        self.events
            .lock()
            .expect("event channel map mutex poisoned")
            .insert(id, sender);
        snapshot
    }

    /// Returns `id`'s current snapshot, or `None` if no such game
    /// exists (never created, or this process restarted since -- there
    /// is no persistence yet, see #67's slice 5).
    pub fn snapshot(&self, id: GameId) -> Option<GameSnapshot> {
        self.games
            .lock()
            .expect("game store mutex poisoned")
            .get(&id)
            .map(GameSnapshot::from)
    }

    /// `id`'s current side to move, straight from `Game::side_to_move`
    /// (the authoritative source) rather than derived from a
    /// snapshot's move-list length -- see `run_engine_loop`, the only
    /// caller.
    fn side_to_move(&self, id: GameId) -> Option<Color> {
        self.games
            .lock()
            .expect("game store mutex poisoned")
            .get(&id)
            .map(Game::side_to_move)
    }

    /// Applies `uci` to `id`'s game. `Err(None)` means no such game
    /// exists; `Err(Some(_))` means the game exists but the move was
    /// refused (see `ApplyMoveError`). Broadcasts `GameEvent::Updated`
    /// on success -- see the module docs on why events carry the new
    /// snapshot directly.
    pub fn apply_move(
        &self,
        id: GameId,
        uci: &str,
    ) -> Result<GameSnapshot, Option<ApplyMoveError>> {
        let snapshot = {
            let mut games = self.games.lock().expect("game store mutex poisoned");
            let game = games.get_mut(&id).ok_or(None)?;
            game.apply_move(uci).map_err(Some)?;
            GameSnapshot::from(&*game)
        };
        self.publish(id, GameEvent::Updated(snapshot.clone()));
        Ok(snapshot)
    }

    /// Marks `id`'s game aborted with `reason`, if it still exists and
    /// is still running -- see `Game::abort`. Silently does nothing if
    /// the game doesn't exist (it may have been created and then this
    /// process restarted, though there's no persistence yet to make
    /// that a real scenario in practice -- still, the engine loop
    /// shouldn't panic over it). Broadcasts `GameEvent::Updated` if the
    /// game existed.
    fn abort(&self, id: GameId, reason: impl Into<String>) {
        let snapshot = {
            let mut games = self.games.lock().expect("game store mutex poisoned");
            let Some(game) = games.get_mut(&id) else {
                return;
            };
            game.abort(reason);
            GameSnapshot::from(&*game)
        };
        self.publish(id, GameEvent::Updated(snapshot));
    }

    /// Subscribes to `id`'s live event stream. Returns `None` if no
    /// such game exists (never created, or already gone -- there is no
    /// persistence yet). A subscription outlives the call that created
    /// it; drop the returned receiver to unsubscribe.
    pub fn subscribe(&self, id: GameId) -> Option<broadcast::Receiver<GameEvent>> {
        self.events
            .lock()
            .expect("event channel map mutex poisoned")
            .get(&id)
            .map(broadcast::Sender::subscribe)
    }

    /// Publishes `event` to `id`'s subscribers, if any and if the game
    /// still has a channel (see `create`/`subscribe`). A
    /// `SendError` (no receivers currently subscribed) is expected and
    /// silently ignored -- nobody is listening right now, which is
    /// fine; it isn't this store's job to guarantee delivery, only to
    /// deliver to whoever happens to be subscribed at the time (see
    /// the module docs on events being transient, not replayed).
    fn publish(&self, id: GameId, event: GameEvent) {
        if let Some(sender) = self
            .events
            .lock()
            .expect("event channel map mutex poisoned")
            .get(&id)
        {
            let _ = sender.send(event);
        }
    }
}

/// One engine-driven side's full configuration: which binary to spawn,
/// plus the `setoption`s and debug flag a direct browser connection to
/// the same engine could already set (see `UciClient.setOption`/
/// `setDebug` in the frontend's `engine.ts`) -- e.g. Stockfish's
/// `UCI_LimitStrength`/`UCI_Elo`, or Bee's debug diagnostics. Applied
/// once, right after the process's `uci`/`isready` handshake, before
/// any `go` -- see `run_engine_loop`.
#[derive(Debug, Clone)]
pub struct EngineConfig {
    pub spec: EngineSpec,
    /// `(name, value)` pairs, sent in order via `setoption`.
    pub options: Vec<(String, String)>,
    pub debug: bool,
}

/// Which side, if any, an engine plays as automated -- `None` is a
/// human slot: the automatic loop below simply waits for that side's
/// move to arrive via `POST /api/games/:id/moves` instead of ever
/// trying to move for it.
#[derive(Debug, Clone, Default)]
pub struct EngineSlots {
    pub white: Option<EngineConfig>,
    pub black: Option<EngineConfig>,
}

impl EngineSlots {
    /// Whether at least one side is engine-driven -- if neither is,
    /// there's nothing for the automatic loop to do at all, and the
    /// caller shouldn't bother spawning it.
    pub fn any_engine(&self) -> bool {
        self.white.is_some() || self.black.is_some()
    }

    fn config_for(&self, color: Color) -> Option<&EngineConfig> {
        match color {
            Color::White => self.white.as_ref(),
            Color::Black => self.black.as_ref(),
        }
    }
}

/// Drives `id`'s game to completion automatically, asking whichever
/// engine-controlled side is on move for a `bestmove` (with a
/// `move_time_ms` budget per move) and applying it via
/// `GameStore::apply_move` -- the same path a human's API call goes
/// through, so there is exactly one place legality/status is decided,
/// regardless of who's moving.
///
/// A side with no `EngineConfig` in `slots` is a human slot: the loop
/// polls (checking back every `HUMAN_MOVE_POLL_INTERVAL`) until that
/// side's move shows up in the game's move list, applied by someone
/// else calling the API, rather than ever trying to move for them.
///
/// Spawns one `UciProcess` per engine-controlled side, once, and reuses
/// it for the rest of the game rather than respawning per move --
/// mirrors how a real UCI GUI drives an engine across a whole game
/// (`position` + `go` repeated on the same process), not a fresh
/// process per ply.
///
/// Ends (returns) once the game reaches any terminal status --
/// `Finished` (checkmate/stalemate, detected the same way 69a already
/// did) or `Aborted` (a process failed to spawn or died mid-game, or
/// the store says the game no longer exists at all -- e.g. this
/// process restarted, though there's no persistence yet to make that
/// likely in practice).
pub async fn run_engine_loop(store: GameStore, id: GameId, slots: EngineSlots, move_time_ms: u64) {
    const HUMAN_MOVE_POLL_INTERVAL: Duration = Duration::from_millis(200);

    let mut white_process = None;
    let mut black_process = None;

    loop {
        let Some(snapshot) = store.snapshot(id) else {
            return; // game no longer exists -- nothing left to drive
        };
        if !matches!(snapshot.status, GameStatus::Running) {
            return; // finished or aborted, by this loop or otherwise
        }

        let Some(side_to_move) = store.side_to_move(id) else {
            return; // game vanished between the snapshot above and now
        };

        let Some(config) = slots.config_for(side_to_move) else {
            // Human slot: wait for their move to show up via the API
            // instead of polling tighter than a human can plausibly
            // move anyway.
            tokio::time::sleep(HUMAN_MOVE_POLL_INTERVAL).await;
            continue;
        };

        let process_slot = match side_to_move {
            Color::White => &mut white_process,
            Color::Black => &mut black_process,
        };

        if process_slot.is_none() {
            let store_for_events = store.clone();
            let on_line = Box::new(move |direction: UciDirection, line: &str| {
                // Best-effort: `publish` already silently no-ops if
                // nobody's subscribed or the game's gone, so nothing
                // here needs its own error handling.
                store_for_events.publish(
                    id,
                    GameEvent::Uci {
                        color: side_to_move,
                        direction,
                        line: line.to_string(),
                    },
                );
            });
            let mut process =
                match UciProcess::spawn(&config.spec.argv, &config.spec.cwd, Some(on_line)).await {
                    Ok(process) => process,
                    Err(err) => {
                        store.abort(id, format!("engine failed to start: {err}"));
                        return;
                    }
                };

            // Apply this side's configuration (Stockfish's Elo limit,
            // Bee's debug flag, etc. -- see `EngineConfig`'s docs) once,
            // right after the handshake, before any `go`.
            for (name, value) in &config.options {
                if let Err(err) = process.set_option(name, value).await {
                    store.abort(id, format!("failed to set {name}={value}: {err}"));
                    return;
                }
            }
            if config.debug {
                if let Err(err) = process.set_debug(true).await {
                    store.abort(id, format!("failed to enable debug: {err}"));
                    return;
                }
            }

            *process_slot = Some(process);
        }
        let process = process_slot.as_mut().expect("just ensured Some above");

        let mv = match process.best_move(&snapshot.moves, move_time_ms).await {
            Ok(Some(mv)) => mv,
            Ok(None) => {
                // The engine itself says no legal move (bestmove
                // 0000) -- it agrees the game is over. Our own
                // checkmate/stalemate detection should have already
                // caught this on the *previous* apply_move and
                // returned above; reaching here anyway would mean our
                // legality view and the engine's disagree, worth
                // surfacing distinctly rather than silently looping.
                store.abort(
                    id,
                    "engine reported no legal move for a position we think is still playable",
                );
                return;
            }
            Err(err) => {
                store.abort(id, format!("engine error: {err}"));
                return;
            }
        };

        if let Err(err) = store.apply_move(id, &mv) {
            // The engine's own chosen move wasn't legal by our
            // canonical rules -- since both sides share
            // bee-chess-core, this should never happen; treat it as
            // the serious bug it would be rather than silently
            // dropping the move.
            store.abort(id, format!("engine played an illegal move {mv}: {err:?}"));
            return;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_game_starts_at_the_standard_position() {
        let game = Game::new();
        assert_eq!(game.fen(), Position::startpos().to_fen());
        assert!(game.moves().is_empty());
        assert_eq!(game.status(), &GameStatus::Running);
    }

    #[test]
    fn legal_move_updates_position_and_move_list() {
        let mut game = Game::new();
        game.apply_move("e2e4")
            .expect("e2e4 should be legal from startpos");

        assert_eq!(game.moves(), &["e2e4".to_string()]);
        assert_ne!(game.fen(), Position::startpos().to_fen());
        assert_eq!(game.status(), &GameStatus::Running);
    }

    #[test]
    fn illegal_move_is_rejected_and_state_is_unchanged() {
        let mut game = Game::new();
        let before_fen = game.fen();

        let result = game.apply_move("e2e5"); // not a legal pawn move

        assert_eq!(result, Err(ApplyMoveError::IllegalMove));
        assert_eq!(game.fen(), before_fen);
        assert!(game.moves().is_empty());
    }

    #[test]
    fn malformed_move_text_is_rejected() {
        let mut game = Game::new();
        let result = game.apply_move("not a move");
        assert_eq!(result, Err(ApplyMoveError::NotAWellFormedMove));
    }

    #[test]
    fn checkmate_finishes_the_game_with_the_mating_sides_win() {
        // Fool's mate: fastest possible checkmate, White gets mated.
        let mut game = Game::new();
        for mv in ["f2f3", "e7e5", "g2g4", "d8h4"] {
            game.apply_move(mv)
                .expect("scholar/fool's mate setup should be legal");
        }

        assert_eq!(
            game.status(),
            &GameStatus::Finished {
                result: GameResult::BlackWins
            }
        );
    }

    #[test]
    fn a_move_after_the_game_is_finished_is_rejected() {
        let mut game = Game::new();
        for mv in ["f2f3", "e7e5", "g2g4", "d8h4"] {
            game.apply_move(mv).expect("setup move should be legal");
        }

        let result = game.apply_move("a2a3");
        assert_eq!(result, Err(ApplyMoveError::GameNotRunning));
    }

    #[test]
    fn stalemate_finishes_the_game_as_a_draw() {
        // Classic stalemate position (also used in engine's own search
        // tests): Black to move, no legal moves, not in check.
        let position = Position::from_fen("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1").expect("valid FEN");
        assert!(
            position.generate_legal_moves().is_empty(),
            "test setup: expected stalemate"
        );
        assert!(!position.in_check(), "test setup: stalemate, not checkmate");

        let game = Game::from_position(position);

        assert_eq!(
            game.status(),
            &GameStatus::Finished {
                result: GameResult::Draw
            }
        );
    }

    #[test]
    fn game_store_create_then_snapshot_round_trips() {
        let store = GameStore::new();
        let created = store.create();

        let snapshot = store
            .snapshot(created.id)
            .expect("just-created game should exist");
        assert_eq!(snapshot.id, created.id);
        assert_eq!(snapshot.fen, Position::startpos().to_fen());
        assert!(snapshot.moves.is_empty());
    }

    #[test]
    fn game_store_snapshot_of_unknown_id_is_none() {
        let store = GameStore::new();
        assert!(store.snapshot(GameId::new()).is_none());
    }

    #[test]
    fn game_store_apply_move_updates_the_stored_game() {
        let store = GameStore::new();
        let created = store.create();

        let snapshot = store
            .apply_move(created.id, "e2e4")
            .expect("e2e4 should be legal");

        assert_eq!(snapshot.moves, vec!["e2e4".to_string()]);
        // Re-fetching independently confirms the store's own copy was
        // mutated, not just the returned snapshot.
        let refetched = store.snapshot(created.id).unwrap();
        assert_eq!(refetched.moves, vec!["e2e4".to_string()]);
    }

    #[test]
    fn game_store_apply_move_on_unknown_id_is_err_none() {
        let store = GameStore::new();
        let result = store.apply_move(GameId::new(), "e2e4");
        assert_eq!(result, Err(None));
    }

    #[test]
    fn game_store_apply_move_illegal_is_err_some() {
        let store = GameStore::new();
        let created = store.create();
        let result = store.apply_move(created.id, "e2e5");
        assert_eq!(result, Err(Some(ApplyMoveError::IllegalMove)));
    }

    #[test]
    fn game_id_round_trips_through_display_and_from_str() {
        let id = GameId::new();
        let text = id.to_string();
        let parsed: GameId = text.parse().expect("should parse back");
        assert_eq!(parsed, id);
    }
}
