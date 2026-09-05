//! Authoritative game state: `Game`, its lifecycle, and an in-memory
//! store keyed by `GameId` -- see #69 (67b).
//!
//! This is slice 69a: the game-state model and its HTTP surface
//! (`POST /api/games`, `GET /api/games/:id`). Moves are applied here
//! via `Game::apply_move`, validated against `bee_chess_core`'s legal
//! move generator -- the same canonical chess-rules implementation
//! `bee-engine` itself uses (see `chess/src/lib.rs`'s docs for why that
//! sharing matters: a server that disagreed with Bee about what's legal
//! would be its own class of bug). Nothing here spawns an engine
//! process or drives a `go`/`bestmove` cycle automatically yet --
//! that's 69b. For now, `apply_move` is called directly (by a human
//! move via the API, or by a test), the same shape 69b's engine loop
//! will call it from once it exists.
//!
//! The API is deliberately game-ID-shaped even though only one
//! concurrent game is supported for now (`GameStore` is just a
//! `HashMap`, nothing stops it holding more) -- see #69's "avoids an
//! API reshape in #67d" note.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use bee_chess_core::{PieceKind, Position, Square};
use serde::Serialize;
use uuid::Uuid;

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
    /// win/loss/draw. Not constructed anywhere yet -- 69b is what
    /// introduces the engine-failure paths that would produce this;
    /// it's part of the API shape now so 69b doesn't need to change
    /// the snapshot's serialized shape when it lands.
    #[allow(dead_code)]
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
#[derive(Clone, Default)]
pub struct GameStore {
    games: Arc<Mutex<HashMap<GameId, Game>>>,
}

impl GameStore {
    pub fn new() -> Self {
        Self::default()
    }

    /// Creates a new game and returns its snapshot.
    pub fn create(&self) -> GameSnapshot {
        let game = Game::new();
        let snapshot = GameSnapshot::from(&game);
        self.games
            .lock()
            .expect("game store mutex poisoned")
            .insert(game.id, game);
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

    /// Applies `uci` to `id`'s game. `Err(None)` means no such game
    /// exists; `Err(Some(_))` means the game exists but the move was
    /// refused (see `ApplyMoveError`).
    pub fn apply_move(
        &self,
        id: GameId,
        uci: &str,
    ) -> Result<GameSnapshot, Option<ApplyMoveError>> {
        let mut games = self.games.lock().expect("game store mutex poisoned");
        let game = games.get_mut(&id).ok_or(None)?;
        game.apply_move(uci).map_err(Some)?;
        Ok(GameSnapshot::from(&*game))
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
