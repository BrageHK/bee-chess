/// Engine
use crate::chess::{PieceKind, Position, Square};

/// A move given as `(from, to, promotion)` could not be matched against
/// any currently legal move. Carries the inputs back so the caller
/// (the UCI adapter) can report a useful error rather than just
/// silently doing nothing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct IllegalMoveError {
    pub from: Square,
    pub to: Square,
    pub promotion: Option<PieceKind>,
}

pub struct Engine {
    debug: bool,
    position: Position,
    // later:
    // evaluator: Box<dyn Evaluator>,
    // searcher: Searcher,
    // transposition_table: TranspositionTable,
    // options: EngineOptions,
}

impl Engine {
    pub fn new() -> Self {
        Self {
            debug: false,
            position: Position::startpos(),
        }
    }

    pub fn set_position(&mut self, position: Position) {
        self.position = position;
    }

    pub fn position(&self) -> &Position {
        &self.position
    }

    pub fn set_debug(&mut self, debug: bool) {
        self.debug = debug;
    }

    pub fn debug(&self) -> bool {
        self.debug
    }

    /// Resets game/search-specific engine state (search history, TT
    /// generation, and similar, once those exist) for a new game. This
    /// does **not** set the board to the starting position -- UCI's
    /// `ucinewgame` is always followed by a `position` command that
    /// establishes the actual position, so doing that here too would
    /// just be redundant with (and potentially race) that follow-up
    /// command. For now, with no such state yet to reset, this is a
    /// deliberate no-op.
    pub fn new_game(&mut self) {}

    /// Applies a single move to the current position, given as UCI-style
    /// `(from, to, promotion)` coordinates (e.g. `e2e4` is
    /// `(e2, e4, None)`; `e7e8q` is `(e7, e8, Some(Queen))`).
    ///
    /// This does not itself know how to turn `(from, to, promotion)`
    /// into a fully-specified `Move` (the flag -- quiet, double push,
    /// en passant, castle -- depends on board context that the bare
    /// coordinates don't carry), so it looks the move up among the
    /// current position's legal moves and applies whichever one
    /// matches. Returns `Err` if no legal move matches, leaving the
    /// position unchanged.
    pub fn apply_move(
        &mut self,
        from: Square,
        to: Square,
        promotion: Option<PieceKind>,
    ) -> Result<(), IllegalMoveError> {
        let matching_move = self.position.generate_legal_moves().into_iter().find(|mv| {
            mv.from() == from && mv.to() == to && mv.flag().promotion_kind() == promotion
        });

        match matching_move {
            Some(mv) => {
                self.position.make_move(mv);
                Ok(())
            }
            None => Err(IllegalMoveError {
                from,
                to,
                promotion,
            }),
        }
    }
}

impl Default for Engine {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chess::Color;

    #[test]
    fn new_engine_starts_at_startpos() {
        let engine = Engine::new();
        assert_eq!(engine.position(), &Position::startpos());
    }

    #[test]
    fn new_engine_debug_is_off_by_default() {
        let engine = Engine::new();
        assert!(!engine.debug());
    }

    #[test]
    fn set_debug_updates_debug() {
        let mut engine = Engine::new();

        engine.set_debug(true);
        assert!(engine.debug());

        engine.set_debug(false);
        assert!(!engine.debug());
    }

    #[test]
    fn set_position_replaces_current_position() {
        let mut engine = Engine::new();
        let mut empty = Position::empty();
        empty.set_side_to_move(Color::Black);

        engine.set_position(empty.clone());

        assert_eq!(engine.position(), &empty);
        assert_ne!(engine.position(), &Position::startpos());
    }

    #[test]
    fn new_game_does_not_change_the_current_position() {
        let mut engine = Engine::new();
        let mut custom = Position::empty();
        custom.set_side_to_move(Color::Black);
        engine.set_position(custom.clone());

        engine.new_game();

        assert_eq!(engine.position(), &custom);
    }

    #[test]
    fn apply_move_updates_the_position() {
        let mut engine = Engine::new();

        engine
            .apply_move(
                Square::from_file_rank(4, 1), // e2
                Square::from_file_rank(4, 3), // e4
                None,
            )
            .expect("e2e4 should be legal from startpos");

        let mut expected = Position::startpos();
        let mv = expected
            .generate_legal_moves()
            .into_iter()
            .find(|mv| {
                mv.from() == Square::from_file_rank(4, 1) && mv.to() == Square::from_file_rank(4, 3)
            })
            .expect("e2e4 should be a legal move");
        expected.make_move(mv);

        assert_eq!(engine.position(), &expected);
    }

    #[test]
    fn apply_move_rejects_illegal_move() {
        let mut engine = Engine::new();
        let before = engine.position().clone();

        let result = engine.apply_move(
            Square::from_file_rank(4, 1), // e2
            Square::from_file_rank(4, 4), // e5: not reachable from e2
            None,
        );

        assert_eq!(
            result,
            Err(IllegalMoveError {
                from: Square::from_file_rank(4, 1),
                to: Square::from_file_rank(4, 4),
                promotion: None,
            })
        );
        assert_eq!(engine.position(), &before);
    }

    #[test]
    fn apply_move_matches_the_requested_promotion_kind() {
        let mut engine = Engine::new();
        engine.set_position(Position::from_fen("8/P6k/8/8/8/8/8/7K w - - 0 1").expect("valid FEN"));

        engine
            .apply_move(
                Square::from_file_rank(0, 6), // a7
                Square::from_file_rank(0, 7), // a8
                Some(PieceKind::Rook),
            )
            .expect("a7a8r should be a legal underpromotion");

        assert_eq!(
            engine.position().piece_at(Square::from_file_rank(0, 7)),
            Some(crate::chess::Piece::new(PieceKind::Rook, Color::White))
        );
    }
}
