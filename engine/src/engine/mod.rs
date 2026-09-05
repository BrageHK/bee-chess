/// Engine
use crate::chess::{PieceKind, Position, Square};
use crate::diagnostics::{Diagnostic, DiagnosticBuffer, DiagnosticLevel, Diagnostics};
use crate::eval::MaterialEvaluator;
use crate::search::{self, SearchResult};

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
    diagnostics: DiagnosticBuffer,
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
            diagnostics: DiagnosticBuffer::new(),
        }
    }

    /// Emits a diagnostic. Engine/search code should call this instead
    /// of ever constructing UCI text directly (`info string ...`,
    /// `println!`, etc.) -- see `crate::diagnostics` for what belongs
    /// here versus in real UCI `info` fields. Whether (and how) this
    /// becomes visible is entirely up to whatever drains
    /// `take_diagnostics` later, typically the UCI adapter gated on
    /// `debug on`/`off`.
    pub fn emit_diagnostic(&mut self, level: DiagnosticLevel, message: impl Into<String>) {
        self.diagnostics.emit(level, message);
    }

    /// Removes and returns every diagnostic emitted since the last
    /// call, in emission order. Intended to be drained by the UCI
    /// adapter after each command, so diagnostics never pile up
    /// unboundedly if nothing is reading them.
    pub fn take_diagnostics(&mut self) -> Vec<Diagnostic> {
        self.diagnostics.drain()
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

    /// Searches the current position to exactly `depth` plies using
    /// fixed-depth negamax alpha-beta (see `crate::search::alpha_beta`)
    /// with a material-only evaluator, and returns the result. Does
    /// not mutate the current position -- `search::search` restores it
    /// fully via make/unmake on every path, including cut-off branches.
    ///
    /// No quiescence, no transposition table, no move ordering, no
    /// real cancellation. `depth` is searched to completion
    /// synchronously. See `search_for_time` for time-bounded iterative
    /// deepening instead of a fixed depth.
    #[must_use]
    pub fn search(&mut self, depth: u32) -> SearchResult {
        search::search(&mut self.position, depth, &MaterialEvaluator)
    }

    /// Searches the current position with iterative deepening (depth
    /// 1, 2, 3, ...) for up to `budget`, calling `on_depth_complete`
    /// after each depth that finishes in time, and returning the last
    /// depth that completed -- never a partially-searched one (see
    /// `crate::search::alpha_beta`'s module docs for why a cut-off
    /// depth's score can't be trusted). Always completes at least
    /// depth 1 even for a zero/expired budget.
    pub fn search_for_time(
        &mut self,
        budget: std::time::Duration,
        on_depth_complete: impl FnMut(&SearchResult),
    ) -> SearchResult {
        search::search_iterative(
            &mut self.position,
            budget,
            &MaterialEvaluator,
            on_depth_complete,
        )
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
    fn new_engine_has_no_diagnostics() {
        let mut engine = Engine::new();
        assert_eq!(engine.take_diagnostics(), Vec::new());
    }

    #[test]
    fn emit_diagnostic_then_take_returns_it() {
        let mut engine = Engine::new();
        engine.emit_diagnostic(DiagnosticLevel::Info, "position set to startpos");

        assert_eq!(
            engine.take_diagnostics(),
            vec![Diagnostic {
                level: DiagnosticLevel::Info,
                message: "position set to startpos".to_string(),
            }]
        );
    }

    #[test]
    fn take_diagnostics_drains_so_a_second_call_is_empty() {
        let mut engine = Engine::new();
        engine.emit_diagnostic(DiagnosticLevel::Debug, "one");
        engine.take_diagnostics();

        assert_eq!(engine.take_diagnostics(), Vec::new());
    }

    #[test]
    fn diagnostics_are_returned_in_emission_order() {
        let mut engine = Engine::new();
        engine.emit_diagnostic(DiagnosticLevel::Info, "first");
        engine.emit_diagnostic(DiagnosticLevel::Warn, "second");
        engine.emit_diagnostic(DiagnosticLevel::Error, "third");

        let messages: Vec<String> = engine
            .take_diagnostics()
            .into_iter()
            .map(|d| d.message)
            .collect();
        assert_eq!(messages, vec!["first", "second", "third"]);
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

    #[test]
    fn search_returns_a_legal_move_and_does_not_mutate_the_position() {
        let mut engine = Engine::new();
        let before = engine.position().clone();

        let result = engine.search(3);

        assert!(result.best_move.is_some());
        assert_eq!(engine.position(), &before);
    }

    #[test]
    fn search_finds_mate_in_one() {
        let mut engine = Engine::new();
        engine.set_position(
            Position::from_fen("6k1/5ppp/8/8/8/8/8/3QK3 w - - 0 1").expect("valid FEN"),
        );

        let result = engine.search(2);

        let best_move = result.best_move.expect("should find a move");
        assert_eq!(best_move.from(), "d1".parse().unwrap());
        assert_eq!(best_move.to(), "d8".parse().unwrap());
    }

    #[test]
    fn search_for_time_returns_a_legal_move_and_does_not_mutate_the_position() {
        let mut engine = Engine::new();
        let before = engine.position().clone();

        let result = engine.search_for_time(std::time::Duration::from_millis(50), |_| {});

        assert!(result.best_move.is_some());
        assert_eq!(engine.position(), &before);
    }

    #[test]
    fn search_for_time_calls_on_depth_complete_for_each_completed_depth() {
        let mut engine = Engine::new();
        let mut depths_seen = Vec::new();

        engine.search_for_time(std::time::Duration::from_millis(200), |result| {
            depths_seen.push(result.depth);
        });

        assert!(
            depths_seen.len() >= 2,
            "should complete more than one depth in 200ms"
        );
        assert_eq!(depths_seen.first(), Some(&1));
    }
}
