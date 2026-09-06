/// Engine
use crate::chess::{PieceKind, Position, Square};
use crate::diagnostics::{Diagnostic, DiagnosticBuffer, DiagnosticLevel, Diagnostics};
use crate::eval::{ExperimentalEvaluator, MaterialEvaluator, PositionalEvaluator};
use crate::search::{self, SearchOptions, SearchResult};

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
    evaluator: EvaluatorKind,
    search_options: SearchOptions,
    position: Position,
    diagnostics: DiagnosticBuffer,
    /// Zobrist hash of every position reached so far in the current
    /// game, in order, including the current position as the last
    /// entry -- what `is_threefold_repetition` checks against. Reset
    /// (not appended to) by `set_position`, since a UCI `position`
    /// command always resends the *entire* move list from a base
    /// position rather than incrementally extending the previous one
    /// (see `uci::PositionCommand::resolve`), so replaying it here
    /// would double-count. Only hashes are kept, not full `Position`
    /// clones, since a repetition check only needs "was this exact
    /// position reached before," not the position itself.
    position_history: Vec<u64>,
    // later:
    // evaluator: Box<dyn Evaluator>,
    // searcher: Searcher,
    // transposition_table: TranspositionTable,
    // options: EngineOptions,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum EvaluatorKind {
    Experimental,
    Material,
    #[default]
    Positional,
}

impl EvaluatorKind {
    pub const fn uci_name(self) -> &'static str {
        match self {
            Self::Experimental => "Experimental",
            Self::Material => "Material",
            Self::Positional => "Positional",
        }
    }

    pub fn parse(value: &str) -> Option<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "experimental" => Some(Self::Experimental),
            "material" => Some(Self::Material),
            "positional" => Some(Self::Positional),
            _ => None,
        }
    }
}

impl Engine {
    pub fn new() -> Self {
        let position = Position::startpos();
        let position_history = vec![position.zobrist_hash()];
        Self {
            debug: false,
            evaluator: EvaluatorKind::default(),
            search_options: SearchOptions::default(),
            position,
            diagnostics: DiagnosticBuffer::new(),
            position_history,
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

    /// Replaces the current position and resets game history to just
    /// this one position -- see `position_history`'s docs on why this
    /// resets rather than appends. A `position` command with a `moves`
    /// suffix rebuilds the rest of that history one `apply_move` call
    /// at a time after this.
    pub fn set_position(&mut self, position: Position) {
        self.position_history = vec![position.zobrist_hash()];
        self.position = position;
    }

    pub fn position(&self) -> &Position {
        &self.position
    }

    /// Whether the current position has now occurred for the third
    /// (or more) time in the game so far, per FIDE's threefold
    /// repetition rule -- a draw either side can claim. Compares
    /// Zobrist hashes, not full positions: two positions with the same
    /// hash are treated as the same position, which is what the hash
    /// is for (see `Position::zobrist_hash`'s docs on its very small
    /// false-positive surface around unusable en passant squares).
    #[must_use]
    pub fn is_threefold_repetition(&self) -> bool {
        let current = self.position.zobrist_hash();
        self.position_history
            .iter()
            .filter(|&&hash| hash == current)
            .count()
            >= 3
    }

    pub fn set_debug(&mut self, debug: bool) {
        self.debug = debug;
    }

    pub fn debug(&self) -> bool {
        self.debug
    }

    pub const fn evaluator(&self) -> EvaluatorKind {
        self.evaluator
    }

    pub fn set_evaluator(&mut self, evaluator: EvaluatorKind) {
        self.evaluator = evaluator;
    }

    pub const fn search_options(&self) -> SearchOptions {
        self.search_options
    }

    pub fn set_use_tt(&mut self, use_tt: bool) {
        self.search_options.use_tt = use_tt;
    }

    pub fn set_use_quiescence(&mut self, use_quiescence: bool) {
        self.search_options.use_quiescence = use_quiescence;
    }

    /// Resets game/search-specific engine state (TT generation and
    /// similar, once that exists) for a new game. This does **not**
    /// reset `position_history` or set the board to the starting
    /// position -- UCI's `ucinewgame` is always followed by a
    /// `position` command that establishes the actual position (and,
    /// via `set_position`, resets history to just that position), so
    /// doing either here too would just be redundant with (and
    /// potentially race) that follow-up command. For now, with no other
    /// such state yet to reset, this is a deliberate no-op.
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
                self.position_history.push(self.position.zobrist_hash());
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
    /// with a tapered positional evaluator, and returns the result. Does
    /// not mutate the current position -- `search::search` restores it
    /// fully via make/unmake on every path, including cut-off branches.
    ///
    /// `depth` is searched to completion synchronously. See `search_for_time` for time-bounded iterative
    /// deepening instead of a fixed depth.
    #[must_use]
    pub fn search(&mut self, depth: u32) -> SearchResult {
        match self.evaluator {
            EvaluatorKind::Experimental => search::search_with_options(
                &mut self.position,
                depth,
                &ExperimentalEvaluator,
                &self.position_history,
                self.search_options,
            ),
            EvaluatorKind::Material => search::search_with_options(
                &mut self.position,
                depth,
                &MaterialEvaluator,
                &self.position_history,
                self.search_options,
            ),
            EvaluatorKind::Positional => search::search_with_options(
                &mut self.position,
                depth,
                &PositionalEvaluator,
                &self.position_history,
                self.search_options,
            ),
        }
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
        match self.evaluator {
            EvaluatorKind::Experimental => search::search_iterative_with_options(
                &mut self.position,
                budget,
                &ExperimentalEvaluator,
                &self.position_history,
                self.search_options,
                on_depth_complete,
            ),
            EvaluatorKind::Material => search::search_iterative_with_options(
                &mut self.position,
                budget,
                &MaterialEvaluator,
                &self.position_history,
                self.search_options,
                on_depth_complete,
            ),
            EvaluatorKind::Positional => search::search_iterative_with_options(
                &mut self.position,
                budget,
                &PositionalEvaluator,
                &self.position_history,
                self.search_options,
                on_depth_complete,
            ),
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

    #[test]
    fn new_engine_is_not_a_repetition() {
        // Startpos has only occurred once so far -- nowhere close to
        // threefold.
        let engine = Engine::new();
        assert!(!engine.is_threefold_repetition());
    }

    #[test]
    fn shuffling_knights_back_to_the_same_position_three_times_is_a_repetition() {
        // Nf3 Nf6 Ng1 Ng8, repeated twice more: each full four-move
        // cycle returns to the exact starting position (same board,
        // same side to move, same castling rights -- nobody's king or
        // rook moved, so rights are untouched -- and no pawn move/
        // capture ever happens, so this is legal repetition, not just a
        // hash coincidence). Startpos itself is occurrence 1; two more
        // cycles bring it to 3.
        let mut engine = Engine::new();

        for _ in 0..2 {
            engine
                .apply_move(
                    Square::from_file_rank(6, 0),
                    Square::from_file_rank(5, 2),
                    None,
                ) // Ng1-f3
                .expect("Nf3 should be legal");
            engine
                .apply_move(
                    Square::from_file_rank(6, 7),
                    Square::from_file_rank(5, 5),
                    None,
                ) // Ng8-f6
                .expect("Nf6 should be legal");
            engine
                .apply_move(
                    Square::from_file_rank(5, 2),
                    Square::from_file_rank(6, 0),
                    None,
                ) // Nf3-g1
                .expect("Ng1 should be legal");
            engine
                .apply_move(
                    Square::from_file_rank(5, 5),
                    Square::from_file_rank(6, 7),
                    None,
                ) // Nf6-g8
                .expect("Ng8 should be legal");
        }

        // Board/side-to-move/castling-rights/en-passant match startpos
        // exactly -- the fields Zobrist hashing (and so repetition)
        // actually cares about. `halfmove_clock`/`fullmove_number`
        // correctly do *not* match (eight non-pawn, non-capture moves
        // were played, and four full moves elapsed): those are real
        // FIDE-rule counters, not part of what makes two positions "the
        // same" for repetition purposes, so `Position`'s own
        // `PartialEq` -- which does compare them -- is the wrong check
        // here; `zobrist_hash` is the one that matters.
        assert_eq!(
            engine.position().zobrist_hash(),
            Position::startpos().zobrist_hash()
        );
        assert!(engine.is_threefold_repetition());
        assert_eq!(engine.search(4).score, 0, "search must honor game history");
    }

    #[test]
    fn set_position_resets_repetition_history() {
        // A `position` command always resends the whole move list from
        // a base position (see `position_history`'s docs) -- a fresh
        // `set_position` call must not let history from a previous
        // `position` command linger and cause a false repetition match.
        let mut engine = Engine::new();
        engine
            .apply_move(
                Square::from_file_rank(6, 0),
                Square::from_file_rank(5, 2),
                None,
            ) // Nf3
            .expect("Nf3 should be legal");
        engine
            .apply_move(
                Square::from_file_rank(6, 7),
                Square::from_file_rank(5, 5),
                None,
            ) // Nf6
            .expect("Nf6 should be legal");

        engine.set_position(Position::startpos());

        assert!(!engine.is_threefold_repetition());
    }
}
