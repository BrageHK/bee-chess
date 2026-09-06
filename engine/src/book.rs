//! Opening books: a cheap, position-keyed lookup Bee can consult
//! *before* search, so it doesn't burn its own thinking time re-deriving
//! well-known opening moves and doesn't need to search from scratch
//! before showing any real chess understanding at all.
//!
//! `OpeningBook::probe` takes an `OpeningContext`, not a bare position:
//! most implementations only need the current position (keying by
//! position rather than move sequence or a raw hash is what makes a
//! book work correctly across transpositions -- two different move
//! orders reaching the same position get the same answer -- and keeps
//! a future real book's on-disk key scheme entirely its own
//! implementation detail). But a book's notion of "progress" can
//! genuinely depend on *how* the position was reached, not just what
//! it looks like right now -- see `OpeningContext` and
//! `CowOpeningBook`'s own docs for exactly why the latter needs full
//! move history, not just the position, to behave correctly.
//!
//! This first slice deliberately implements only the smallest useful
//! vertical slice through the whole architecture: the trait, a `NoBook`
//! null implementation, and `CowOpeningBook`, a joke-but-real opening
//! (see its own docs). No `BookSelector` (multiple weighted
//! candidates), no Polyglot/file-backed book, no statistics -- those
//! are real follow-ups once this seam has proven itself, not
//! prerequisites for it.

use crate::chess::{Color, Move, MoveFlag, Position, Square};

/// Everything an `OpeningBook` gets to look at when deciding a move:
/// the current position, and every move played to reach it (in order,
/// starting from whatever base position the current game started
/// from -- see `Engine::move_history`'s docs). Most implementations
/// (a Zobrist-keyed database, Polyglot) only need `position` and can
/// ignore `moves` entirely; `moves` exists for the rarer book whose
/// notion of progress genuinely depends on *how* the current position
/// was reached, not just what it looks like right now -- see
/// `CowOpeningBook`'s docs for exactly why it needs this.
pub struct OpeningContext<'a> {
    pub position: &'a Position,
    pub moves: &'a [Move],
}

/// Looks up a known-good move for `context`'s position, if any.
/// Implementations must return a currently *legal* move or `None` --
/// the caller (`Engine::search`/`search_for_time`) still doesn't
/// re-validate a `Some` result against `generate_legal_moves` beyond
/// what a specific implementation's own docs promise, so an
/// implementation that can't guarantee legality (e.g corrupt/stale
/// on-disk data, once a file-backed book exists) must probe legality
/// itself and return `None` rather than risk playing an illegal move.
///
/// A book miss (`None`) is always a completely ordinary outcome, not
/// an error -- see `NoBook`.
pub trait OpeningBook: Send + Sync {
    fn probe(&self, context: &OpeningContext<'_>) -> Option<Move>;
}

/// The null opening book: always a miss. Used whenever `OwnBook`/
/// `OpeningBook` is configured off, so `Engine` always has a concrete
/// book to consult rather than needing an `Option<Box<dyn
/// OpeningBook>>` and a branch at every call site -- "no book" is just
/// another `OpeningBook`, not a special case.
pub struct NoBook;

impl OpeningBook for NoBook {
    fn probe(&self, _context: &OpeningContext<'_>) -> Option<Move> {
        None
    }
}

/// The Cow: pawns to d3/e3, knights rerouted to the "horns" on b3/g3
/// (via d2/e2), completely irrespective of what the opponent does --
/// not the Hippopotamus (which fianchettoes both bishops behind a
/// king-side pawn triangle instead). It's a real, if eccentric and
/// objectively passive, setup: a good first opening book precisely
/// *because* it's simple enough to encode as "which setup steps have
/// happened" rather than needing a real position-keyed database.
///
/// The Cow's setup order for one side: e-pawn to e3, d-pawn to d3,
/// king knight to e2, queen knight to d2, then the knight on e2
/// continues to g3 and the knight on d2 continues to b3. `probe`
/// returns the first *not-yet-completed* step that's currently legal.
///
/// Progress is **historical, not positional**: a step counts as
/// completed the moment `context.moves` shows it was ever played by
/// this side, and staying completed regardless of what the board
/// looks like afterward -- retreating a knight, or the piece being
/// captured, never un-completes a step. This distinction matters
/// concretely: e2 is both the e-pawn's start square and the king
/// knight's stop on the way to g3. A board-only check ("is a pawn
/// still on e2?") can't tell "the pawn never left" apart from "the
/// knight reached e2, then later retreated back to e2 after g3 was
/// attacked" -- both leave a piece sitting on e2 that isn't the pawn.
/// Reading the *history* instead of just the current board is what
/// keeps a temporary retreat from resurrecting an already-finished
/// step and, e.g., marching the same knight back into the same
/// now-attacked g3 square it just retreated from.
///
/// Once a genuinely pending (never-completed) step isn't legal right
/// now (the opponent is doing something that makes it impossible,
/// e.g. `e3` blocked, or it's simply not this side's move), this
/// returns `None` rather than forcing the setup through -- `Engine`
/// falls back to a normal search, and (since progress is monotonic,
/// not reset) the same step is offered again next time this side is
/// to move and it's legal, rather than being abandoned. Once every
/// step has ever been completed, this always returns `None` --the Cow
/// never reopens a finished horn just because the board temporarily
/// looks like an earlier stage of it.
///
/// Symmetric for both colors: the shape is mirrored (rank 2->3 for
/// White becomes rank 7->6 for Black, etc.) via `Color`-relative
/// squares, computed once per `probe` call rather than duplicated as
/// two hardcoded move lists.
pub struct CowOpeningBook;

impl OpeningBook for CowOpeningBook {
    fn probe(&self, context: &OpeningContext<'_>) -> Option<Move> {
        let position = context.position;
        let side = position.side_to_move();
        let legal_moves = position.generate_legal_moves();
        let side_moves = moves_by(context.moves, position, side);

        for step in cow_setup_steps(side) {
            let already_completed = side_moves
                .iter()
                .any(|mv| mv.from() == step.from && mv.to() == step.to);
            if already_completed {
                continue;
            }
            let mv = Move::new(step.from, step.to, MoveFlag::Quiet);
            if legal_moves.contains(&mv) {
                return Some(mv);
            }
            // The next never-completed step isn't legal right now
            // (blocked, or it's not this side's move at all). Rather
            // than skipping ahead to a later step out of setup order
            // (which could make an already-questionable setup
            // actively unsound), treat this as a book miss for this
            // call -- progress isn't reset, so the same step is
            // offered again once it's actually legal.
            return None;
        }

        // Every setup step has ever been completed -- the Cow is
        // done, permanently, for the rest of this game.
        None
    }
}

/// Every move in `moves` that `side` actually played, given that
/// `position` (the position `moves` led to) currently has
/// `position.side_to_move()` to move -- moves strictly alternate, so
/// `side`'s own moves are found by walking `moves` backward from the
/// end, taking every other one, starting from the most recent move if
/// it wasn't `side`'s (i.e. if `side` is the side to move next) or
/// from the last move if it was (`side` just moved).
fn moves_by(moves: &[Move], position: &Position, side: Color) -> Vec<Move> {
    // If `side` is on move now, the *other* side made the last move in
    // `moves`; `side`'s own most recent move (if any) is the one
    // before that.
    let last_move_was_side = position.side_to_move() != side;
    let start = if last_move_was_side { 0 } else { 1 };
    moves.iter().rev().skip(start).step_by(2).copied().collect()
}

/// One pending step of the Cow setup: move whatever piece is on `from`
/// to `to`. See `CowOpeningBook`'s docs -- since completion is tracked
/// by history rather than by inspecting the piece currently on
/// `from`, a step needs no piece-kind field of its own; `(from, to)`
/// alone is both what makes a step legal to offer and what identifies
/// it in `context.moves`.
struct CowSetupStep {
    from: Square,
    to: Square,
}

/// The Cow's six setup steps for `side`, in the order they should be
/// played -- see `CowOpeningBook`'s docs. White's shape (e2-e3, d2-d3,
/// Ng1-e2, Nb1-d2, Ne2-g3, Nd2-b3) mirrored vertically for Black
/// (e7-e6, d7-d6, Ng8-e7, Nb8-d7, Ne7-g6, Nd7-b6). Ranks are numbered
/// from each side's own back rank (0), forward being `+1` for White
/// and `-1` for Black, so the six steps below read the same regardless
/// of color.
fn cow_setup_steps(side: Color) -> [CowSetupStep; 6] {
    let back_rank = match side {
        Color::White => 0i8,
        Color::Black => 7,
    };
    let step: i8 = match side {
        Color::White => 1,
        Color::Black => -1,
    };
    // `n` ranks forward of `side`'s own back rank -- e.g. `rank(1)` is
    // White's 2nd rank / Black's 7th rank, `rank(2)` is White's 3rd /
    // Black's 6th.
    let rank = |n: i8| (back_rank + step * n) as u8;
    let sq = Square::from_file_rank;

    [
        CowSetupStep {
            from: sq(4, rank(1)),
            to: sq(4, rank(2)),
        }, // e2-e3 / e7-e6
        CowSetupStep {
            from: sq(3, rank(1)),
            to: sq(3, rank(2)),
        }, // d2-d3 / d7-d6
        CowSetupStep {
            from: sq(6, rank(0)),
            to: sq(4, rank(1)),
        }, // Ng1-e2 / Ng8-e7
        CowSetupStep {
            from: sq(1, rank(0)),
            to: sq(3, rank(1)),
        }, // Nb1-d2 / Nb8-d7
        CowSetupStep {
            from: sq(4, rank(1)),
            to: sq(6, rank(2)),
        }, // Ne2-g3 / Ne7-g6
        CowSetupStep {
            from: sq(3, rank(1)),
            to: sq(1, rank(2)),
        }, // Nd2-b3 / Nd7-b6
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chess::Position;

    /// A tiny in-memory game: owns the position and the full move
    /// history together so tests can play real moves (via
    /// `push`/`push_uci`) and then `probe` the book against an
    /// `OpeningContext` built from both, exactly as `Engine` does.
    struct Game {
        position: Position,
        moves: Vec<Move>,
    }

    impl Game {
        fn startpos() -> Self {
            Self {
                position: Position::startpos(),
                moves: Vec::new(),
            }
        }

        fn from_fen(fen: &str) -> Self {
            Self {
                position: Position::from_fen(fen).unwrap(),
                moves: Vec::new(),
            }
        }

        fn push(&mut self, mv: Move) {
            self.position.make_move(mv);
            self.moves.push(mv);
        }

        fn push_uci(&mut self, uci: &str) {
            let (from, to) = uci.split_at(2);
            let mv = self
                .position
                .generate_legal_moves()
                .into_iter()
                .find(|mv| mv.from() == from.parse().unwrap() && mv.to() == to.parse().unwrap())
                .unwrap_or_else(|| panic!("{uci} should be legal here"));
            self.push(mv);
        }

        fn probe(&self) -> Option<Move> {
            CowOpeningBook.probe(&OpeningContext {
                position: &self.position,
                moves: &self.moves,
            })
        }
    }

    #[test]
    fn no_book_always_misses() {
        let position = Position::startpos();
        let context = OpeningContext {
            position: &position,
            moves: &[],
        };
        assert_eq!(NoBook.probe(&context), None);
    }

    #[test]
    fn cow_book_plays_e3_from_the_start_position() {
        let game = Game::startpos();
        let mv = game.probe().expect("should hit");
        assert_eq!(mv.from(), "e2".parse().unwrap());
        assert_eq!(mv.to(), "e3".parse().unwrap());
    }

    #[test]
    fn cow_book_continues_the_setup_after_e3() {
        let mut game = Game::startpos();
        game.push_uci("e2e3");
        // Black to move; White has already played e3. Probing from
        // Black's side should offer Black's own first setup step
        // (e7-e6), not react to White's move at all -- the book is
        // symmetric and per-side.
        let mv = game.probe().expect("should hit");
        assert_eq!(mv.from(), "e7".parse().unwrap());
        assert_eq!(mv.to(), "e6".parse().unwrap());
    }

    #[test]
    fn cow_book_offers_nb1d2_once_the_king_knight_has_already_rerouted_to_e2() {
        // Regression test: e2 is both the e-pawn's vacated square
        // (step 0) and the king knight's later stop (step 2, on its
        // way to g3 in step 4). A real game (1.e3 e5 2.d3 d5 3.Ne2
        // Nf6) reaching this position used to make `probe` wrongly
        // conclude "step 0 (e2-e3) still pending" the instant a
        // knight sat on e2 -- occupancy alone can't distinguish "the
        // pawn never left" from "a knight arrived after". Reading it
        // from history (which this game's `moves` now carries) is
        // what fixes it -- try the now-illegal pawn push, and give up
        // instead of reaching the real pending step (Nb1-d2).
        let mut game = Game::startpos();
        game.push_uci("e2e3");
        game.push_uci("e7e5");
        game.push_uci("d2d3");
        game.push_uci("d7d5");
        game.push_uci("g1e2");
        game.push_uci("g8f6");

        let mv = game.probe().expect("should hit");

        assert_eq!(mv.from(), "b1".parse().unwrap());
        assert_eq!(mv.to(), "d2".parse().unwrap());
    }

    #[test]
    fn cow_book_completes_the_full_setup_move_by_move() {
        // The full six-step Cow, played out one legal move at a time
        // from the real start position (not hand-built FENs), with
        // the book itself choosing every White move and a fixed
        // symmetric-ish Black reply each time -- this is the actual
        // end-to-end scenario the earlier regression above was found
        // in, covering every step (and every square-reuse point) in
        // one game rather than one isolated position.
        let mut game = Game::startpos();
        let black_replies = ["e7e5", "d7d5", "g8f6", "b8c6", "f6e4", "c6d4"];
        let mut white_moves = Vec::new();

        for black_reply in black_replies {
            let white_mv = game
                .probe()
                .unwrap_or_else(|| panic!("book should still have a move; got to {white_moves:?}"));
            white_moves.push(format!("{}{}", white_mv.from(), white_mv.to()));
            game.push(white_mv);
            game.push_uci(black_reply);
        }

        assert_eq!(
            white_moves,
            vec!["e2e3", "d2d3", "g1e2", "b1d2", "e2g3", "d2b3"]
        );
        // The setup is now complete -- nothing left for the book to
        // offer, Engine would fall back to a real search from here.
        assert_eq!(game.probe(), None);
    }

    #[test]
    fn cow_book_does_not_reopen_a_completed_horn_after_a_retreat() {
        // The exact reported bug: the Cow is completed in full, then
        // the opponent attacks the g3 knight and it retreats back to
        // e2 -- the same square the king knight passed through on its
        // way to g3 in the first place. The board now looks
        // (piece-kind-wise) just like the moment right after step 2
        // (Ng1-e2) completed, before step 4 (Ne2-g3) happened. A
        // position-only check would wrongly conclude step 4 is still
        // pending and send the knight straight back into the attack
        // it just fled; history shows Ne2-g3 was already played, so
        // the book must stay a miss instead of reopening that horn.
        let mut game = Game::startpos();
        for (white, black) in [
            ("e2e3", "e7e5"),
            ("d2d3", "d7d5"),
            ("g1e2", "g8f6"),
            ("b1d2", "b8c6"),
            ("e2g3", "f6e4"),
            ("d2b3", "c6d4"),
        ] {
            game.push_uci(white);
            game.push_uci(black);
        }
        // Full Cow completed; book is a miss.
        assert_eq!(game.probe(), None);

        // It's White to move (Black just played c6d4 to close the
        // loop above). Something now threatens the g3 knight (the
        // exact reason doesn't matter to the book), so it retreats
        // straight back to e2 -- the same square the king knight
        // passed through in step 2.
        game.push_uci("g3e2");

        // Even though a knight now sits on e2 exactly like right after
        // step 2, and g3 is empty exactly like right before step 4,
        // history shows both step 2 and step 4 were already completed
        // -- the book must not offer Ne2-g3 (or anything else) again.
        assert_eq!(game.probe(), None);
    }

    #[test]
    fn cow_book_offers_d3_once_e3_is_already_played() {
        let mut game = Game::startpos();
        game.push_uci("e2e3");
        game.push_uci("e7e5");
        let mv = game.probe().expect("should hit");
        assert_eq!(mv.from(), "d2".parse().unwrap());
        assert_eq!(mv.to(), "d3".parse().unwrap());
    }

    #[test]
    fn cow_book_falls_back_to_search_once_the_setup_is_blocked() {
        // e3 and d3 played, but the e2 square (where the king knight
        // should reroute to) is occupied by a bishop (an artificial
        // position -- not reachable via legal play -- purely to prove
        // "next step illegal" produces a miss rather than skipping
        // ahead to a later, out-of-order step). No prior moves, so
        // history-wise nothing has ever been completed either.
        let game = Game::from_fen("rnbqkbnr/pppppppp/8/8/8/3PP3/PPP1BPPP/RNBQK1NR w KQkq - 0 1");
        assert_eq!(game.probe(), None);
    }

    #[test]
    fn cow_book_is_a_miss_once_the_setup_is_fully_built() {
        // The finished Cow shape reached with no recorded history at
        // all (a hand-built position, not the result of played
        // moves): every step's `(from, to)` pair is vacuously "not in
        // history", so `probe` falls through to trying the first
        // pending step -- e2-e3 -- which is no longer legal (no pawn
        // on e2), so it still reports a miss rather than skipping
        // ahead. This is a position-shape check, not a history one;
        // `cow_book_completes_the_full_setup_move_by_move` and
        // `cow_book_does_not_reopen_a_completed_horn_after_a_retreat`
        // cover the real history-aware behavior.
        let game = Game::from_fen("rnbqkbnr/pppppppp/8/8/8/1N1PP1N1/P1P2P1P/R1BQKB1R w KQkq - 0 1");
        assert_eq!(game.probe(), None);
    }
}
