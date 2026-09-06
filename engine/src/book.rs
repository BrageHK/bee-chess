//! Opening books: a cheap, position-keyed lookup Bee can consult
//! *before* search, so it doesn't burn its own thinking time re-deriving
//! well-known opening moves and doesn't need to search from scratch
//! before showing any real chess understanding at all.
//!
//! `OpeningBook::probe` takes a `&Position`, not a Zobrist hash or ply
//! count: keying by the actual position (not move sequence) is what
//! makes a book work correctly across transpositions -- two different
//! move orders reaching the same position get the same answer -- and
//! keying by position rather than a raw hash keeps a future real
//! book's on-disk key scheme entirely its own implementation detail,
//! never something the engine or this trait needs to agree on ahead of
//! time (see `CowOpeningBook`'s own docs for why this first
//! implementation doesn't even need a hash at all).
//!
//! This first slice deliberately implements only the smallest useful
//! vertical slice through the whole architecture: the trait, a `NoBook`
//! null implementation, and `CowOpeningBook`, a joke-but-real opening
//! (see its own docs). No `BookSelector` (multiple weighted
//! candidates), no Polyglot/file-backed book, no statistics -- those
//! are real follow-ups once this seam has proven itself, not
//! prerequisites for it.

use crate::chess::{Color, Move, MoveFlag, PieceKind, Position, Square};

/// Looks up a known-good move for `position`, if any. Implementations
/// must return a currently *legal* move or `None` -- the caller
/// (`Engine::search`/`search_for_time`) still doesn't re-validate a
/// `Some` result against `generate_legal_moves` beyond what a specific
/// implementation's own docs promise, so an implementation that can't
/// guarantee legality (e.g corrupt/stale on-disk data, once a
/// file-backed book exists) must probe legality itself and return
/// `None` rather than risk playing an illegal move.
///
/// A book miss (`None`) is always a completely ordinary outcome, not
/// an error -- see `NoBook`.
pub trait OpeningBook: Send + Sync {
    fn probe(&self, position: &Position) -> Option<Move>;
}

/// The null opening book: always a miss. Used whenever `OwnBook`/
/// `OpeningBook` is configured off, so `Engine` always has a concrete
/// book to consult rather than needing an `Option<Box<dyn
/// OpeningBook>>` and a branch at every call site -- "no book" is just
/// another `OpeningBook`, not a special case.
pub struct NoBook;

impl OpeningBook for NoBook {
    fn probe(&self, _position: &Position) -> Option<Move> {
        None
    }
}

/// The Cow: pawns to d3/e3, knights rerouted to the "horns" on b3/g3
/// (via d2/e2), completely irrespective of what the opponent does --
/// not the Hippopotamus (which fianchettoes both bishops behind a
/// king-side pawn triangle instead). It's a real, if eccentric and
/// objectively passive, setup: a good first opening book precisely
/// *because* it's simple enough to encode as "which setup move is
/// still pending" rather than needing a real position-keyed database.
///
/// The Cow's setup order for one side, adjusted for which pieces have
/// already been played: e-pawn to e3, d-pawn to d3, king knight to e2,
/// queen knight to d2, then the knight on e2 continues to g3 and the
/// knight on d2 continues to b3. `probe` walks this list and returns
/// the first step that (a) hasn't happened yet and (b) is currently a
/// legal move.
///
/// "Hasn't happened yet" is checked by looking for the *expected piece
/// kind* still sitting on the step's `from` square -- not just whether
/// `from` is occupied at all. That distinction matters here
/// specifically because two steps share a square: the e-pawn vacates
/// e2 (step 0) and the king's knight later arrives on e2 (step 2) to
/// continue on to g3 (step 4). Checking bare occupancy would read "a
/// piece is on e2" once the knight gets there and wrongly conclude
/// step 0 (the pawn push) still needs playing, sending `probe` off to
/// try an already-completed (and by then illegal) pawn move instead of
/// reaching the real pending step. Checking "is there still a *pawn*
/// on e2" instead correctly reads that step as done and moves on. This
/// is still genuinely keyed by the position's actual piece placement,
/// not a move counter -- just checking placement precisely enough to
/// handle the setup's own square reuse.
///
/// If the pending step isn't legal right now (the opponent is doing
/// something that makes it impossible, e.g. `e3` blocked or the
/// knight's target square defended in a way that matters, or,
/// plainly, it's simply not this side's move), this returns `None`
/// rather than forcing the setup through, and `Engine` falls back to
/// a normal search -- the Cow setup is a starting point Bee commits to
/// only while it stays reasonable, never a script it plays blindly.
///
/// Symmetric for both colors: the shape is mirrored (rank 2->3 for
/// White becomes rank 7->6 for Black, etc.) via `Color`-relative
/// squares, computed once per `probe` call rather than duplicated as
/// two hardcoded move lists.
pub struct CowOpeningBook;

impl OpeningBook for CowOpeningBook {
    fn probe(&self, position: &Position) -> Option<Move> {
        let side = position.side_to_move();
        let legal_moves = position.generate_legal_moves();

        for step in cow_setup_steps(side) {
            // Already played this step (a piece of `step.piece` no
            // longer sits on `step.from`, having moved to `step.to` or
            // been captured) -- move on to the next one rather than
            // getting stuck offering the same move forever. See this
            // book's docs on why this checks piece *kind*, not just
            // occupancy: e2 is both the e-pawn's start and the king
            // knight's later stop, so occupancy alone can't tell two
            // different steps apart.
            let piece_still_pending = position
                .piece_at(step.from)
                .is_some_and(|piece| piece.kind == step.piece && piece.color == side);
            if !piece_still_pending {
                continue;
            }
            let mv = Move::new(step.from, step.to, MoveFlag::Quiet);
            if legal_moves.contains(&mv) {
                return Some(mv);
            }
            // The next pending step isn't legal right now (blocked, or
            // the expected piece has been captured/promoted away
            // without technically "moving" in a way the check above
            // would catch). Rather than skipping ahead to a later step
            // out of setup order (which could make an
            // already-questionable setup actively unsound), treat this
            // as a book miss.
            return None;
        }

        // Every setup step has already been played -- the Cow is
        // fully built, nothing left for this book to offer.
        None
    }
}

/// One pending step of the Cow setup: move the piece of kind `piece`
/// currently expected on `from` to `to`. See `CowOpeningBook`'s docs.
struct CowSetupStep {
    from: Square,
    to: Square,
    piece: PieceKind,
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
            piece: PieceKind::Pawn,
        }, // e2-e3 / e7-e6
        CowSetupStep {
            from: sq(3, rank(1)),
            to: sq(3, rank(2)),
            piece: PieceKind::Pawn,
        }, // d2-d3 / d7-d6
        CowSetupStep {
            from: sq(6, rank(0)),
            to: sq(4, rank(1)),
            piece: PieceKind::Knight,
        }, // Ng1-e2 / Ng8-e7
        CowSetupStep {
            from: sq(1, rank(0)),
            to: sq(3, rank(1)),
            piece: PieceKind::Knight,
        }, // Nb1-d2 / Nb8-d7
        CowSetupStep {
            from: sq(4, rank(1)),
            to: sq(6, rank(2)),
            piece: PieceKind::Knight,
        }, // Ne2-g3 / Ne7-g6
        CowSetupStep {
            from: sq(3, rank(1)),
            to: sq(1, rank(2)),
            piece: PieceKind::Knight,
        }, // Nd2-b3 / Nd7-b6
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chess::Position;

    #[test]
    fn no_book_always_misses() {
        assert_eq!(NoBook.probe(&Position::startpos()), None);
    }

    #[test]
    fn cow_book_plays_e3_from_the_start_position() {
        let mv = CowOpeningBook
            .probe(&Position::startpos())
            .expect("should hit");
        assert_eq!(mv.from(), "e2".parse().unwrap());
        assert_eq!(mv.to(), "e3".parse().unwrap());
    }

    #[test]
    fn cow_book_continues_the_setup_after_e3() {
        let position =
            Position::from_fen("rnbqkbnr/pppppppp/8/8/8/4P3/PPPP1PPP/RNBQKBNR b KQkq - 0 1")
                .unwrap();
        // Black to move; White has already played e3. Probing from
        // Black's side should offer Black's own first setup step
        // (e7-e6), not react to White's move at all -- the book is
        // symmetric and per-side.
        let mv = CowOpeningBook.probe(&position).expect("should hit");
        assert_eq!(mv.from(), "e7".parse().unwrap());
        assert_eq!(mv.to(), "e6".parse().unwrap());
    }

    #[test]
    fn cow_book_works_the_same_regardless_of_move_order_into_the_same_position() {
        // 1.e3 Nf6 2.d3, versus 1.d3 Nf6 2.e3 -- two different move
        // sequences (same reply from Black both times, since a
        // position needs someone to move on both sides) reaching the
        // exact same resulting position. Since `probe` is keyed by
        // position (piece placement), not by which moves got there,
        // both must offer the same next step (a knight reroute),
        // proving this isn't secretly keyed by ply count or move
        // history.
        fn find(position: &Position, from: &str, to: &str) -> Move {
            position
                .generate_legal_moves()
                .into_iter()
                .find(|mv| mv.from() == from.parse().unwrap() && mv.to() == to.parse().unwrap())
                .unwrap_or_else(|| panic!("{from}{to} should be legal here"))
        }

        let e3_nf6_d3 = {
            let mut position = Position::startpos();
            position.make_move(find(&position, "e2", "e3"));
            position.make_move(find(&position, "g8", "f6"));
            position.make_move(find(&position, "d2", "d3"));
            position
        };
        let d3_nf6_e3 = {
            let mut position = Position::startpos();
            position.make_move(find(&position, "d2", "d3"));
            position.make_move(find(&position, "g8", "f6"));
            position.make_move(find(&position, "e2", "e3"));
            position
        };

        assert_eq!(
            e3_nf6_d3.zobrist_hash(),
            d3_nf6_e3.zobrist_hash(),
            "should be the same position"
        );
        assert_eq!(
            CowOpeningBook.probe(&e3_nf6_d3),
            CowOpeningBook.probe(&d3_nf6_e3)
        );
        // After 1.e3 Nf6 2.d3, it's Black to move -- both White pawn
        // steps are already played, so the book offers Black's own
        // first pending step, not White's next one.
        let mv = CowOpeningBook.probe(&e3_nf6_d3).expect("should hit");
        assert_eq!(mv.from(), "e7".parse().unwrap());
        assert_eq!(mv.to(), "e6".parse().unwrap());
    }

    #[test]
    fn cow_book_offers_nb1d2_once_the_king_knight_has_already_rerouted_to_e2() {
        // Regression test: e2 is both the e-pawn's vacated square
        // (step 0) and the king knight's later stop (step 2, on its
        // way to g3 in step 4). A real game (1.e3 e5 2.d3 d5 3.Ne2
        // Nf6) reaching this exact position used to make `probe`
        // wrongly conclude "step 0 (e2-e3) still pending" the instant
        // a knight sat on e2 -- occupancy alone can't distinguish "the
        // pawn never left" from "a knight arrived after" -- try the
        // now-illegal pawn push, and give up instead of reaching the
        // real pending step (Nb1-d2).
        let position =
            Position::from_fen("rnbqkb1r/ppp2ppp/5n2/3pp3/8/3PP3/PPP1NPPP/RNBQKB1R w KQkq - 0 4")
                .unwrap();

        let mv = CowOpeningBook.probe(&position).expect("should hit");

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
        let mut position = Position::startpos();
        let black_replies = ["e7e5", "d7d5", "g8f6", "b8c6", "f6e4", "c6d4"];
        let mut white_moves = Vec::new();

        for black_reply in black_replies {
            let white_mv = CowOpeningBook
                .probe(&position)
                .unwrap_or_else(|| panic!("book should still have a move; got to {white_moves:?}"));
            white_moves.push(format!("{}{}", white_mv.from(), white_mv.to()));
            position.make_move(white_mv);

            let black_mv = position
                .generate_legal_moves()
                .into_iter()
                .find(|mv| format!("{}{}", mv.from(), mv.to()) == black_reply)
                .unwrap_or_else(|| panic!("{black_reply} should be legal"));
            position.make_move(black_mv);
        }

        assert_eq!(
            white_moves,
            vec!["e2e3", "d2d3", "g1e2", "b1d2", "e2g3", "d2b3"]
        );
        // The setup is now complete -- nothing left for the book to
        // offer, Engine would fall back to a real search from here.
        assert_eq!(CowOpeningBook.probe(&position), None);
    }

    #[test]
    fn cow_book_offers_d3_once_e3_is_already_played() {
        let position =
            Position::from_fen("rnbqkbnr/pppppppp/8/8/8/4P3/PPPP1PPP/RNBQKBNR w KQkq - 0 1")
                .unwrap();
        let mv = CowOpeningBook.probe(&position).expect("should hit");
        assert_eq!(mv.from(), "d2".parse().unwrap());
        assert_eq!(mv.to(), "d3".parse().unwrap());
    }

    #[test]
    fn cow_book_falls_back_to_search_once_the_setup_is_blocked() {
        // e3 and d3 played, but the e2 square (where the king knight
        // should reroute to) is occupied by a bishop (an artificial
        // position -- not reachable via legal play -- purely to prove
        // "next step illegal" produces a miss rather than skipping
        // ahead to a later, out-of-order step).
        let position =
            Position::from_fen("rnbqkbnr/pppppppp/8/8/8/3PP3/PPP1BPPP/RNBQK1NR w KQkq - 0 1")
                .unwrap();
        assert_eq!(CowOpeningBook.probe(&position), None);
    }

    #[test]
    fn cow_book_is_a_miss_once_the_setup_is_fully_built() {
        // The finished Cow shape: pawns on d3/e3, knights on b3/g3,
        // b1/g1/d2/e2 all vacated (a hand-built position, not
        // necessarily one reachable via legal play in this exact move
        // count, but a faithful "setup complete" board) -- nothing
        // left for this book to offer.
        let position =
            Position::from_fen("rnbqkbnr/pppppppp/8/8/8/1N1PP1N1/P1P2P1P/R1BQKB1R w KQkq - 0 1")
                .unwrap();
        assert_eq!(CowOpeningBook.probe(&position), None);
    }
}
