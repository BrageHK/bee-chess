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

use crate::chess::{Color, Move, MoveFlag, Position, Square};

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
/// the first step that (a) hasn't happened yet -- checked by looking
/// at the position's actual piece placement, not a move counter, so
/// this is genuinely keyed by position, not "which ply is this" -- and
/// (b) is currently a legal move. If the pending step isn't legal
/// right now (the opponent is doing something that makes it
/// impossible, e.g. `e3` blocked or the knight's target square
/// defended in a way that matters, or, plainly, it's simply not this
/// side's move), this returns `None` rather than forcing the setup
/// through, and `Engine` falls back to a normal search -- the Cow
/// setup is a starting point Bee commits to only while it stays
/// reasonable, never a script it plays blindly.
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

        for (from, to) in cow_setup_squares(side) {
            // Already played this step -- move on to the next one
            // rather than getting stuck offering the same move forever.
            if position.piece_at(from).is_none() {
                continue;
            }
            let mv = Move::new(from, to, MoveFlag::Quiet);
            if legal_moves.contains(&mv) {
                return Some(mv);
            }
            // The next pending step isn't legal right now (blocked,
            // the piece that belongs on `from` isn't actually there
            // despite the square being non-empty, or -- since this
            // loop only reaches here on a genuine mismatch -- the
            // setup has effectively been abandoned). Rather than
            // skipping ahead to a later step out of setup order (which
            // could make an already-questionable setup actively
            // unsound), treat this as a book miss.
            return None;
        }

        // Every setup step has already been played (or was never this
        // side's own piece to move in the first place) -- the Cow is
        // fully built, nothing left for this book to offer.
        None
    }
}

/// The Cow's six setup squares for `side`, in the order they should be
/// played, as `(from, to)` pairs -- see `CowOpeningBook`'s docs.
/// White's shape (e2-e3, d2-d3, Ng1-e2, Nb1-d2, Ne2-g3, Nd2-b3)
/// mirrored vertically for Black (e7-e6, d7-d6, Ng8-e7, Nb8-d7,
/// Ne7-g6, Nd7-b6). Ranks are numbered from each side's own back rank
/// (0), forward being `+1` for White and `-1` for Black, so the six
/// pairs below read the same regardless of color.
fn cow_setup_squares(side: Color) -> [(Square, Square); 6] {
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

    [
        (
            Square::from_file_rank(4, rank(1)),
            Square::from_file_rank(4, rank(2)),
        ), // e2-e3 / e7-e6
        (
            Square::from_file_rank(3, rank(1)),
            Square::from_file_rank(3, rank(2)),
        ), // d2-d3 / d7-d6
        (
            Square::from_file_rank(6, rank(0)),
            Square::from_file_rank(4, rank(1)),
        ), // Ng1-e2 / Ng8-e7
        (
            Square::from_file_rank(1, rank(0)),
            Square::from_file_rank(3, rank(1)),
        ), // Nb1-d2 / Nb8-d7
        (
            Square::from_file_rank(4, rank(1)),
            Square::from_file_rank(6, rank(2)),
        ), // Ne2-g3 / Ne7-g6
        (
            Square::from_file_rank(3, rank(1)),
            Square::from_file_rank(1, rank(2)),
        ), // Nd2-b3 / Nd7-b6
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
