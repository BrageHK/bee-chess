//! Zobrist hashing: a single `u64` fingerprint of a `Position`, used for
//! repetition detection (three identical positions => draw) and,
//! eventually, transposition table indexing (#6's remaining TT work).
//!
//! The hash is deliberately **not** stored as a `Position` field: it is
//! computed fresh by `Position::zobrist_hash()` from the board, side to
//! move, castling rights, and en passant square. This trades a little
//! CPU (an O(64) scan over the board, only paid when something actually
//! needs the hash -- once per node for a repetition check or TT probe,
//! not on every `make_move`/`unmake_move`) for a hash that can never
//! silently desync from the position it describes, and for leaving the
//! existing low-level `Position` setters (`set_piece`, `set_side_to_move`,
//! etc., several of them `const fn`) untouched. An incrementally
//! maintained hash (XORed in as each setter runs) is the standard
//! approach real engines use and is a reasonable later optimization if
//! this ever shows up as a hot path -- it isn't a correctness concern
//! either way, just speed, so it's deferred per this milestone's
//! "correctness first" approach elsewhere in the codebase.
//!
//! Random constants come from a fixed-seed splitmix64 generator run at
//! compile time (`const fn`, no runtime cost, no external RNG
//! dependency) -- deterministic across builds/platforms, which matters
//! since this hash is meant to be internally consistent for one running
//! process, not to match another engine's or another run's hash values
//! bit-for-bit.
//!
//! En passant is hashed by target square whenever one is set, not only
//! when a pawn could actually capture there. Some engines only hash it
//! in the latter, stricter case, to avoid two positions that differ
//! only in an unusable en passant square hashing differently (and so
//! never being recognized as "the same position" for repetition
//! purposes). This engine hashes the simpler way for now: it costs at
//! most a slightly too-conservative repetition/TT match (treating two
//! truly-identical-to-a-player positions as different because one has a
//! dead en passant square and the other doesn't), never an incorrect
//! one, so it's a reasonable simplification to defer tightening later.

use super::piece::{Color, Piece, PieceKind};
use super::position::Position;
use super::square::Square;

/// One splitmix64 step: cheap, well-distributed, and (unlike most of
/// Rust's `rand` ecosystem) usable in a `const fn`, which is what lets
/// every table below be built once at compile time instead of on first
/// use at runtime.
const fn splitmix64(seed: u64) -> (u64, u64) {
    let seed = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = seed;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^= z >> 31;
    (z, seed)
}

/// Fills `table` with successive splitmix64 outputs from a fixed seed.
/// A tiny hand-rolled const-context loop (no iterators/closures, which
/// aren't available in `const fn`) -- this only ever runs at compile
/// time, so it doesn't need to be idiomatic runtime Rust.
const fn fill<const N: usize>(mut seed: u64) -> [u64; N] {
    let mut table = [0u64; N];
    let mut i = 0;
    while i < N {
        let (value, next_seed) = splitmix64(seed);
        table[i] = value;
        seed = next_seed;
        i += 1;
    }
    table
}

const PIECE_SQUARE_SEED: u64 = 0x1357_9BDF_2468_ACE0;
const SIDE_TO_MOVE_SEED: u64 = 0xC0FF_EE00_D15E_A5E5;
const CASTLING_SEED: u64 = 0xFEED_FACE_CAFE_BEEF;
const EN_PASSANT_SEED: u64 = 0xDEAD_BEEF_1234_5678;

/// One random value per (piece kind, color, square) combination: 6
/// kinds x 2 colors x 64 squares. Indexed by `piece_square_index`.
const PIECE_SQUARE_KEYS: [u64; 6 * 2 * Square::COUNT] = fill(PIECE_SQUARE_SEED);

/// XORed in whenever it's Black to move (White contributes nothing --
/// XORing this same value back out when it becomes White's turn again
/// is equivalent to just not XORing anything for White).
const SIDE_TO_MOVE_KEY: u64 = {
    let (value, _) = splitmix64(SIDE_TO_MOVE_SEED);
    value
};

/// One random value per castling right (white/black x kingside/queenside).
const CASTLING_KEYS: [u64; 4] = fill(CASTLING_SEED);

/// One random value per file, for the en passant target square (see
/// this module's docs on why file-only, not full square, would also be
/// a defensible choice -- full square is used here for simplicity).
const EN_PASSANT_KEYS: [u64; Square::COUNT] = fill(EN_PASSANT_SEED);

const fn piece_square_index(piece: Piece, square: Square) -> usize {
    let kind_index = match piece.kind {
        PieceKind::Pawn => 0,
        PieceKind::Knight => 1,
        PieceKind::Bishop => 2,
        PieceKind::Rook => 3,
        PieceKind::Queen => 4,
        PieceKind::King => 5,
    };
    let color_index = match piece.color {
        Color::White => 0,
        Color::Black => 1,
    };
    (kind_index * 2 + color_index) * Square::COUNT + square.index() as usize
}

impl Position {
    /// Computes this position's Zobrist hash from scratch: every
    /// occupied square, side to move, castling rights, and en passant
    /// square. See the module docs for why this isn't incrementally
    /// cached on `Position` itself.
    #[must_use]
    pub fn zobrist_hash(&self) -> u64 {
        let mut hash = 0u64;

        for index in 0..Square::COUNT as u8 {
            let square = Square::new(index);
            if let Some(piece) = self.piece_at(square) {
                hash ^= PIECE_SQUARE_KEYS[piece_square_index(piece, square)];
            }
        }

        if self.side_to_move() == Color::Black {
            hash ^= SIDE_TO_MOVE_KEY;
        }

        let rights = self.castling_rights();
        if rights.white_kingside {
            hash ^= CASTLING_KEYS[0];
        }
        if rights.white_queenside {
            hash ^= CASTLING_KEYS[1];
        }
        if rights.black_kingside {
            hash ^= CASTLING_KEYS[2];
        }
        if rights.black_queenside {
            hash ^= CASTLING_KEYS[3];
        }

        if let Some(square) = self.en_passant_square() {
            hash ^= EN_PASSANT_KEYS[square.index() as usize];
        }

        hash
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chess::{Move, MoveFlag};

    #[test]
    fn identical_positions_hash_identically() {
        let a = Position::startpos();
        let b = Position::startpos();
        assert_eq!(a.zobrist_hash(), b.zobrist_hash());
    }

    #[test]
    fn different_positions_hash_differently() {
        let startpos = Position::startpos();
        let empty = Position::empty();
        assert_ne!(startpos.zobrist_hash(), empty.zobrist_hash());
    }

    #[test]
    fn side_to_move_affects_the_hash() {
        let mut white_to_move = Position::startpos();
        white_to_move.set_side_to_move(Color::White);
        let mut black_to_move = Position::startpos();
        black_to_move.set_side_to_move(Color::Black);

        assert_ne!(white_to_move.zobrist_hash(), black_to_move.zobrist_hash());
    }

    #[test]
    fn castling_rights_affect_the_hash() {
        use crate::chess::CastlingRights;

        let mut with_rights = Position::startpos();
        with_rights.set_castling_rights(CastlingRights::all());
        let mut without_rights = Position::startpos();
        without_rights.set_castling_rights(CastlingRights::none());

        assert_ne!(with_rights.zobrist_hash(), without_rights.zobrist_hash());
    }

    #[test]
    fn en_passant_square_affects_the_hash() {
        let mut with_ep = Position::startpos();
        with_ep.set_en_passant_square(Some(Square::from_file_rank(4, 2)));
        let mut without_ep = Position::startpos();
        without_ep.set_en_passant_square(None);

        assert_ne!(with_ep.zobrist_hash(), without_ep.zobrist_hash());
    }

    #[test]
    fn make_then_unmake_restores_the_original_hash() {
        // The hash isn't incrementally maintained (see module docs), so
        // this is really just re-confirming make/unmake fully restores
        // position equality -- but it's the property search/TT/
        // repetition code will actually depend on, so it's worth
        // asserting directly rather than only indirectly through
        // Position's PartialEq.
        let mut position = Position::startpos();
        let before_hash = position.zobrist_hash();

        let mv = Move::new(
            Square::from_file_rank(4, 1),
            Square::from_file_rank(4, 3),
            MoveFlag::DoublePawnPush,
        );
        let undo = position.make_move(mv);
        assert_ne!(position.zobrist_hash(), before_hash);

        position.unmake_move(mv, undo);
        assert_eq!(position.zobrist_hash(), before_hash);
    }

    #[test]
    fn reaching_the_same_position_by_different_move_orders_hashes_identically() {
        // 1. Nf3 Nf6  vs  1. Nf3 Nf6 (transposed via a different pair of
        // knight moves reaching the identical resulting position) --
        // the classic repetition/transposition-table scenario: the hash
        // must depend only on the resulting position, not on the path
        // taken to reach it.
        let mut via_knights_first = Position::startpos();
        via_knights_first.make_move(Move::new(
            Square::from_file_rank(6, 0), // g1
            Square::from_file_rank(5, 2), // f3
            MoveFlag::Quiet,
        ));
        via_knights_first.make_move(Move::new(
            Square::from_file_rank(6, 7), // g8
            Square::from_file_rank(5, 5), // f6
            MoveFlag::Quiet,
        ));

        let mut via_other_order = Position::startpos();
        via_other_order.make_move(Move::new(
            Square::from_file_rank(6, 7),
            Square::from_file_rank(5, 5),
            MoveFlag::Quiet,
        ));
        via_other_order.make_move(Move::new(
            Square::from_file_rank(6, 0),
            Square::from_file_rank(5, 2),
            MoveFlag::Quiet,
        ));

        assert_eq!(
            via_knights_first.zobrist_hash(),
            via_other_order.zobrist_hash()
        );
    }
}
