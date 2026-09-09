//! [`BookPositionKey`]: the position identity an `ExperienceBook` is
//! keyed by. Deliberately its own type, computed independently of
//! `bee_chess_core::Position::zobrist_hash` even though today's
//! algorithm happens to be identical -- see the module docs below for
//! why sharing the function itself would be a mistake.

use bee_chess_core::{Color, Piece, PieceKind, Position, Square};

/// The book-key scheme version this build writes and reads. Bumped
/// whenever the fields folded into the key change, so an artifact built
/// under an old scheme is never silently misread as a newer one (or
/// vice versa) -- see `book::format`'s header, which stores this
/// alongside the artifact's own format version.
pub const KEY_SCHEME_VERSION: u16 = 1;

/// A book's notion of "this is the same position", independent of
/// `Position::zobrist_hash`.
///
/// `zobrist_hash`'s own module docs already flag it as headed toward
/// transposition-table use ("eventually, transposition table
/// indexing"); once that happens it's a search-internal detail that can
/// reasonably change for speed or collision-resistance reasons having
/// nothing to do with book semantics. A book artifact, by contrast, is
/// meant to survive many engine versions unchanged -- baked into a
/// binary today, still readable years later. Keying it off the same
/// function `alpha_beta.rs` uses for repetition detection would silently
/// couple "what a book key means" to "how search happens to hash
/// positions today", so this computes its own hash from scratch instead.
///
/// The contract (deliberately identical in shape to `zobrist_hash`'s
/// today, but that's this module's choice to keep, not an inherited
/// one): every occupied square (piece kind + color), side to move,
/// castling rights, and en passant square. Explicitly **not** included:
/// halfmove clock or fullmove number -- two positions that are the same
/// position for repetition/opening-book purposes but reached at
/// different move-count "depths" (e.g. via a different game, or a
/// transposition) must produce the same key.
#[must_use]
pub fn book_position_key(position: &Position) -> u64 {
    let mut hash = 0u64;

    for index in 0..Square::COUNT as u8 {
        let square = Square::new(index);
        if let Some(piece) = position.piece_at(square) {
            hash ^= piece_square_key(piece, square);
        }
    }

    if position.side_to_move() == Color::Black {
        hash ^= SIDE_TO_MOVE_KEY;
    }

    let rights = position.castling_rights();
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

    if let Some(square) = position.en_passant_square() {
        hash ^= EN_PASSANT_KEYS[square.index() as usize];
    }

    hash
}

/// One splitmix64 step -- see `bee_chess_core`'s `zobrist` module for
/// the algorithm this mirrors. Duplicated deliberately (see this
/// module's docs) rather than imported, since `bee-chess-core`'s own
/// splitmix64/table-filling helpers are private to that crate.
const fn splitmix64(seed: u64) -> (u64, u64) {
    let seed = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = seed;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^= z >> 31;
    (z, seed)
}

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

// Different seeds from `bee_chess_core::zobrist`'s on purpose: nothing
// should ever depend on these two hash spaces colliding or lining up,
// and using different constants makes that impossible by construction
// rather than "true so far, but only because the seeds happen to match".
const PIECE_SQUARE_SEED: u64 = 0xB00C_1357_9BDF_2468;
const SIDE_TO_MOVE_SEED: u64 = 0xB00C_C0FF_EE00_D15E;
const CASTLING_SEED: u64 = 0xB00C_FEED_FACE_CAFE;
const EN_PASSANT_SEED: u64 = 0xB00C_DEAD_BEEF_1234;

const PIECE_SQUARE_KEYS: [u64; 6 * 2 * Square::COUNT] = fill(PIECE_SQUARE_SEED);
const SIDE_TO_MOVE_KEY: u64 = {
    let (value, _) = splitmix64(SIDE_TO_MOVE_SEED);
    value
};
const CASTLING_KEYS: [u64; 4] = fill(CASTLING_SEED);
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

fn piece_square_key(piece: Piece, square: Square) -> u64 {
    PIECE_SQUARE_KEYS[piece_square_index(piece, square)]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identical_positions_key_identically() {
        assert_eq!(
            book_position_key(&Position::startpos()),
            book_position_key(&Position::startpos())
        );
    }

    #[test]
    fn different_positions_key_differently() {
        assert_ne!(
            book_position_key(&Position::startpos()),
            book_position_key(&Position::empty())
        );
    }

    #[test]
    fn transposing_to_the_same_position_keys_identically() {
        use bee_chess_core::{Move, MoveFlag};

        let mut via_a = Position::startpos();
        via_a.make_move(Move::new(
            Square::from_file_rank(6, 0),
            Square::from_file_rank(5, 2),
            MoveFlag::Quiet,
        ));
        via_a.make_move(Move::new(
            Square::from_file_rank(6, 7),
            Square::from_file_rank(5, 5),
            MoveFlag::Quiet,
        ));

        let mut via_b = Position::startpos();
        via_b.make_move(Move::new(
            Square::from_file_rank(6, 7),
            Square::from_file_rank(5, 5),
            MoveFlag::Quiet,
        ));
        via_b.make_move(Move::new(
            Square::from_file_rank(6, 0),
            Square::from_file_rank(5, 2),
            MoveFlag::Quiet,
        ));

        assert_eq!(book_position_key(&via_a), book_position_key(&via_b));
    }

    #[test]
    fn halfmove_clock_and_fullmove_number_do_not_affect_the_key() {
        let mut a = Position::startpos();
        let mut b = Position::startpos();
        a.set_halfmove_clock(17);
        a.set_fullmove_number(9);
        b.set_halfmove_clock(0);
        b.set_fullmove_number(1);

        assert_eq!(book_position_key(&a), book_position_key(&b));
    }

    #[test]
    fn this_key_space_is_independent_of_the_engine_s_zobrist_hash() {
        // Different seeds by construction (see this module's docs) --
        // confirm they don't happen to collide on the start position.
        assert_ne!(
            book_position_key(&Position::startpos()),
            Position::startpos().zobrist_hash()
        );
    }
}
