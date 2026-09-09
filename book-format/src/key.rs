//! [`BookPositionKey`]: the position identity an `ExperienceBook` is
//! keyed by. Deliberately its own type, computed independently of
//! `bee_chess_core::Position::zobrist_hash` even though today's
//! algorithm happens to be identical -- see the module docs below for
//! why sharing the function itself would be a mistake.

use bee_chess_core::{Color, Piece, PieceKind, Position, Square};

/// The book-key scheme version this build writes and reads. Bumped
/// whenever the fields folded into the key change, so an artifact built
/// under an old scheme is never silently misread as a newer one (or
/// vice versa) -- see `crate::format`'s header, which stores this
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
    fn side_to_move_affects_the_key() {
        use bee_chess_core::Color;

        let mut white_to_move = Position::startpos();
        white_to_move.set_side_to_move(Color::White);
        let mut black_to_move = Position::startpos();
        black_to_move.set_side_to_move(Color::Black);

        assert_ne!(
            book_position_key(&white_to_move),
            book_position_key(&black_to_move)
        );
    }

    #[test]
    fn each_castling_right_independently_affects_the_key() {
        use bee_chess_core::CastlingRights;

        let base = {
            let mut p = Position::startpos();
            p.set_castling_rights(CastlingRights::none());
            p
        };
        let base_key = book_position_key(&base);

        let with_right = |rights: CastlingRights| {
            let mut p = base.clone();
            p.set_castling_rights(rights);
            book_position_key(&p)
        };

        let white_kingside = with_right(CastlingRights {
            white_kingside: true,
            ..CastlingRights::none()
        });
        let white_queenside = with_right(CastlingRights {
            white_queenside: true,
            ..CastlingRights::none()
        });
        let black_kingside = with_right(CastlingRights {
            black_kingside: true,
            ..CastlingRights::none()
        });
        let black_queenside = with_right(CastlingRights {
            black_queenside: true,
            ..CastlingRights::none()
        });

        // Every right, on its own, must differ from having none...
        assert_ne!(base_key, white_kingside);
        assert_ne!(base_key, white_queenside);
        assert_ne!(base_key, black_kingside);
        assert_ne!(base_key, black_queenside);
        // ...and from each other -- otherwise two of these would be
        // indistinguishable positions for a book that cares about
        // castling rights (e.g. "kingside still possible" vs.
        // "queenside still possible" genuinely change what's sound).
        let all = [
            white_kingside,
            white_queenside,
            black_kingside,
            black_queenside,
        ];
        for i in 0..all.len() {
            for j in (i + 1)..all.len() {
                assert_ne!(all[i], all[j], "rights {i} and {j} collided");
            }
        }

        assert_eq!(
            with_right(CastlingRights::all()),
            with_right(CastlingRights::all()),
            "the same combination of rights must still key identically"
        );
    }

    #[test]
    fn a_usable_en_passant_square_affects_the_key() {
        // White has just played d2-d4; e5 pawn can capture en passant
        // on d3. Same board otherwise, with and without the ep square
        // recorded, must key differently.
        let mut with_ep = Position::from_fen("4k3/8/8/4p3/3P4/8/8/4K3 b - d3 0 1").unwrap();
        let mut without_ep = with_ep.clone();
        with_ep.set_en_passant_square(Some(Square::from_file_rank(3, 2)));
        without_ep.set_en_passant_square(None);

        assert_ne!(book_position_key(&with_ep), book_position_key(&without_ep));
    }

    #[test]
    fn a_different_en_passant_file_affects_the_key() {
        let mut on_d_file = Position::empty();
        on_d_file.set_en_passant_square(Some(Square::from_file_rank(3, 2)));
        let mut on_e_file = Position::empty();
        on_e_file.set_en_passant_square(Some(Square::from_file_rank(4, 2)));

        assert_ne!(book_position_key(&on_d_file), book_position_key(&on_e_file));
    }

    #[test]
    fn moving_any_single_piece_changes_the_key() {
        // A cheap proxy for "every occupied square actually
        // contributes to the hash": displacing each of White's back-
        // rank pieces one at a time must change the key each time,
        // relative to the unmodified start position.
        let base_key = book_position_key(&Position::startpos());
        for file in 0..8u8 {
            let mut moved = Position::startpos();
            let piece = moved
                .piece_at(Square::from_file_rank(file, 0))
                .expect("back rank is fully occupied at the start position");
            moved.set_piece(Square::from_file_rank(file, 0), None);
            moved.set_piece(Square::from_file_rank(file, 3), Some(piece));
            assert_ne!(
                book_position_key(&moved),
                base_key,
                "moving the piece on file {file} should change the key"
            );
        }
    }
}
