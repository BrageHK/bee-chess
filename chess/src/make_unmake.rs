//! Make/unmake move application.
//!
//! `Position::make_move` mutates the position in place and returns an
//! `Undo` record; `Position::unmake_move` uses that record to restore
//! the exact prior state. This avoids cloning the whole `Position` at
//! every ply, which matters once search/perft is calling this billions
//! of times.
//!
//! `make_move` trusts its caller: it does not itself check legality (or
//! even pseudo-legality) of `mv` against the current position. Move
//! generation (a follow-up milestone step) is responsible for only ever
//! producing moves this function can apply correctly. What `make_move`
//! *does* guarantee is that applying `mv` and then undoing it with the
//! returned `Undo` restores the position to bit-for-bit the same state,
//! including en passant, castling rights, halfmove clock, promotion,
//! and side to move.

use super::castling::CastlingRights;
use super::moves::{Move, MoveFlag};
use super::piece::{Color, Piece, PieceKind};
use super::position::Position;
use super::square::Square;

/// Everything needed to reverse a `make_move` call. Deliberately small:
/// only what cannot be cheaply re-derived from `mv` itself.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Undo {
    /// The piece captured by this move, if any, and the square it was
    /// captured on. For en passant this is the pawn on the pre-capture
    /// square, not `mv.to()` — the capturing pawn lands on an empty
    /// square.
    captured: Option<(Square, Piece)>,
    en_passant_square: Option<Square>,
    castling_rights: CastlingRights,
    halfmove_clock: u32,
    fullmove_number: u32,
}

impl Position {
    /// Applies `mv` to this position, mutating it in place, and returns
    /// an `Undo` record that can later restore the pre-move state via
    /// `unmake_move`.
    ///
    /// # Panics
    ///
    /// Panics if there is no piece on `mv.from()`. Anything generating
    /// moves is expected to only ever produce moves that start on an
    /// occupied square.
    pub fn make_move(&mut self, mv: Move) -> Undo {
        let from = mv.from();
        let to = mv.to();
        let flag = mv.flag();
        let moving_color = self.side_to_move();
        let moving_piece = self
            .piece_at(from)
            .expect("make_move: no piece on from-square");

        let undo = Undo {
            captured: None,
            en_passant_square: self.en_passant_square(),
            castling_rights: self.castling_rights(),
            halfmove_clock: self.halfmove_clock(),
            fullmove_number: self.fullmove_number(),
        };

        // Determine the captured piece (if any) and the square it sits
        // on, before mutating the board. En passant captures a pawn on
        // a different square than `to`.
        let capture_square = if flag == MoveFlag::EnPassant {
            Square::from_file_rank(to.file(), from.rank())
        } else {
            to
        };
        let captured_piece = self.piece_at(capture_square);
        let undo = Undo {
            captured: captured_piece.map(|piece| (capture_square, piece)),
            ..undo
        };

        // Halfmove clock: reset on capture or pawn move, otherwise
        // increment.
        let is_pawn_move = moving_piece.kind == PieceKind::Pawn;
        let is_capture = captured_piece.is_some();
        let new_halfmove_clock = if is_pawn_move || is_capture {
            0
        } else {
            self.halfmove_clock() + 1
        };
        self.set_halfmove_clock(new_halfmove_clock);

        // Move the piece, clearing any captured piece first (en passant
        // clears a square other than `to`).
        if flag == MoveFlag::EnPassant {
            self.set_piece(capture_square, None);
        }
        self.set_piece(from, None);
        let placed_piece = match flag.promotion_kind() {
            Some(promotion_kind) => Piece::new(promotion_kind, moving_color),
            None => moving_piece,
        };
        self.set_piece(to, Some(placed_piece));

        // Castling also moves the rook.
        match flag {
            MoveFlag::CastleKingside => {
                let rank = from.rank();
                let rook_from = Square::from_file_rank(7, rank);
                let rook_to = Square::from_file_rank(5, rank);
                let rook = self.piece_at(rook_from);
                self.set_piece(rook_from, None);
                self.set_piece(rook_to, rook);
            }
            MoveFlag::CastleQueenside => {
                let rank = from.rank();
                let rook_from = Square::from_file_rank(0, rank);
                let rook_to = Square::from_file_rank(3, rank);
                let rook = self.piece_at(rook_from);
                self.set_piece(rook_from, None);
                self.set_piece(rook_to, rook);
            }
            _ => {}
        }

        // New en passant square: only set immediately after a double
        // pawn push, to the square the pawn skipped over.
        let new_en_passant_square = if flag == MoveFlag::DoublePawnPush {
            let skipped_rank = (from.rank() + to.rank()) / 2;
            Some(Square::from_file_rank(from.file(), skipped_rank))
        } else {
            None
        };
        self.set_en_passant_square(new_en_passant_square);

        // Castling rights: lost when a king or rook moves, or when a
        // rook is captured on its home square.
        let mut rights = self.castling_rights();
        update_castling_rights_on_departure(&mut rights, from, Some(moving_piece));
        update_castling_rights_on_departure(&mut rights, capture_square, captured_piece);
        self.set_castling_rights(rights);

        // Side to move and fullmove number.
        self.set_side_to_move(moving_color.opposite());
        if moving_color == Color::Black {
            self.set_fullmove_number(self.fullmove_number() + 1);
        }

        undo
    }

    /// Reverses a previous `make_move(mv)` call using the `Undo` record
    /// it returned, restoring this position to exactly its pre-move
    /// state.
    ///
    /// # Panics
    ///
    /// Panics if there is no piece on `mv.to()`. `undo` must be the
    /// value returned by the matching `make_move(mv)` call on this same
    /// position; passing a mismatched move/undo pair leaves the
    /// position corrupted rather than panicking predictably.
    pub fn unmake_move(&mut self, mv: Move, undo: Undo) {
        let from = mv.from();
        let to = mv.to();
        let flag = mv.flag();

        // The side that made this move is the opposite of the current
        // (post-move) side to move.
        let moving_color = self.side_to_move().opposite();

        let moved_piece = self
            .piece_at(to)
            .expect("unmake_move: no piece on to-square");
        let original_piece = match flag.promotion_kind() {
            Some(_) => Piece::new(PieceKind::Pawn, moving_color),
            None => moved_piece,
        };

        self.set_piece(to, None);
        self.set_piece(from, Some(original_piece));

        match undo.captured {
            Some((square, piece)) => self.set_piece(square, Some(piece)),
            None => {
                // Nothing to restore, but en passant's capture square
                // differs from `to`, which was already cleared above.
            }
        }

        if flag == MoveFlag::CastleKingside {
            let rank = from.rank();
            let rook_to = Square::from_file_rank(5, rank);
            let rook_from = Square::from_file_rank(7, rank);
            let rook = self.piece_at(rook_to);
            self.set_piece(rook_to, None);
            self.set_piece(rook_from, rook);
        } else if flag == MoveFlag::CastleQueenside {
            let rank = from.rank();
            let rook_to = Square::from_file_rank(3, rank);
            let rook_from = Square::from_file_rank(0, rank);
            let rook = self.piece_at(rook_to);
            self.set_piece(rook_to, None);
            self.set_piece(rook_from, rook);
        }

        self.set_en_passant_square(undo.en_passant_square);
        self.set_castling_rights(undo.castling_rights);
        self.set_halfmove_clock(undo.halfmove_clock);
        self.set_fullmove_number(undo.fullmove_number);
        self.set_side_to_move(moving_color);
    }
}

/// Clears the castling right associated with `square` if a king or rook
/// left it (via moving or being captured). `piece` is the piece that
/// was on `square` before this event, if any.
fn update_castling_rights_on_departure(
    rights: &mut CastlingRights,
    square: Square,
    piece: Option<Piece>,
) {
    let Some(piece) = piece else {
        return;
    };

    match (piece.color, piece.kind, square.file(), square.rank()) {
        (Color::White, PieceKind::King, _, 0) => {
            rights.white_kingside = false;
            rights.white_queenside = false;
        }
        (Color::Black, PieceKind::King, _, 7) => {
            rights.black_kingside = false;
            rights.black_queenside = false;
        }
        (Color::White, PieceKind::Rook, 0, 0) => rights.white_queenside = false,
        (Color::White, PieceKind::Rook, 7, 0) => rights.white_kingside = false,
        (Color::Black, PieceKind::Rook, 0, 7) => rights.black_queenside = false,
        (Color::Black, PieceKind::Rook, 7, 7) => rights.black_kingside = false,
        _ => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Color;

    #[test]
    fn undo_record_stays_small() {
        // Undo must stay cheap: it is created and destroyed on every
        // single ply of search/perft. This is a soft budget, not a hard
        // ABI guarantee, but a regression here should be a deliberate
        // choice, not an accident. It is deliberately much smaller than
        // Position, which is the whole point of not cloning the board.
        assert!(std::mem::size_of::<Undo>() <= 24);
        assert!(std::mem::size_of::<Undo>() < std::mem::size_of::<Position>());
    }

    #[test]
    fn quiet_move_updates_board_and_side_to_move() {
        let mut position = Position::startpos();
        let mv = Move::new(
            Square::from_file_rank(4, 1), // e2
            Square::from_file_rank(4, 2), // e3
            MoveFlag::Quiet,
        );

        position.make_move(mv);

        assert_eq!(position.piece_at(Square::from_file_rank(4, 1)), None);
        assert_eq!(
            position.piece_at(Square::from_file_rank(4, 2)),
            Some(Piece::new(PieceKind::Pawn, Color::White))
        );
        assert_eq!(position.side_to_move(), Color::Black);
    }

    #[test]
    fn make_then_unmake_quiet_move_restores_position() {
        let mut position = Position::startpos();
        let before = position.clone();
        let mv = Move::new(
            Square::from_file_rank(4, 1),
            Square::from_file_rank(4, 2),
            MoveFlag::Quiet,
        );

        let undo = position.make_move(mv);
        position.unmake_move(mv, undo);

        assert_eq!(position, before);
    }

    #[test]
    fn double_pawn_push_sets_en_passant_square() {
        let mut position = Position::startpos();
        let mv = Move::new(
            Square::from_file_rank(4, 1), // e2
            Square::from_file_rank(4, 3), // e4
            MoveFlag::DoublePawnPush,
        );

        position.make_move(mv);

        assert_eq!(
            position.en_passant_square(),
            Some(Square::from_file_rank(4, 2)) // e3
        );
    }

    #[test]
    fn double_pawn_push_round_trips() {
        let mut position = Position::startpos();
        let before = position.clone();
        let mv = Move::new(
            Square::from_file_rank(4, 1),
            Square::from_file_rank(4, 3),
            MoveFlag::DoublePawnPush,
        );

        let undo = position.make_move(mv);
        position.unmake_move(mv, undo);

        assert_eq!(position, before);
    }

    #[test]
    fn en_passant_capture_removes_the_captured_pawn() {
        // White pawn on e5, black just played d7-d5, en passant target d6.
        let mut position =
            Position::from_fen("rnbqkbnr/ppp1pppp/8/3pP3/8/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 3")
                .expect("valid FEN");

        let mv = Move::new(
            Square::from_file_rank(4, 4), // e5
            Square::from_file_rank(3, 5), // d6
            MoveFlag::EnPassant,
        );

        position.make_move(mv);

        assert_eq!(
            position.piece_at(Square::from_file_rank(3, 5)),
            Some(Piece::new(PieceKind::Pawn, Color::White))
        );
        // The captured black pawn sat on d5, not on the destination d6.
        assert_eq!(position.piece_at(Square::from_file_rank(3, 4)), None);
        assert_eq!(position.en_passant_square(), None);
    }

    #[test]
    fn en_passant_capture_round_trips() {
        let mut position =
            Position::from_fen("rnbqkbnr/ppp1pppp/8/3pP3/8/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 3")
                .expect("valid FEN");
        let before = position.clone();

        let mv = Move::new(
            Square::from_file_rank(4, 4),
            Square::from_file_rank(3, 5),
            MoveFlag::EnPassant,
        );

        let undo = position.make_move(mv);
        position.unmake_move(mv, undo);

        assert_eq!(position, before);
    }

    #[test]
    fn promotion_replaces_pawn_with_promoted_piece() {
        let mut position = Position::from_fen("8/P7/8/8/8/8/8/k6K w - - 0 1").expect("valid FEN");

        let mv = Move::new(
            Square::from_file_rank(0, 6), // a7
            Square::from_file_rank(0, 7), // a8
            MoveFlag::PromoteQueen,
        );

        position.make_move(mv);

        assert_eq!(
            position.piece_at(Square::from_file_rank(0, 7)),
            Some(Piece::new(PieceKind::Queen, Color::White))
        );
        assert_eq!(position.piece_at(Square::from_file_rank(0, 6)), None);
    }

    #[test]
    fn promotion_round_trips() {
        let mut position = Position::from_fen("8/P7/8/8/8/8/8/k6K w - - 0 1").expect("valid FEN");
        let before = position.clone();

        let mv = Move::new(
            Square::from_file_rank(0, 6),
            Square::from_file_rank(0, 7),
            MoveFlag::PromoteQueen,
        );

        let undo = position.make_move(mv);
        position.unmake_move(mv, undo);

        assert_eq!(position, before);
    }

    #[test]
    fn promotion_with_capture_round_trips() {
        let mut position = Position::from_fen("1n6/P7/8/8/8/8/8/k6K w - - 0 1").expect("valid FEN");
        let before = position.clone();

        let mv = Move::new(
            Square::from_file_rank(0, 6), // a7
            Square::from_file_rank(1, 7), // b8, capturing the knight
            MoveFlag::PromoteQueen,
        );

        let undo = position.make_move(mv);
        assert_eq!(
            position.piece_at(Square::from_file_rank(1, 7)),
            Some(Piece::new(PieceKind::Queen, Color::White))
        );

        position.unmake_move(mv, undo);
        assert_eq!(position, before);
    }

    #[test]
    fn kingside_castle_moves_rook_and_king() {
        let mut position =
            Position::from_fen("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1").expect("valid FEN");

        let mv = Move::new(
            Square::from_file_rank(4, 0), // e1
            Square::from_file_rank(6, 0), // g1
            MoveFlag::CastleKingside,
        );

        position.make_move(mv);

        assert_eq!(
            position.piece_at(Square::from_file_rank(6, 0)),
            Some(Piece::new(PieceKind::King, Color::White))
        );
        assert_eq!(
            position.piece_at(Square::from_file_rank(5, 0)),
            Some(Piece::new(PieceKind::Rook, Color::White))
        );
        assert_eq!(position.piece_at(Square::from_file_rank(4, 0)), None);
        assert_eq!(position.piece_at(Square::from_file_rank(7, 0)), None);
    }

    #[test]
    fn kingside_castle_round_trips() {
        let mut position =
            Position::from_fen("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1").expect("valid FEN");
        let before = position.clone();

        let mv = Move::new(
            Square::from_file_rank(4, 0),
            Square::from_file_rank(6, 0),
            MoveFlag::CastleKingside,
        );

        let undo = position.make_move(mv);
        position.unmake_move(mv, undo);

        assert_eq!(position, before);
    }

    #[test]
    fn queenside_castle_round_trips() {
        let mut position =
            Position::from_fen("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1").expect("valid FEN");
        let before = position.clone();

        let mv = Move::new(
            Square::from_file_rank(4, 0),
            Square::from_file_rank(2, 0),
            MoveFlag::CastleQueenside,
        );

        let undo = position.make_move(mv);
        position.unmake_move(mv, undo);

        assert_eq!(position, before);
    }

    #[test]
    fn black_castle_round_trips() {
        let mut position =
            Position::from_fen("r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 0 1").expect("valid FEN");
        let before = position.clone();

        let mv = Move::new(
            Square::from_file_rank(4, 7),
            Square::from_file_rank(6, 7),
            MoveFlag::CastleKingside,
        );

        let undo = position.make_move(mv);
        position.unmake_move(mv, undo);

        assert_eq!(position, before);
    }

    #[test]
    fn king_move_clears_both_castling_rights() {
        let mut position =
            Position::from_fen("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1").expect("valid FEN");

        let mv = Move::new(
            Square::from_file_rank(4, 0), // e1
            Square::from_file_rank(4, 1), // e2
            MoveFlag::Quiet,
        );

        position.make_move(mv);

        let rights = position.castling_rights();
        assert!(!rights.white_kingside);
        assert!(!rights.white_queenside);
        assert!(rights.black_kingside);
        assert!(rights.black_queenside);
    }

    #[test]
    fn king_move_clearing_castling_rights_round_trips() {
        let mut position =
            Position::from_fen("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1").expect("valid FEN");
        let before = position.clone();

        let mv = Move::new(
            Square::from_file_rank(4, 0),
            Square::from_file_rank(4, 1),
            MoveFlag::Quiet,
        );

        let undo = position.make_move(mv);
        position.unmake_move(mv, undo);

        assert_eq!(position, before);
    }

    #[test]
    fn rook_move_clears_only_that_sides_castling_right() {
        let mut position =
            Position::from_fen("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1").expect("valid FEN");

        let mv = Move::new(
            Square::from_file_rank(0, 0), // a1 rook
            Square::from_file_rank(0, 1), // a2
            MoveFlag::Quiet,
        );

        position.make_move(mv);

        let rights = position.castling_rights();
        assert!(rights.white_kingside);
        assert!(!rights.white_queenside);
        assert!(rights.black_kingside);
        assert!(rights.black_queenside);
    }

    #[test]
    fn capturing_a_rook_clears_that_sides_castling_right() {
        // White rook on a7 captures the black rook on a8.
        let mut position =
            Position::from_fen("r3k2r/R7/8/8/8/8/8/4K2R w Kkq - 0 1").expect("valid FEN");
        let before = position.clone();

        let mv = Move::new(
            Square::from_file_rank(0, 6), // a7
            Square::from_file_rank(0, 7), // a8, capturing black rook
            MoveFlag::Quiet,
        );

        let undo = position.make_move(mv);

        let rights = position.castling_rights();
        assert!(rights.white_kingside);
        assert!(rights.black_kingside); // h8 rook untouched
        assert!(!rights.black_queenside); // a8 rook captured

        position.unmake_move(mv, undo);
        assert_eq!(position, before);
    }

    #[test]
    fn capture_resets_halfmove_clock() {
        let mut position =
            Position::from_fen("r3k2r/R7/8/8/8/8/8/4K2R w Kkq - 5 10").expect("valid FEN");

        let mv = Move::new(
            Square::from_file_rank(0, 6),
            Square::from_file_rank(0, 7),
            MoveFlag::Quiet,
        );

        position.make_move(mv);

        assert_eq!(position.halfmove_clock(), 0);
    }

    #[test]
    fn non_capture_non_pawn_move_increments_halfmove_clock() {
        let mut position =
            Position::from_fen("4k3/8/8/8/8/8/8/R3K3 w - - 5 10").expect("valid FEN");

        let mv = Move::new(
            Square::from_file_rank(0, 0), // a1
            Square::from_file_rank(0, 1), // a2
            MoveFlag::Quiet,
        );

        position.make_move(mv);

        assert_eq!(position.halfmove_clock(), 6);
    }

    #[test]
    fn pawn_move_resets_halfmove_clock() {
        let mut position =
            Position::from_fen("4k3/8/8/8/8/8/PPPPPPPP/4K3 w - - 5 10").expect("valid FEN");

        let mv = Move::new(
            Square::from_file_rank(4, 1),
            Square::from_file_rank(4, 2),
            MoveFlag::Quiet,
        );

        position.make_move(mv);

        assert_eq!(position.halfmove_clock(), 0);
    }

    #[test]
    fn fullmove_number_increments_only_after_black_moves() {
        let mut position = Position::startpos();
        assert_eq!(position.fullmove_number(), 1);

        let white_move = Move::new(
            Square::from_file_rank(4, 1),
            Square::from_file_rank(4, 3),
            MoveFlag::DoublePawnPush,
        );
        position.make_move(white_move);
        assert_eq!(position.fullmove_number(), 1);

        let black_move = Move::new(
            Square::from_file_rank(4, 6),
            Square::from_file_rank(4, 4),
            MoveFlag::DoublePawnPush,
        );
        position.make_move(black_move);
        assert_eq!(position.fullmove_number(), 2);
    }

    #[test]
    fn sequential_make_unmake_round_trips_multiple_plies() {
        let mut position = Position::startpos();
        let before = position.clone();

        let moves = [
            Move::new(
                Square::from_file_rank(4, 1),
                Square::from_file_rank(4, 3),
                MoveFlag::DoublePawnPush,
            ),
            Move::new(
                Square::from_file_rank(4, 6),
                Square::from_file_rank(4, 4),
                MoveFlag::DoublePawnPush,
            ),
            Move::new(
                Square::from_file_rank(6, 0),
                Square::from_file_rank(5, 2),
                MoveFlag::Quiet,
            ),
        ];

        let mut undos = Vec::new();
        for mv in moves {
            undos.push(position.make_move(mv));
        }
        for (mv, undo) in moves.into_iter().zip(undos).rev() {
            position.unmake_move(mv, undo);
        }

        assert_eq!(position, before);
    }
}
