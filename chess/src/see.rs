//! Static Exchange Evaluation (SEE) support.
//!
//! SEE answers "if both sides keep capturing on this one square with
//! their cheapest available attacker, who comes out ahead materially?"
//! -- a cheap way to judge a capture sequence without a real search.
//! This module supplies the one primitive the simulation loop (in
//! `bee_engine::search`, see that crate's docs for why the actual
//! swap-off algorithm lives there rather than here) needs repeatedly:
//! "of every piece of `color` currently attacking `square`, which one
//! is cheapest to move there?" -- called again after each simulated
//! capture removes a piece, since removing an attacker can reveal a
//! new one behind it (an x-ray attack, e.g. a rook standing behind a
//! knight that just captured).
//!
//! Deliberately separate from `super::attacks`'s `is_square_attacked`:
//! that answers a yes/no question and doesn't care which specific piece
//! is attacking, which is all check detection and legal move filtering
//! ever needed. SEE needs the actual attacking square (to remove that
//! exact piece from a scratch board between iterations) and its kind
//! (to compare attacker costs), so it reuses the same attack-pattern
//! tables `attacks.rs` already has rather than introducing a second
//! set of them.

use super::movegen::{
    offset_square, BISHOP_DIRECTIONS, KING_OFFSETS, KNIGHT_OFFSETS, ROOK_DIRECTIONS,
};
use super::piece::{Color, PieceKind};
use super::position::Position;
use super::square::Square;

impl Position {
    /// The cheapest piece of `color` currently attacking `square`, if
    /// any -- `(attacker_square, attacker_kind)`. "Cheapest" uses the
    /// same classical material ordering SEE's own simulation compares
    /// exchanges with (pawn < knight/bishop < rook < queen < king);
    /// ties (e.g. two rooks) resolve to whichever this scans first,
    /// which is an arbitrary but immaterial choice -- SEE's result
    /// doesn't depend on which same-valued attacker is picked, only on
    /// its value.
    ///
    /// Looks outward from `square` along each attack pattern (pawn,
    /// knight, king, then bishop/rook sliding directions, exactly like
    /// `is_square_attacked`), rather than generating every pseudo-legal
    /// move for `color` and filtering by destination -- the same
    /// cost discipline `is_square_attacked` itself already established.
    #[must_use]
    pub fn least_valuable_attacker(
        &self,
        square: Square,
        color: Color,
    ) -> Option<(Square, PieceKind)> {
        // Checked in ascending material order so the first match found
        // is already the cheapest -- no need to collect every attacker
        // and sort.
        if let Some(from) = self.pawn_attacker(square, color) {
            return Some((from, PieceKind::Pawn));
        }
        if let Some(from) = self.knight_attacker(square, color) {
            return Some((from, PieceKind::Knight));
        }
        // Bishops and knights share a material value in this engine's
        // classical scale (see `ordering_piece_value` in
        // `bee_engine::search::alpha_beta`), so a bishop is checked
        // right after knights, before rooks.
        if let Some(from) =
            self.sliding_attacker(square, color, &BISHOP_DIRECTIONS, PieceKind::Bishop)
        {
            return Some((from, PieceKind::Bishop));
        }
        if let Some(from) = self.sliding_attacker(square, color, &ROOK_DIRECTIONS, PieceKind::Rook)
        {
            return Some((from, PieceKind::Rook));
        }
        if let Some(from) = self.queen_attacker(square, color) {
            return Some((from, PieceKind::Queen));
        }
        if let Some(from) = self.king_attacker(square, color) {
            return Some((from, PieceKind::King));
        }
        None
    }

    fn pawn_attacker(&self, square: Square, color: Color) -> Option<Square> {
        let behind: i8 = match color {
            Color::White => -1,
            Color::Black => 1,
        };
        [-1i8, 1i8].into_iter().find_map(|df| {
            let attacker_square = offset_square(square, df, behind)?;
            matches!(
                self.piece_at(attacker_square),
                Some(piece) if piece.kind == PieceKind::Pawn && piece.color == color
            )
            .then_some(attacker_square)
        })
    }

    fn knight_attacker(&self, square: Square, color: Color) -> Option<Square> {
        KNIGHT_OFFSETS.iter().find_map(|&(df, dr)| {
            let attacker_square = offset_square(square, df, dr)?;
            matches!(
                self.piece_at(attacker_square),
                Some(piece) if piece.kind == PieceKind::Knight && piece.color == color
            )
            .then_some(attacker_square)
        })
    }

    fn king_attacker(&self, square: Square, color: Color) -> Option<Square> {
        KING_OFFSETS.iter().find_map(|&(df, dr)| {
            let attacker_square = offset_square(square, df, dr)?;
            matches!(
                self.piece_at(attacker_square),
                Some(piece) if piece.kind == PieceKind::King && piece.color == color
            )
            .then_some(attacker_square)
        })
    }

    /// A queen specifically (not "a bishop/rook slider or a queen," the
    /// way `attacked_by_sliding` treats it) -- `least_valuable_attacker`
    /// already tried bishop- and rook-pattern squares for an actual
    /// bishop/rook above, so by the time this is reached the only
    /// sliding piece left that could be attacking along either pattern
    /// is a queen.
    fn queen_attacker(&self, square: Square, color: Color) -> Option<Square> {
        for directions in [&BISHOP_DIRECTIONS, &ROOK_DIRECTIONS] {
            if let Some(from) = self.sliding_attacker(square, color, directions, PieceKind::Queen) {
                return Some(from);
            }
        }
        None
    }

    /// Looks outward from `square` along `directions` for the first
    /// piece encountered, returning its square if it's `color` and
    /// matches `kind` (queens are matched separately by `queen_attacker`
    /// -- unlike `attacked_by_sliding`, this does *not* also accept a
    /// queen when asked about a bishop/rook, since `least_valuable_
    /// attacker`'s ascending-value scan needs to tell "a bishop is
    /// attacking" apart from "only a queen is attacking along this same
    /// line" to report the right, cheaper kind first).
    fn sliding_attacker(
        &self,
        square: Square,
        color: Color,
        directions: &[(i8, i8)],
        kind: PieceKind,
    ) -> Option<Square> {
        directions.iter().find_map(|&(df, dr)| {
            let mut current = square;
            loop {
                let next = offset_square(current, df, dr)?;
                match self.piece_at(next) {
                    None => current = next,
                    Some(piece) if piece.color == color && piece.kind == kind => {
                        return Some(next);
                    }
                    Some(_) => return None,
                }
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sq(file: u8, rank: u8) -> Square {
        Square::from_file_rank(file, rank)
    }

    #[test]
    fn no_attacker_on_an_empty_board_is_none() {
        let position = Position::from_fen("4k3/8/8/8/8/8/8/4K3 w - - 0 1").unwrap();
        assert_eq!(
            position.least_valuable_attacker(sq(4, 4), Color::White),
            None
        );
    }

    #[test]
    fn a_lone_pawn_attacker_is_found() {
        let position = Position::from_fen("4k3/8/8/3P4/8/8/8/4K3 w - - 0 1").unwrap();
        // White pawn on d5 attacks e6 (and c6) diagonally forward.
        assert_eq!(
            position.least_valuable_attacker(sq(4, 5), Color::White),
            Some((sq(3, 4), PieceKind::Pawn))
        );
    }

    #[test]
    fn a_pawn_is_preferred_over_a_more_valuable_attacker_on_the_same_square() {
        // Both a white pawn (c4) and a white rook (a5) attack b5;
        // the pawn is cheaper and must be returned.
        let position = Position::from_fen("4k3/8/8/RP6/1PP5/8/8/4K3 w - - 0 1").unwrap();
        assert_eq!(
            position.least_valuable_attacker(sq(1, 4), Color::White),
            Some((sq(2, 3), PieceKind::Pawn))
        );
    }

    #[test]
    fn a_sliding_attacker_is_blocked_by_an_intervening_piece() {
        // A white rook on a1 would attack a8, but a white pawn on a4
        // blocks the line -- there's no white attacker on a8 at all.
        let position = Position::from_fen("4k3/8/8/8/P7/8/8/R3K3 w - - 0 1").unwrap();
        assert_eq!(
            position.least_valuable_attacker(sq(0, 7), Color::White),
            None
        );
    }

    #[test]
    fn removing_the_blocker_reveals_the_slider_behind_it() {
        // Same as above, but the blocking pawn is gone -- the rook on
        // a1 now attacks straight up the fully open a-file.
        let position = Position::from_fen("4k3/8/8/8/8/8/8/R3K3 w - - 0 1").unwrap();
        assert_eq!(
            position.least_valuable_attacker(sq(0, 7), Color::White),
            Some((sq(0, 0), PieceKind::Rook))
        );
    }

    #[test]
    fn a_bishop_is_found_before_a_queen_on_the_same_diagonal() {
        // White bishop on e5 and White queen on f6 both sit on the same
        // diagonal running through d4; the bishop is closer (and
        // cheaper), so it must be found -- the queen behind it is both
        // more expensive and actually blocked from reaching d4 at all.
        let position = Position::from_fen("4k3/6Q1/8/4B3/8/8/8/4K3 w - - 0 1").unwrap();
        assert_eq!(
            position.least_valuable_attacker(sq(3, 3), Color::White),
            Some((sq(4, 4), PieceKind::Bishop))
        );
    }

    #[test]
    fn a_queen_is_found_when_no_cheaper_slider_is_on_the_line() {
        let position = Position::from_fen("4k3/8/8/8/8/8/8/Q3K3 w - - 0 1").unwrap();
        // The queen on a1 attacks straight up the open a-file.
        assert_eq!(
            position.least_valuable_attacker(sq(0, 6), Color::White),
            Some((sq(0, 0), PieceKind::Queen))
        );
    }

    #[test]
    fn a_knight_attacker_is_found() {
        let position = Position::from_fen("4k3/8/8/8/8/2N5/8/4K3 w - - 0 1").unwrap();
        // Knight on c3 attacks e4 (among others).
        assert_eq!(
            position.least_valuable_attacker(sq(4, 3), Color::White),
            Some((sq(2, 2), PieceKind::Knight))
        );
    }

    #[test]
    fn a_king_attacker_is_found_only_adjacent() {
        let position = Position::from_fen("8/8/8/3k4/8/8/8/4K3 w - - 0 1").unwrap();
        assert_eq!(
            position.least_valuable_attacker(sq(4, 4), Color::Black),
            Some((sq(3, 4), PieceKind::King))
        );
        assert_eq!(
            position.least_valuable_attacker(sq(4, 6), Color::Black),
            None
        );
    }

    #[test]
    fn attacker_search_is_independent_of_side_to_move() {
        // The position's side to move is White, but this asks about
        // Black's attackers -- least_valuable_attacker must not care
        // whose turn it actually is (unlike generate_pseudo_legal_moves).
        // Knight on d5 (file 3, rank 4) attacks c3 (file 2, rank 2).
        let position = Position::from_fen("4k3/8/8/3n4/8/8/8/4K3 w - - 0 1").unwrap();
        assert_eq!(
            position.least_valuable_attacker(sq(2, 2), Color::Black),
            Some((sq(3, 4), PieceKind::Knight))
        );
    }
}
