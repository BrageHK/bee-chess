//! Compact move representation.
//!
//! `Move` is deliberately packed into a `u16` (from/to squares plus a
//! 4-bit flag) rather than a struct of separate fields, since move
//! generation and search will create very large numbers of these.

use super::piece::PieceKind;
use super::square::Square;

/// The special-move category a `Move` belongs to. Plain captures are not
/// distinguished from plain quiet moves here — that requires looking at
/// the board, not the move alone — but promotion, en passant, castling,
/// and double pawn pushes are encoded directly since move generation
/// always knows them at creation time.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MoveFlag {
    Quiet,
    DoublePawnPush,
    EnPassant,
    CastleKingside,
    CastleQueenside,
    PromoteKnight,
    PromoteBishop,
    PromoteRook,
    PromoteQueen,
}

impl MoveFlag {
    const fn to_bits(self) -> u16 {
        match self {
            MoveFlag::Quiet => 0,
            MoveFlag::DoublePawnPush => 1,
            MoveFlag::EnPassant => 2,
            MoveFlag::CastleKingside => 3,
            MoveFlag::CastleQueenside => 4,
            MoveFlag::PromoteKnight => 5,
            MoveFlag::PromoteBishop => 6,
            MoveFlag::PromoteRook => 7,
            MoveFlag::PromoteQueen => 8,
        }
    }

    const fn from_bits(bits: u16) -> Self {
        match bits {
            0 => MoveFlag::Quiet,
            1 => MoveFlag::DoublePawnPush,
            2 => MoveFlag::EnPassant,
            3 => MoveFlag::CastleKingside,
            4 => MoveFlag::CastleQueenside,
            5 => MoveFlag::PromoteKnight,
            6 => MoveFlag::PromoteBishop,
            7 => MoveFlag::PromoteRook,
            8 => MoveFlag::PromoteQueen,
            _ => panic!("invalid move flag bits"),
        }
    }

    /// The piece kind to promote to, if this flag is a promotion.
    #[must_use]
    pub const fn promotion_kind(self) -> Option<PieceKind> {
        match self {
            MoveFlag::PromoteKnight => Some(PieceKind::Knight),
            MoveFlag::PromoteBishop => Some(PieceKind::Bishop),
            MoveFlag::PromoteRook => Some(PieceKind::Rook),
            MoveFlag::PromoteQueen => Some(PieceKind::Queen),
            _ => None,
        }
    }
}

const FROM_SHIFT: u16 = 0;
const TO_SHIFT: u16 = 6;
const FLAG_SHIFT: u16 = 12;
const SQUARE_MASK: u16 = 0b0011_1111;
const FLAG_MASK: u16 = 0b1111;

/// A single chess move, packed into 16 bits: 6 bits `from` square, 6 bits
/// `to` square, 4 bits flag. This encodes everything needed to apply the
/// move to a `Position`; it does not carry the captured piece or any
/// other information needed to *undo* it, which belongs in a
/// `Position`-produced undo record instead (see the make/unmake work in
/// a follow-up PR).
#[derive(Clone, Copy, PartialEq, Eq, Default)]
pub struct Move(u16);

impl Move {
    #[must_use]
    pub const fn new(from: Square, to: Square, flag: MoveFlag) -> Self {
        let bits = ((from.index() as u16 & SQUARE_MASK) << FROM_SHIFT)
            | ((to.index() as u16 & SQUARE_MASK) << TO_SHIFT)
            | (flag.to_bits() << FLAG_SHIFT);
        Move(bits)
    }

    #[must_use]
    pub const fn from(self) -> Square {
        Square::new(((self.0 >> FROM_SHIFT) & SQUARE_MASK) as u8)
    }

    #[must_use]
    pub const fn to(self) -> Square {
        Square::new(((self.0 >> TO_SHIFT) & SQUARE_MASK) as u8)
    }

    #[must_use]
    pub const fn flag(self) -> MoveFlag {
        MoveFlag::from_bits((self.0 >> FLAG_SHIFT) & FLAG_MASK)
    }

    #[must_use]
    pub const fn is_promotion(self) -> bool {
        matches!(
            self.flag(),
            MoveFlag::PromoteKnight
                | MoveFlag::PromoteBishop
                | MoveFlag::PromoteRook
                | MoveFlag::PromoteQueen
        )
    }
}

impl std::fmt::Debug for Move {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Move")
            .field("from", &self.from())
            .field("to", &self.to())
            .field("flag", &self.flag())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trips_from_to_and_flag() {
        let from = Square::from_file_rank(4, 1); // e2
        let to = Square::from_file_rank(4, 3); // e4
        let mv = Move::new(from, to, MoveFlag::DoublePawnPush);
        assert_eq!(mv.from(), from);
        assert_eq!(mv.to(), to);
        assert_eq!(mv.flag(), MoveFlag::DoublePawnPush);
    }

    #[test]
    fn promotion_flag_carries_piece_kind() {
        let mv = Move::new(
            Square::from_file_rank(0, 6),
            Square::from_file_rank(0, 7),
            MoveFlag::PromoteQueen,
        );
        assert!(mv.is_promotion());
        assert_eq!(mv.flag().promotion_kind(), Some(PieceKind::Queen));
    }

    #[test]
    fn quiet_move_is_not_promotion() {
        let mv = Move::new(
            Square::from_file_rank(4, 1),
            Square::from_file_rank(4, 2),
            MoveFlag::Quiet,
        );
        assert!(!mv.is_promotion());
        assert_eq!(mv.flag().promotion_kind(), None);
    }

    #[test]
    fn move_fits_in_two_bytes() {
        assert_eq!(std::mem::size_of::<Move>(), 2);
    }
}
