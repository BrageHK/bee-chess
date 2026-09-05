//! Piece types and colors.

/// The side to move or the owner of a piece.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Color {
    White,
    Black,
}

impl Color {
    #[must_use]
    pub const fn opposite(self) -> Self {
        match self {
            Color::White => Color::Black,
            Color::Black => Color::White,
        }
    }
}

/// A chess piece type, independent of color.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PieceKind {
    Pawn,
    Knight,
    Bishop,
    Rook,
    Queen,
    King,
}

/// A piece: a kind plus the color that owns it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Piece {
    pub kind: PieceKind,
    pub color: Color,
}

impl Piece {
    #[must_use]
    pub const fn new(kind: PieceKind, color: Color) -> Self {
        Piece { kind, color }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn opposite_color_round_trips() {
        assert_eq!(Color::White.opposite(), Color::Black);
        assert_eq!(Color::Black.opposite(), Color::White);
        assert_eq!(Color::White.opposite().opposite(), Color::White);
    }

    #[test]
    fn piece_carries_kind_and_color() {
        let piece = Piece::new(PieceKind::Knight, Color::Black);
        assert_eq!(piece.kind, PieceKind::Knight);
        assert_eq!(piece.color, Color::Black);
    }
}
