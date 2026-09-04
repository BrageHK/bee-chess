//! Board squares.

use std::fmt;

/// A square on the board, encoded as `rank * 8 + file` (0 = a1, 63 = h8).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Square(u8);

impl Square {
    pub const COUNT: usize = 64;

    /// Builds a square from a 0..64 index. Panics if `index >= 64`.
    #[must_use]
    pub const fn new(index: u8) -> Self {
        assert!(index < 64, "square index out of range");
        Square(index)
    }

    /// Builds a square from zero-based file (0=a..7=h) and rank (0=1..7=8).
    /// Panics if either coordinate is out of range.
    #[must_use]
    pub const fn from_file_rank(file: u8, rank: u8) -> Self {
        assert!(file < 8, "file out of range");
        assert!(rank < 8, "rank out of range");
        Square(rank * 8 + file)
    }

    #[must_use]
    pub const fn index(self) -> u8 {
        self.0
    }

    #[must_use]
    pub const fn file(self) -> u8 {
        self.0 % 8
    }

    #[must_use]
    pub const fn rank(self) -> u8 {
        self.0 / 8
    }
}

impl fmt::Display for Square {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let file = (b'a' + self.file()) as char;
        let rank = (b'1' + self.rank()) as char;
        write!(f, "{file}{rank}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a1_is_index_zero() {
        assert_eq!(Square::from_file_rank(0, 0).index(), 0);
        assert_eq!(Square::from_file_rank(0, 0).to_string(), "a1");
    }

    #[test]
    fn h8_is_index_63() {
        assert_eq!(Square::from_file_rank(7, 7).index(), 63);
        assert_eq!(Square::from_file_rank(7, 7).to_string(), "h8");
    }

    #[test]
    fn file_and_rank_round_trip() {
        for index in 0..64u8 {
            let square = Square::new(index);
            assert_eq!(Square::from_file_rank(square.file(), square.rank()), square);
        }
    }
}
