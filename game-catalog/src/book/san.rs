//! Resolves one SAN move token (as Lichess's `GameRecord::moves` gives
//! them, e.g. `"e4"`, `"Nf3"`, `"Bxc6"`, `"O-O"`, `"e8=Q+"`) against a
//! `Position`'s legal moves.
//!
//! There is no SAN *generator* here, only a *matcher*: `bee-chess-core`
//! already knows how to enumerate every legal move from a position
//! (`Position::generate_legal_moves`), so resolving a SAN token only
//! needs to parse out the token's own structural pieces (moving piece,
//! destination, disambiguation, promotion, castling) and find the one
//! legal move they uniquely identify -- there's no need to reimplement
//! move generation or check/checkmate detection to do that. Suffixes
//! like `+`/`#`/`!`/`?` are stripped and ignored; this resolver only
//! cares whether a move is legal and matches the token, not whether the
//! game's own annotation of it was accurate.
//!
//! A pawn double-push (`e4` from the start position) and en passant
//! need no special-casing either, for the same reason: destination +
//! piece-kind + disambiguator matching already picks out the unique
//! legal move, and `Position::generate_legal_moves` is the source of
//! truth for which flag (`Quiet`, `DoublePawnPush`, `EnPassant`, ...)
//! that move actually carries.

use bee_chess_core::{Move, MoveFlag, PieceKind, Position, Square};

/// Why a SAN token couldn't be resolved against `position`'s legal
/// moves. Every variant means "this game's move list can't be trusted
/// past this point" -- the builder skips the rest of that game rather
/// than guessing (see `book::builder`'s docs).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SanError {
    /// The token itself isn't a shape this resolver understands.
    Malformed(String),
    /// No legal move matches the token.
    NoMatch(String),
    /// More than one legal move matches the token -- a genuinely
    /// ambiguous SAN string would have carried a disambiguator, so this
    /// means either the token or the position is wrong.
    Ambiguous(String),
}

impl std::fmt::Display for SanError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SanError::Malformed(s) => write!(f, "malformed SAN token: {s}"),
            SanError::NoMatch(s) => write!(f, "no legal move matches SAN token: {s}"),
            SanError::Ambiguous(s) => write!(f, "SAN token is ambiguous against legal moves: {s}"),
        }
    }
}

/// Resolves `san` (one whitespace-trimmed SAN token, check/mate/
/// annotation suffixes still attached) to the unique legal move in
/// `position` it identifies.
pub fn resolve(position: &Position, san: &str) -> Result<Move, SanError> {
    let san = san.trim();
    let core = strip_suffixes(san);

    if core == "O-O" || core == "0-0" {
        return find_unique(position, san, |mv| mv.flag() == MoveFlag::CastleKingside);
    }
    if core == "O-O-O" || core == "0-0-0" {
        return find_unique(position, san, |mv| mv.flag() == MoveFlag::CastleQueenside);
    }

    let (core, promotion) = split_promotion(core)?;
    let (piece_kind, rest) = leading_piece_kind(core);
    let rest = rest.trim_start_matches('x'); // capture marker, if the piece letter preceded it
    let (disambiguator, destination) = split_destination(rest, san)?;
    let to: Square = destination
        .parse()
        .map_err(|()| SanError::Malformed(san.to_string()))?;

    find_unique(position, san, |mv| {
        if mv.to() != to || mv.flag().promotion_kind() != promotion {
            return false;
        }
        let Some(piece) = position.piece_at(mv.from()) else {
            return false;
        };
        if piece.kind != piece_kind || piece.color != position.side_to_move() {
            return false;
        }
        disambiguator.matches(mv.from())
    })
}

/// Runs `predicate` over every legal move in `position`, requiring
/// exactly one match.
fn find_unique(
    position: &Position,
    original: &str,
    predicate: impl Fn(Move) -> bool,
) -> Result<Move, SanError> {
    let mut matches = position
        .generate_legal_moves()
        .into_iter()
        .filter(|mv| predicate(*mv));
    let first = matches
        .next()
        .ok_or_else(|| SanError::NoMatch(original.to_string()))?;
    if matches.next().is_some() {
        return Err(SanError::Ambiguous(original.to_string()));
    }
    Ok(first)
}

/// Strips check (`+`), checkmate (`#`), and NAG-style annotation
/// suffixes (`!`, `?`, and combinations like `!?`) from the end of a
/// SAN token, since none of them affect which move it identifies.
fn strip_suffixes(san: &str) -> &str {
    san.trim_end_matches(['+', '#', '!', '?'])
}

/// Splits off a trailing `=X` promotion suffix, if present.
fn split_promotion(core: &str) -> Result<(&str, Option<PieceKind>), SanError> {
    match core.split_once('=') {
        Some((rest, promo)) => {
            let kind = match promo {
                "Q" => PieceKind::Queen,
                "R" => PieceKind::Rook,
                "B" => PieceKind::Bishop,
                "N" => PieceKind::Knight,
                _ => return Err(SanError::Malformed(core.to_string())),
            };
            Ok((rest, Some(kind)))
        }
        None => Ok((core, None)),
    }
}

/// Reads a leading piece letter (`N`, `B`, `R`, `Q`, `K`), defaulting to
/// `Pawn` if the token starts with a file letter instead (a pawn move
/// never names its own piece in SAN).
fn leading_piece_kind(core: &str) -> (PieceKind, &str) {
    match core.chars().next() {
        Some('N') => (PieceKind::Knight, &core[1..]),
        Some('B') => (PieceKind::Bishop, &core[1..]),
        Some('R') => (PieceKind::Rook, &core[1..]),
        Some('Q') => (PieceKind::Queen, &core[1..]),
        Some('K') => (PieceKind::King, &core[1..]),
        _ => (PieceKind::Pawn, core),
    }
}

/// What's left to narrow down the moving piece's origin, beyond
/// destination + piece kind + color: nothing, a file, a rank, or a full
/// square. SAN only adds a disambiguator when the destination and piece
/// kind alone don't already pick a unique legal move.
enum Disambiguator {
    None,
    File(u8),
    Rank(u8),
    Square(Square),
}

impl Disambiguator {
    fn matches(&self, from: Square) -> bool {
        match self {
            Disambiguator::None => true,
            Disambiguator::File(file) => from.file() == *file,
            Disambiguator::Rank(rank) => from.rank() == *rank,
            Disambiguator::Square(square) => from == *square,
        }
    }
}

/// Splits `rest` (the token after any leading piece letter and capture
/// marker) into an optional disambiguator and the two-character
/// destination square at the end.
fn split_destination<'a>(
    rest: &'a str,
    original: &str,
) -> Result<(Disambiguator, &'a str), SanError> {
    if rest.len() < 2 {
        return Err(SanError::Malformed(original.to_string()));
    }
    let (prefix, destination) = rest.split_at(rest.len() - 2);
    // A capture marker can also appear right before the destination
    // (e.g. "Nbxd7"); strip it before reading the disambiguator.
    let prefix = prefix.trim_end_matches('x');

    let disambiguator = match prefix.len() {
        0 => Disambiguator::None,
        1 => {
            let ch = prefix.chars().next().unwrap();
            if ch.is_ascii_lowercase() {
                Disambiguator::File(ch as u8 - b'a')
            } else if ch.is_ascii_digit() {
                Disambiguator::Rank(ch as u8 - b'1')
            } else {
                return Err(SanError::Malformed(original.to_string()));
            }
        }
        2 => Disambiguator::Square(
            prefix
                .parse()
                .map_err(|()| SanError::Malformed(original.to_string()))?,
        ),
        _ => return Err(SanError::Malformed(original.to_string())),
    };

    Ok((disambiguator, destination))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn play(position: &mut Position, moves: &[&str]) {
        for san in moves {
            let mv = resolve(position, san).unwrap_or_else(|err| panic!("{san}: {err}"));
            position.make_move(mv);
        }
    }

    #[test]
    fn resolves_simple_pawn_and_knight_moves() {
        let mut position = Position::startpos();
        let mv = resolve(&position, "e4").unwrap();
        assert_eq!(mv.from(), "e2".parse().unwrap());
        assert_eq!(mv.to(), "e4".parse().unwrap());
        assert_eq!(mv.flag(), MoveFlag::DoublePawnPush);
        position.make_move(mv);

        let mv = resolve(&position, "Nf6").unwrap();
        assert_eq!(mv.from(), "g8".parse().unwrap());
        assert_eq!(mv.to(), "f6".parse().unwrap());
    }

    #[test]
    fn resolves_pawn_capture_with_file_disambiguator() {
        let mut position = Position::startpos();
        play(&mut position, &["e4", "d5"]);
        let mv = resolve(&position, "exd5").unwrap();
        assert_eq!(mv.from(), "e4".parse().unwrap());
        assert_eq!(mv.to(), "d5".parse().unwrap());
    }

    #[test]
    fn resolves_castling_both_sides() {
        let position = Position::from_fen("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1").unwrap();
        let mv = resolve(&position, "O-O").unwrap();
        assert_eq!(mv.flag(), MoveFlag::CastleKingside);

        let mut position2 = position.clone();
        position2.make_move(mv);
        let black_mv = resolve(&position2, "O-O-O").unwrap();
        assert_eq!(black_mv.flag(), MoveFlag::CastleQueenside);
    }

    #[test]
    fn resolves_promotion() {
        let position = Position::from_fen("8/4P3/8/8/8/8/8/4K2k w - - 0 1").unwrap();
        let mv = resolve(&position, "e8=Q").unwrap();
        assert_eq!(mv.flag(), MoveFlag::PromoteQueen);
        assert_eq!(mv.to(), "e8".parse().unwrap());
    }

    #[test]
    fn resolves_check_and_mate_suffixes() {
        // Fool's mate: the mating move is annotated "#" in real PGN.
        let mut position = Position::startpos();
        play(&mut position, &["f3", "e5", "g4"]);
        let mv = resolve(&position, "Qh4#").unwrap();
        assert_eq!(mv.to(), "h4".parse().unwrap());
    }

    #[test]
    fn disambiguates_two_knights_that_can_reach_the_same_square() {
        // Knights on b1 and f1 can both reach d2 (b1-d2 and f1-d2 are
        // each a valid knight move), requiring a file disambiguator.
        let position = Position::from_fen("4k3/8/8/8/8/8/8/1N2KN2 w - - 0 1").unwrap();
        let mv = resolve(&position, "Nbd2").unwrap();
        assert_eq!(mv.from(), "b1".parse().unwrap());
        let mv2 = resolve(&position, "Nfd2").unwrap();
        assert_eq!(mv2.from(), "f1".parse().unwrap());
    }

    #[test]
    fn unknown_token_is_malformed_not_a_panic() {
        let position = Position::startpos();
        assert!(matches!(
            resolve(&position, "??"),
            Err(SanError::Malformed(_))
        ));
    }

    #[test]
    fn illegal_move_is_no_match() {
        let position = Position::startpos();
        assert!(matches!(
            resolve(&position, "e5"),
            Err(SanError::NoMatch(_))
        ));
    }
}
