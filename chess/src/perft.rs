//! Perft (performance test): counts leaf nodes in the legal move tree to
//! a fixed depth, as a correctness oracle for move generation and
//! make/unmake. See <https://www.chessprogramming.org/Perft>.
//!
//! This is the milestone's closing gate: known starting-position counts
//! (20, 400, 8_902, 197_281, 4_865_609 for depths 1-5) and a handful of
//! "nasty" castling/en-passant/promotion positions with published perft
//! results (the Chess Programming Wiki's standard perft test suite) must
//! match exactly before search work begins. A perft mismatch means a
//! move generation or make/unmake bug -- either missing/extra
//! pseudo-legal moves, wrong legality filtering, or incorrect
//! make/unmake restoration -- rather than a strength problem.

use super::moves::Move;
use super::position::Position;

/// Counts leaf nodes in the legal move tree rooted at `position`, to
/// `depth` plies.
#[must_use]
pub fn perft(position: &mut Position, depth: u32) -> u64 {
    if depth == 0 {
        return 1;
    }

    let moves = position.generate_legal_moves();

    if depth == 1 {
        // At the last ply we only need the count, not a recursive
        // descent into each resulting position.
        return moves.len() as u64;
    }

    let mut nodes = 0;
    for mv in moves {
        let undo = position.make_move(mv);
        nodes += perft(position, depth - 1);
        position.unmake_move(mv, undo);
    }
    nodes
}

/// Per-root-move node counts ("perft divide"), useful for narrowing
/// down exactly which branch of the move tree disagrees with a known
/// perft result.
#[must_use]
pub fn perft_divide(position: &mut Position, depth: u32) -> Vec<(Move, u64)> {
    if depth == 0 {
        return Vec::new();
    }

    position
        .generate_legal_moves()
        .into_iter()
        .map(|mv| {
            let undo = position.make_move(mv);
            let nodes = perft(position, depth - 1);
            position.unmake_move(mv, undo);
            (mv, nodes)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn startpos_perft_1() {
        let mut position = Position::startpos();
        assert_eq!(perft(&mut position, 1), 20);
    }

    #[test]
    fn startpos_perft_2() {
        let mut position = Position::startpos();
        assert_eq!(perft(&mut position, 2), 400);
    }

    #[test]
    fn startpos_perft_3() {
        let mut position = Position::startpos();
        assert_eq!(perft(&mut position, 3), 8_902);
    }

    #[test]
    fn startpos_perft_4() {
        let mut position = Position::startpos();
        assert_eq!(perft(&mut position, 4), 197_281);
    }

    // Depth 5 from the starting position visits ~4.9M leaf nodes. With
    // this milestone's deliberately simple, unoptimized move generation
    // and legality-by-make/unmake, that's too slow for the default
    // `cargo test` run; #[ignore] keeps it out of the fast path while
    // still being one `cargo test -- --ignored` away; CI runs it
    // explicitly in release mode (see ci-rust.yml).
    #[test]
    #[ignore = "slow: ~4.9M nodes, run with `cargo test -- --ignored` (release mode recommended)"]
    fn startpos_perft_5() {
        let mut position = Position::startpos();
        assert_eq!(perft(&mut position, 5), 4_865_609);
    }

    // "Kiwipete": the second position in the Chess Programming Wiki's
    // standard perft test suite. Dense with pinned pieces, discovered
    // checks, all four castling rights, and multiple promotion-capture
    // opportunities.
    const KIWIPETE_FEN: &str =
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1";

    #[test]
    fn kiwipete_perft_1() {
        let mut position = Position::from_fen(KIWIPETE_FEN).expect("valid FEN");
        assert_eq!(perft(&mut position, 1), 48);
    }

    #[test]
    fn kiwipete_perft_2() {
        let mut position = Position::from_fen(KIWIPETE_FEN).expect("valid FEN");
        assert_eq!(perft(&mut position, 2), 2_039);
    }

    #[test]
    fn kiwipete_perft_3() {
        let mut position = Position::from_fen(KIWIPETE_FEN).expect("valid FEN");
        assert_eq!(perft(&mut position, 3), 97_862);
    }

    // Position 3 from the standard suite: sparse and endgame-like, but
    // exercises en passant heavily (both sides have pawns positioned to
    // capture en passant, including en passant discovered checks).
    const POSITION_3_FEN: &str = "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1";

    #[test]
    fn position_3_perft_1() {
        let mut position = Position::from_fen(POSITION_3_FEN).expect("valid FEN");
        assert_eq!(perft(&mut position, 1), 14);
    }

    #[test]
    fn position_3_perft_2() {
        let mut position = Position::from_fen(POSITION_3_FEN).expect("valid FEN");
        assert_eq!(perft(&mut position, 2), 191);
    }

    #[test]
    fn position_3_perft_3() {
        let mut position = Position::from_fen(POSITION_3_FEN).expect("valid FEN");
        assert_eq!(perft(&mut position, 3), 2_812);
    }

    #[test]
    fn position_3_perft_4() {
        let mut position = Position::from_fen(POSITION_3_FEN).expect("valid FEN");
        assert_eq!(perft(&mut position, 4), 43_238);
    }

    // Position 4 from the standard suite: asymmetric, heavy on
    // promotions (including underpromotion) and castling interactions.
    const POSITION_4_FEN: &str = "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1";

    #[test]
    fn position_4_perft_1() {
        let mut position = Position::from_fen(POSITION_4_FEN).expect("valid FEN");
        assert_eq!(perft(&mut position, 1), 6);
    }

    #[test]
    fn position_4_perft_2() {
        let mut position = Position::from_fen(POSITION_4_FEN).expect("valid FEN");
        assert_eq!(perft(&mut position, 2), 264);
    }

    #[test]
    fn position_4_perft_3() {
        let mut position = Position::from_fen(POSITION_4_FEN).expect("valid FEN");
        assert_eq!(perft(&mut position, 3), 9_467);
    }

    // Position 5 from the standard suite: catches a historically common
    // castling-rights bug (moving a rook must clear only that rook's
    // castling right, not both).
    const POSITION_5_FEN: &str = "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8";

    #[test]
    fn position_5_perft_1() {
        let mut position = Position::from_fen(POSITION_5_FEN).expect("valid FEN");
        assert_eq!(perft(&mut position, 1), 44);
    }

    #[test]
    fn position_5_perft_2() {
        let mut position = Position::from_fen(POSITION_5_FEN).expect("valid FEN");
        assert_eq!(perft(&mut position, 2), 1_486);
    }

    #[test]
    fn position_5_perft_3() {
        let mut position = Position::from_fen(POSITION_5_FEN).expect("valid FEN");
        assert_eq!(perft(&mut position, 3), 62_379);
    }

    // Position 6 from the standard suite (a symmetric middlegame-ish
    // position independently useful as a cross-check against a
    // different construction than the starting position/Kiwipete).
    const POSITION_6_FEN: &str =
        "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10";

    #[test]
    fn position_6_perft_1() {
        let mut position = Position::from_fen(POSITION_6_FEN).expect("valid FEN");
        assert_eq!(perft(&mut position, 1), 46);
    }

    #[test]
    fn position_6_perft_2() {
        let mut position = Position::from_fen(POSITION_6_FEN).expect("valid FEN");
        assert_eq!(perft(&mut position, 2), 2_079);
    }

    #[test]
    fn position_6_perft_3() {
        let mut position = Position::from_fen(POSITION_6_FEN).expect("valid FEN");
        assert_eq!(perft(&mut position, 3), 89_890);
    }

    #[test]
    fn perft_divide_sums_to_perft_total() {
        let mut position = Position::startpos();
        let divide = perft_divide(&mut position, 3);
        let total: u64 = divide.iter().map(|(_, nodes)| nodes).sum();
        assert_eq!(total, perft(&mut position, 3));
        assert_eq!(divide.len(), 20); // one entry per root move
    }

    #[test]
    fn perft_zero_depth_is_one() {
        let mut position = Position::startpos();
        assert_eq!(perft(&mut position, 0), 1);
    }
}
