//! Optional local Syzygy probing. All Fathom access is serialized: its file
//! mappings are process-global and root/DTZ probes are not thread safe.
//! No protocol strings or network access belong in this module.

use std::ffi::CString;
use std::sync::Mutex;

use fathom_syzygy_sys as fathom;

use crate::chess::{CastlingRights, Color, Move, PieceKind, Position, Square};
use crate::search::Score;

pub const DEFAULT_PROBE_LIMIT: u32 = 6;
pub const MAX_PROBE_LIMIT: u32 = 7;
/// Deliberately outside normal evaluation, below Bee's mate-score range.
pub const TABLEBASE_WIN: Score = 20_000;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Wdl {
    Loss,
    BlessedLoss,
    Draw,
    CursedWin,
    Win,
}

impl Wdl {
    fn decode(value: u32) -> Option<Self> {
        match value {
            fathom::TB_LOSS => Some(Self::Loss),
            fathom::TB_BLESSED_LOSS => Some(Self::BlessedLoss),
            fathom::TB_DRAW => Some(Self::Draw),
            fathom::TB_CURSED_WIN => Some(Self::CursedWin),
            fathom::TB_WIN => Some(Self::Win),
            _ => None,
        }
    }

    pub const fn score(self) -> Score {
        match self {
            Self::Win => TABLEBASE_WIN,
            Self::Loss => -TABLEBASE_WIN,
            Self::Draw | Self::CursedWin | Self::BlessedLoss => 0,
        }
    }

    /// WDL tables assume a just-reset fifty-move counter. Draws remain
    /// draws at a later counter, but decisive results need DTZ to be exact.
    pub fn search_score(self, halfmove_clock: u32) -> Option<Score> {
        (halfmove_clock == 0 || self.score() == 0).then(|| self.score())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TablebaseHit {
    pub pieces: u32,
    pub wdl: Wdl,
    pub dtz: Option<u32>,
    pub exact: bool,
    pub eval_cp: Score,
}

/// Cumulative over completed iterations of one search. Probes count actual
/// backend calls, including misses; hits count successful WDL/DTZ calls.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct TablebaseStats {
    pub probes: u64,
    pub hits: u64,
    pub draw_hits: u64,
    pub first_hit: Option<TablebaseHit>,
    pub root_hit: Option<TablebaseHit>,
    pub root_resolved: bool,
}

#[derive(Debug)]
pub struct Syzygy {
    path: String,
    limit: u32,
    largest: u32,
}

impl Default for Syzygy {
    fn default() -> Self {
        Self {
            path: String::new(),
            limit: DEFAULT_PROBE_LIMIT,
            largest: 0,
        }
    }
}

#[derive(Default)]
struct Backend {
    path: Option<String>,
    largest: u32,
}

static BACKEND: Mutex<Backend> = Mutex::new(Backend {
    path: None,
    largest: 0,
});

impl Backend {
    fn load(&mut self, path: &str) -> Result<u32, String> {
        if self.path.as_deref() == Some(path) {
            return Ok(self.largest);
        }
        let cpath = CString::new(path).map_err(|_| "tablebase path contains a NUL byte")?;
        // SAFETY: every Fathom call, including reinitialization, holds BACKEND.
        // CString lives throughout the call; Fathom copies the path.
        let ok = unsafe { fathom::tb_init(cpath.as_ptr().cast()) };
        self.path = Some(path.to_owned());
        self.largest = if ok { unsafe { fathom::TB_LARGEST } } else { 0 };
        if ok {
            Ok(self.largest)
        } else {
            Err("could not initialize tablebases".into())
        }
    }
}

impl Syzygy {
    pub fn path(&self) -> &str {
        &self.path
    }
    pub fn limit(&self) -> u32 {
        self.limit
    }

    pub fn set_limit(&mut self, limit: u32) {
        self.limit = limit.min(MAX_PROBE_LIMIT);
    }

    /// Empty paths disable probing. A failed configuration also disables it;
    /// no old tablebase knowledge is silently retained under a new path.
    pub fn set_path(&mut self, path: &str) -> Result<u32, String> {
        self.path = path.to_owned();
        self.largest = 0;
        if path.is_empty() {
            return Ok(0);
        }
        let mut backend = BACKEND.lock().map_err(|_| "tablebase lock poisoned")?;
        // Explicit setoption also allows rescanning a path after files change.
        backend.path = None;
        self.largest = backend.load(path)?;
        Ok(self.largest)
    }

    fn position(&self, position: &Position) -> Option<ProbePosition> {
        if self.path.is_empty()
            || self.limit < 2
            || self.largest == 0
            || position.castling_rights() != CastlingRights::none()
            || position.halfmove_clock() >= 100
        {
            return None;
        }
        ProbePosition::new(position, self.limit.min(self.largest))
    }

    pub(crate) fn probe_wdl(&self, position: &Position, stats: &mut TablebaseStats) -> Option<Wdl> {
        let p = self.position(position)?;
        let mut backend = BACKEND.lock().ok()?;
        backend.load(&self.path).ok()?;
        stats.probes += 1;
        // SAFETY: validated disjoint bitboards, one king per side, legal
        // non-moving king and EP square; no castling. The global lock protects
        // Fathom's mappings from concurrent reload, WDL and DTZ calls.
        let raw = unsafe {
            fathom::tb_probe_wdl(
                p.white, p.black, p.kings, p.queens, p.rooks, p.bishops, p.knights, p.pawns, 0, 0,
                p.ep, p.turn,
            )
        };
        let wdl = Wdl::decode(raw)?;
        stats.hits += 1;
        stats.draw_hits += u64::from(wdl.score() == 0);
        Some(wdl)
    }

    /// DTZ is optional and only consulted at the root. Verify the encoded
    /// move against Bee's own legal moves before it can be played.
    pub(crate) fn probe_root(
        &self,
        position: &Position,
        moves: &[Move],
        stats: &mut TablebaseStats,
    ) -> Option<(Wdl, u32, Move)> {
        let p = self.position(position)?;
        let mut backend = BACKEND.lock().ok()?;
        backend.load(&self.path).ok()?;
        stats.probes += 1;
        // SAFETY: same position/mapping invariants as probe_wdl. Null results
        // requests only the best move, so no foreign output buffer is needed.
        let raw = unsafe {
            fathom::tb_probe_root(
                p.white,
                p.black,
                p.kings,
                p.queens,
                p.rooks,
                p.bishops,
                p.knights,
                p.pawns,
                position.halfmove_clock(),
                0,
                p.ep,
                p.turn,
                std::ptr::null_mut(),
            )
        };
        if [
            fathom::TB_RESULT_FAILED,
            fathom::TB_RESULT_CHECKMATE,
            fathom::TB_RESULT_STALEMATE,
        ]
        .contains(&raw)
        {
            return None;
        }
        let wdl = Wdl::decode(raw & 0xf)?;
        let mv = decode_move(raw, moves)?;
        stats.hits += 1;
        stats.draw_hits += u64::from(wdl.score() == 0);
        Some((wdl, raw >> 20, mv))
    }
}

fn decode_move(raw: u32, moves: &[Move]) -> Option<Move> {
    // The sys crate's PROMOTES_MASK/EP_MASK constants are transposed in
    // 0.1.0. Decode according to its bundled Fathom tbprobe.h instead.
    let promotion = match (raw >> 16) & 7 {
        0 => None,
        1 => Some(PieceKind::Queen),
        2 => Some(PieceKind::Rook),
        3 => Some(PieceKind::Bishop),
        4 => Some(PieceKind::Knight),
        _ => return None,
    };
    moves.iter().copied().find(|mv| {
        mv.from().index() as u32 == (raw >> 10) & 63
            && mv.to().index() as u32 == (raw >> 4) & 63
            && mv.flag().promotion_kind() == promotion
    })
}

#[derive(Default)]
struct ProbePosition {
    white: u64,
    black: u64,
    kings: u64,
    queens: u64,
    rooks: u64,
    bishops: u64,
    knights: u64,
    pawns: u64,
    ep: u32,
    turn: u8,
}

pub fn piece_count(position: &Position) -> u32 {
    (0..64)
        .filter(|&i| position.piece_at(Square::new(i)).is_some())
        .count() as u32
}

impl ProbePosition {
    fn new(position: &Position, limit: u32) -> Option<Self> {
        let mut p = Self::default();
        for i in 0..64 {
            let Some(piece) = position.piece_at(Square::new(i)) else {
                continue;
            };
            let bit = 1u64 << i;
            if piece.color == Color::White {
                p.white |= bit;
            } else {
                p.black |= bit;
            }
            *match piece.kind {
                PieceKind::King => &mut p.kings,
                PieceKind::Queen => &mut p.queens,
                PieceKind::Rook => &mut p.rooks,
                PieceKind::Bishop => &mut p.bishops,
                PieceKind::Knight => &mut p.knights,
                PieceKind::Pawn => &mut p.pawns,
            } |= bit;
        }
        if (p.white | p.black).count_ones() > limit
            || (p.kings & p.white).count_ones() != 1
            || (p.kings & p.black).count_ones() != 1
            || p.pawns & 0xff000000000000ff != 0
        {
            return None;
        }
        let opponent = if position.side_to_move() == Color::White {
            p.black
        } else {
            p.white
        };
        let king = Square::new((p.kings & opponent).trailing_zeros() as u8);
        if position.is_square_attacked(king, position.side_to_move()) {
            return None;
        }
        // FEN parsing is permissive. Reject malformed EP metadata before C
        // move generation can interpret it as a capture of a nonexistent pawn.
        if let Some(ep) = position.en_passant_square() {
            let white = position.side_to_move() == Color::White;
            let rank = ep.index() / 8;
            if rank != if white { 5 } else { 2 } || position.piece_at(ep).is_some() {
                return None;
            }
            let captured = Square::new(if white {
                ep.index() - 8
            } else {
                ep.index() + 8
            });
            let pawn = position.piece_at(captured)?;
            if pawn.kind != PieceKind::Pawn || pawn.color == position.side_to_move() {
                return None;
            }
            p.ep = ep.index() as u32;
        }
        p.turn = u8::from(position.side_to_move() == Color::White);
        Some(p)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tables() -> Syzygy {
        let mut tb = Syzygy::default();
        assert_eq!(
            tb.set_path(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/tests/fixtures/syzygy"
            ))
            .unwrap(),
            3
        );
        tb
    }

    #[test]
    fn real_wdl_hits_cover_both_sides_and_draws() {
        let tb = tables();
        let mut stats = TablebaseStats::default();
        for (fen, expected) in [
            ("8/8/8/8/8/2K5/4Q3/k7 w - - 0 1", Wdl::Win),
            ("8/8/8/8/8/2K5/4Q3/k7 b - - 0 1", Wdl::Loss),
            ("k7/P7/2K5/8/8/8/8/8 w - - 0 1", Wdl::Draw),
        ] {
            assert_eq!(
                tb.probe_wdl(&Position::from_fen(fen).unwrap(), &mut stats),
                Some(expected),
                "{fen}"
            );
        }
        assert_eq!((stats.probes, stats.hits, stats.draw_hits), (3, 3, 1));
    }

    #[test]
    fn disabled_missing_and_out_of_range_fall_back() {
        let pos = Position::from_fen("k7/P7/2K5/8/8/8/8/8 w - - 0 1").unwrap();
        let mut tb = Syzygy::default();
        let mut stats = TablebaseStats::default();
        assert_eq!(tb.probe_wdl(&pos, &mut stats), None);
        assert_eq!(tb.set_path("/bee/nonexistent/syzygy").unwrap(), 0);
        assert_eq!(tb.probe_wdl(&pos, &mut stats), None);
        tb = tables();
        tb.set_limit(2);
        assert_eq!(tb.probe_wdl(&pos, &mut stats), None);
        tb.set_limit(0);
        assert_eq!(tb.probe_wdl(&pos, &mut stats), None);
        assert_eq!(stats.probes, 0);
        tb.set_limit(7);
        assert!(tb.probe_wdl(&pos, &mut stats).is_some());
        tb.set_path("").unwrap();
        assert_eq!(tb.probe_wdl(&pos, &mut stats), None);
        assert!(tb.set_path("bad\0path").is_err());
        assert_eq!(tb.probe_wdl(&pos, &mut stats), None);
    }

    #[test]
    fn invalid_positions_never_enter_fathom() {
        let tb = tables();
        let mut stats = TablebaseStats::default();
        for fen in [
            "8/8/8/8/8/8/8/8 w - - 0 1",
            "8/8/8/8/8/8/4K3/4k3 w - - 0 1",
            "k7/8/2K5/8/8/8/8/P7 w - - 0 1",
            "k7/P7/2K5/8/8/8/8/8 w K - 0 1",
            "k7/P7/2K5/8/8/8/8/8 w - e3 0 1",
            "k7/P7/2K5/8/8/8/8/8 w - e6 0 1",
            "k7/P7/2K5/8/8/8/8/8 w - - 100 1",
        ] {
            assert_eq!(
                tb.probe_wdl(&Position::from_fen(fen).unwrap(), &mut stats),
                None,
                "{fen}"
            );
        }
        assert_eq!(stats.probes, 0);
    }

    #[test]
    fn decisive_wdl_needs_a_reset_clock_but_cursed_results_are_draws() {
        assert_eq!(Wdl::Win.search_score(0), Some(TABLEBASE_WIN));
        assert_eq!(Wdl::Loss.search_score(0), Some(-TABLEBASE_WIN));
        assert_eq!(Wdl::Win.search_score(99), None);
        assert_eq!(Wdl::Loss.search_score(1), None);
        for wdl in [Wdl::Draw, Wdl::BlessedLoss, Wdl::CursedWin] {
            assert_eq!(wdl.search_score(99), Some(0));
        }
        assert_eq!(crate::search::mate_in_plies(TABLEBASE_WIN), None);
    }

    #[test]
    fn dtz_accounts_for_the_existing_halfmove_clock() {
        let tb = tables();
        let mut stats = TablebaseStats::default();
        let mut pos = Position::from_fen("7k/8/8/8/8/8/8/3QK3 w - - 0 1").unwrap();
        let moves = pos.generate_legal_moves();
        let (wdl, dtz, mv) = tb.probe_root(&pos, &moves, &mut stats).unwrap();
        assert_eq!(wdl, Wdl::Win);
        assert!(dtz > 1);
        assert!(moves.contains(&mv));
        pos.set_halfmove_clock(99);
        assert_eq!(
            tb.probe_root(&pos, &moves, &mut stats).unwrap().0,
            Wdl::CursedWin
        );
    }

    #[test]
    fn root_promotion_matches_bees_legal_move() {
        let tb = tables();
        let pos = Position::from_fen("8/4P1k1/4K3/8/8/8/8/8 w - - 0 1").unwrap();
        let moves = pos.generate_legal_moves();
        let (wdl, _, mv) = tb
            .probe_root(&pos, &moves, &mut TablebaseStats::default())
            .unwrap();
        assert_eq!(wdl, Wdl::Win);
        assert_eq!(mv.flag().promotion_kind(), Some(PieceKind::Queen));
        for (code, kind) in [
            (1, PieceKind::Queen),
            (2, PieceKind::Rook),
            (3, PieceKind::Bishop),
            (4, PieceKind::Knight),
        ] {
            let raw = (mv.from().index() as u32) << 10 | (mv.to().index() as u32) << 4 | code << 16;
            assert_eq!(
                decode_move(raw, &moves).unwrap().flag().promotion_kind(),
                Some(kind)
            );
        }
    }

    #[test]
    fn en_passant_bitboards_preserve_color_and_square() {
        let pos = Position::from_fen("k7/8/8/3pP3/8/8/8/7K w - d6 0 1").unwrap();
        let p = ProbePosition::new(&pos, 4).unwrap();
        assert_eq!(p.ep, 43);
        assert_eq!(p.turn, 1);
        assert_eq!(p.pawns, (1u64 << 35) | (1u64 << 36));
        assert_eq!(p.white, (1u64 << 7) | (1u64 << 36));
    }
}
