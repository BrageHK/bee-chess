//! Offline, fixed-node analysis of both players in standard catalog games.
//!
//! See `docs/game-analysis.md` for the versioned score/phase conventions and CLI.

#[cfg(test)]
mod tests;
mod uci;

use std::collections::HashSet;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use bee_chess_core::{Color as ChessColor, Move, PieceKind, Position, Square};
use sha2::{Digest, Sha256};

use crate::analysis::{GamePhase, NewAnalysisRun, NewGameAnalysis, NewMoveAnalysis};
use crate::{book::san, Color, GameCatalog, GameFilter, GameRecord};
use uci::{Score, Stockfish};

/// Version of the analysis semantics, independent of the SQLite schema.
pub const ANALYSIS_VERSION: i64 = 2;

#[derive(Debug, thiserror::Error)]
pub enum AnalysisError {
    #[error(transparent)]
    Catalog(#[from] crate::Error),
    #[error("Stockfish I/O: {0}")]
    Io(#[from] std::io::Error),
    #[error("Stockfish protocol: {0}")]
    Engine(String),
    #[error("analysis configuration: {0}")]
    Config(String),
}

#[derive(Debug, Clone)]
pub struct AnalyzeConfig {
    pub stockfish: PathBuf,
    pub players: Vec<String>,
    pub nodes_per_position: u64,
    /// Select the most recent N matching games, including completed games.
    /// Applying this before resume filtering makes repeating a limited pass a no-op.
    pub limit: Option<usize>,
}

#[derive(Debug)]
pub enum Progress {
    Started {
        run_id: i64,
        games: usize,
        already_analyzed: usize,
    },
    Completed {
        game_id: String,
        plies: usize,
    },
    Rejected {
        game_id: String,
        reason: String,
    },
}

#[derive(Debug, Default)]
pub struct AnalysisReport {
    pub run_id: i64,
    pub games_analyzed: usize,
    pub games_already_analyzed: usize,
    pub plies_analyzed: usize,
    pub searches: usize,
    pub rejected: Vec<(String, String)>,
}

/// Starts Stockfish locally. No downloads or network calls occur on this path.
pub fn analyze(
    catalog: &GameCatalog,
    config: &AnalyzeConfig,
    progress: impl FnMut(Progress),
) -> Result<AnalysisReport, AnalysisError> {
    let players = normalized_players(&config.players)?;
    if config.nodes_per_position == 0 || config.nodes_per_position > i64::MAX as u64 {
        return Err(AnalysisError::Config(
            "nodes must be between 1 and i64::MAX".into(),
        ));
    }
    if config.limit == Some(0) {
        return Err(AnalysisError::Config("limit must be positive".into()));
    }
    let path = config.stockfish.canonicalize()?;
    let digest = binary_digest(&path)?;
    let mut engine = Stockfish::spawn(&path, Duration::from_secs(60))?;
    let configuration = serde_json::json!({
        "engine_sha256": digest,
        "players": players,
        "threads": 1, "hash_mb": 16, "multipv": 1,
        "skill_level": 20, "limit_strength": false, "ponder": false,
        "chess960": false, "syzygy_probe_limit": 0, "nodestime": 0,
        "history": "startpos-with-all-moves", "reset": "ucinewgame-per-position",
        "score_selection": "last-exact-primary-pv",
        "platform": format!("{}-{}", std::env::consts::OS, std::env::consts::ARCH),
    })
    .to_string();
    let run_id = match catalog.latest_analysis_run_with_config(
        &engine.name,
        Some(config.nodes_per_position as i64),
        Some(1),
        ANALYSIS_VERSION,
        Some(&configuration),
    )? {
        Some(run) => run.id,
        None => catalog.record_analysis_run(&NewAnalysisRun {
            engine: engine.name.clone(),
            nodes_per_position: Some(config.nodes_per_position as i64),
            multipv: Some(1),
            schema_version: ANALYSIS_VERSION,
            created_at: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map_err(|e| AnalysisError::Config(e.to_string()))?
                .as_millis() as i64,
            configuration: Some(configuration),
        })?,
    };
    analyze_with_engine(catalog, config, &players, run_id, &mut engine, progress)
}

fn normalized_players(players: &[String]) -> Result<Vec<String>, AnalysisError> {
    if players.is_empty() || players.iter().any(|p| p.trim().is_empty()) {
        return Err(AnalysisError::Config(
            "at least one nonempty --player is required".into(),
        ));
    }
    let mut players: Vec<_> = players
        .iter()
        .map(|p| p.trim().to_ascii_lowercase())
        .collect();
    players.sort();
    players.dedup();
    Ok(players)
}

fn binary_digest(path: &Path) -> Result<String, AnalysisError> {
    let mut file = std::fs::File::open(path)?;
    let mut hash = Sha256::new();
    let mut buffer = [0u8; 65536];
    loop {
        let n = file.read(&mut buffer)?;
        if n == 0 {
            break;
        }
        hash.update(&buffer[..n]);
    }
    Ok(format!("{:x}", hash.finalize()))
}

#[derive(Debug, Clone)]
struct Search {
    score: Score,
    best_move: Option<String>,
    pv: Vec<String>,
}

trait SearchEngine {
    fn search(
        &mut self,
        history: &[String],
        position: &Position,
        nodes: u64,
    ) -> Result<Search, AnalysisError>;
}

fn analyze_with_engine(
    catalog: &GameCatalog,
    config: &AnalyzeConfig,
    players: &[String],
    run_id: i64,
    engine: &mut impl SearchEngine,
    mut progress: impl FnMut(Progress),
) -> Result<AnalysisReport, AnalysisError> {
    let matches = |name: Option<&str>| {
        name.is_some_and(|n| players.iter().any(|p| p.eq_ignore_ascii_case(n)))
    };
    let mut games: Vec<_> = catalog
        .games(&GameFilter::all())?
        .into_iter()
        .filter(|g| matches(g.white.as_deref()) || matches(g.black.as_deref()))
        .collect();
    games.sort_by(|a, b| b.played_at.cmp(&a.played_at).then_with(|| a.id.cmp(&b.id)));
    if let Some(limit) = config.limit {
        games.truncate(limit);
    }
    let pending: HashSet<_> = catalog
        .unanalyzed_game_ids(
            run_id,
            &games.iter().map(|g| g.id.clone()).collect::<Vec<_>>(),
        )?
        .into_iter()
        .collect();
    let mut report = AnalysisReport {
        run_id,
        games_already_analyzed: games.len() - pending.len(),
        ..Default::default()
    };
    progress(Progress::Started {
        run_id,
        games: games.len(),
        already_analyzed: report.games_already_analyzed,
    });
    for game in games.into_iter().filter(|g| pending.contains(&g.id)) {
        let replay = replay(&game).and_then(|r| {
            match (
                matches(game.white.as_deref()),
                matches(game.black.as_deref()),
            ) {
                (true, false) => Ok((r, Color::White)),
                (false, true) => Ok((r, Color::Black)),
                _ => Err(
                    "both players match Bee identities; select one identity for self-play".into(),
                ),
            }
        });
        let (replay, bee_color) = match replay {
            Ok(value) => value,
            Err(reason) => {
                progress(Progress::Rejected {
                    game_id: game.id.clone(),
                    reason: reason.clone(),
                });
                report.rejected.push((game.id, reason));
                continue;
            }
        };
        // N+1 position searches supply both evaluations for all N plies. The
        // next position's score is negated across the move boundary exactly once.
        let mut before = engine.search(&[], &replay.positions[0], config.nodes_per_position)?;
        report.searches += 1;
        let mut moves = Vec::with_capacity(replay.moves.len());
        for (ply, played) in replay.moves.iter().enumerate() {
            let after = engine.search(
                &replay.moves[..=ply],
                &replay.positions[ply + 1],
                config.nodes_per_position,
            )?;
            report.searches += 1;
            let position = &replay.positions[ply];
            let color = catalog_color(position.side_to_move());
            let after_score = after.score.negated();
            let loss = match (before.score.cp(), after_score.cp()) {
                (Some(b), Some(a)) => Some(b.saturating_sub(a).max(0)),
                _ => None,
            };
            moves.push(NewMoveAnalysis {
                analysis_run_id: run_id,
                game_id: game.id.clone(),
                ply: ply as u32,
                fen_before: position.to_fen(),
                played_move: played.clone(),
                best_move: before.best_move.clone(),
                eval_before_cp: before.score.cp(),
                eval_after_cp: after_score.cp(),
                centipawn_loss: loss,
                mate_before: before.score.mate(),
                mate_after: after_score.mate(),
                phase: classify_phase(position, ply),
                mover_color: Some(color),
                is_bee: Some(color == bee_color),
                pv: Some(before.pv.join(" ")),
            });
            before = after;
        }
        let summary = game_summary(run_id, &game.id, bee_color, &moves);
        catalog.record_complete_game_analysis(&moves, &summary)?;
        report.games_analyzed += 1;
        report.plies_analyzed += moves.len();
        progress(Progress::Completed {
            game_id: game.id,
            plies: moves.len(),
        });
    }
    Ok(report)
}

struct Replay {
    positions: Vec<Position>,
    moves: Vec<String>,
}

fn replay(game: &GameRecord) -> Result<Replay, String> {
    if game
        .variant
        .as_deref()
        .is_some_and(|v| !v.eq_ignore_ascii_case("standard"))
    {
        return Err("only standard chess from the initial position is supported".into());
    }
    if game.raw_pgn.as_deref().is_some_and(|pgn| {
        pgn.lines().any(|line| {
            let line = line.trim();
            line.starts_with("[FEN ") || line.starts_with("[SetUp \"1\"")
        })
    }) {
        return Err("PGN setup positions are not supported".into());
    }
    if !matches!(game.result.as_deref(), Some("1-0" | "0-1" | "1/2-1/2")) {
        return Err("game has no final result".into());
    }
    let tokens = game.plies();
    if tokens.is_empty() {
        return Err("game has no moves".into());
    }
    let mut position = Position::startpos();
    let mut replay = Replay {
        positions: vec![position.clone()],
        moves: Vec::with_capacity(tokens.len()),
    };
    // Validate the whole game before spending any Stockfish nodes.
    for (ply, token) in tokens.iter().enumerate() {
        let mv = san::resolve(&position, token).map_err(|e| format!("ply {ply}: {e}"))?;
        replay.moves.push(move_uci(mv));
        position.make_move(mv);
        replay.positions.push(position.clone());
    }
    Ok(replay)
}

fn move_uci(mv: Move) -> String {
    let suffix = match mv.flag().promotion_kind() {
        Some(PieceKind::Queen) => "q",
        Some(PieceKind::Rook) => "r",
        Some(PieceKind::Bishop) => "b",
        Some(PieceKind::Knight) => "n",
        _ => "",
    };
    format!("{}{}{suffix}", mv.from(), mv.to())
}

fn catalog_color(color: ChessColor) -> Color {
    match color {
        ChessColor::White => Color::White,
        ChessColor::Black => Color::Black,
    }
}

/// Endgame takes priority: total phase material <= 8 (N/B=1, R=2, Q=4).
/// Otherwise the first 20 plies are opening and later plies middlegame.
pub fn classify_phase(position: &Position, ply: usize) -> GamePhase {
    let material: u32 = (0..64)
        .filter_map(|i| position.piece_at(Square::new(i)))
        .map(|p| match p.kind {
            PieceKind::Knight | PieceKind::Bishop => 1,
            PieceKind::Rook => 2,
            PieceKind::Queen => 4,
            _ => 0,
        })
        .sum();
    if material <= 8 {
        GamePhase::Endgame
    } else if ply < 20 {
        GamePhase::Opening
    } else {
        GamePhase::Middlegame
    }
}

#[derive(Debug, Default)]
pub struct LossStats {
    pub moves: usize,
    pub cp_moves: usize,
    pub total_loss: i64,
    pub worst_loss: Option<i32>,
    pub inaccuracies: u32,
    pub mistakes: u32,
    pub blunders: u32,
    pub over_200: usize,
    pub mate_moves: usize,
}

impl LossStats {
    pub fn add(&mut self, cp_loss: Option<i32>, mate_before: Option<i32>, mate_after: Option<i32>) {
        self.moves += 1;
        if mate_before.is_some() || mate_after.is_some() {
            self.mate_moves += 1;
        }
        if let Some(cp) = cp_loss {
            self.cp_moves += 1;
            self.total_loss += i64::from(cp);
            self.worst_loss = Some(self.worst_loss.map_or(cp, |old| old.max(cp)));
            match cp {
                50..=99 => self.inaccuracies += 1,
                100..=199 => self.mistakes += 1,
                200.. => self.blunders += 1,
                _ => {}
            }
            if cp > 200 {
                self.over_200 += 1;
            }
        }
    }

    pub fn acpl(&self) -> Option<f64> {
        (self.cp_moves > 0).then(|| self.total_loss as f64 / self.cp_moves as f64)
    }
}

fn game_summary(
    run_id: i64,
    game_id: &str,
    bee_color: Color,
    moves: &[NewMoveAnalysis],
) -> NewGameAnalysis {
    let mut all = LossStats::default();
    let mut phases = [
        LossStats::default(),
        LossStats::default(),
        LossStats::default(),
    ];
    for m in moves.iter().filter(|m| m.is_bee == Some(true)) {
        all.add(m.centipawn_loss, m.mate_before, m.mate_after);
        let phase = match m.phase {
            GamePhase::Opening => 0,
            GamePhase::Middlegame => 1,
            GamePhase::Endgame => 2,
        };
        phases[phase].add(m.centipawn_loss, m.mate_before, m.mate_after);
    }
    NewGameAnalysis {
        analysis_run_id: run_id,
        game_id: game_id.into(),
        bee_color,
        avg_centipawn_loss: all.acpl(),
        worst_move_cp_loss: all.worst_loss,
        inaccuracies: all.inaccuracies,
        mistakes: all.mistakes,
        blunders: all.blunders,
        opening_avg_loss: phases[0].acpl(),
        middlegame_avg_loss: phases[1].acpl(),
        endgame_avg_loss: phases[2].acpl(),
    }
}
