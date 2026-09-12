//! Stored analysis reports shared by the offline export and Lab. No engine calls.

use std::collections::{HashMap, HashSet};

use serde::Serialize;

use crate::analysis::{AnalysisRun, GamePhase, MoveAnalysisRecord};
use crate::{Color, GameCatalog, GameFilter, GameRecord, Result};

#[derive(Debug, Default, Serialize)]
pub struct MoveStats {
    pub bee_moves: usize,
    pub cp_moves: usize,
    pub avg_cpl: Option<f64>,
    /// These thresholds are strict and cumulative, not exclusive buckets.
    pub over_100: usize,
    pub over_200: usize,
    pub over_400: usize,
    pub mate_moves: usize,
    /// Positive CP drop although the stored best move equals the played move.
    /// Keep raw totals but expose this finite-search disagreement for review.
    pub score_disagreements: usize,
}

#[derive(Debug, Serialize)]
pub struct PhaseStats {
    pub phase: GamePhase,
    #[serde(flatten)]
    pub stats: MoveStats,
}

#[derive(Debug, Serialize)]
pub struct AnalysisSummary {
    pub games_analyzed: usize,
    #[serde(flatten)]
    pub stats: MoveStats,
    pub phases: Vec<PhaseStats>,
}

#[derive(Debug, Serialize)]
pub struct ReviewMove {
    #[serde(flatten)]
    pub analysis: MoveAnalysisRecord,
    pub white: Option<String>,
    pub black: Option<String>,
    pub result: Option<String>,
    pub original_game_url: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct LossReview {
    pub worst_move: ReviewMove,
    pub total_cpl: i64,
    /// Share of recorded CP loss, not a claim that this move caused the loss.
    /// Mate transitions are excluded from this denominator.
    pub worst_move_share: Option<f64>,
}

#[derive(Debug, Serialize)]
pub struct StoredAnalysisReport {
    pub run: AnalysisRun,
    /// Summary and loss-game ranking cover the entire run; table filters only
    /// affect `moves`/`matching_moves`, so phase comparisons remain meaningful.
    pub summary: AnalysisSummary,
    pub matching_moves: usize,
    pub moves: Vec<ReviewMove>,
    pub losses: Vec<LossReview>,
}

#[derive(Debug, Clone)]
pub struct ReportFilter {
    pub phase: Option<GamePhase>,
    pub losses_only: bool,
    pub over_cp: Option<u32>,
    pub offset: usize,
    pub limit: usize,
}

impl Default for ReportFilter {
    fn default() -> Self {
        Self {
            phase: None,
            losses_only: false,
            over_cp: None,
            offset: 0,
            limit: 50,
        }
    }
}

pub fn report(
    catalog: &GameCatalog,
    run_id: i64,
    filter: &ReportFilter,
) -> Result<Option<StoredAnalysisReport>> {
    let Some(run) = catalog.analysis_run(run_id)? else {
        return Ok(None);
    };
    let all_moves = catalog.analyzed_moves(run_id, false, None)?;
    let games: HashMap<_, _> = catalog
        .games(&GameFilter::all())?
        .into_iter()
        .map(|g| (g.id.clone(), g))
        .collect();
    let bee_moves: Vec<_> = all_moves
        .iter()
        .filter(|m| m.is_bee == Some(true))
        .collect();
    let summary = AnalysisSummary {
        games_analyzed: all_moves
            .iter()
            .map(|m| &m.game_id)
            .collect::<HashSet<_>>()
            .len(),
        stats: stats(&bee_moves),
        phases: [
            GamePhase::Opening,
            GamePhase::Middlegame,
            GamePhase::Endgame,
        ]
        .into_iter()
        .map(|phase| PhaseStats {
            phase,
            stats: stats(
                &bee_moves
                    .iter()
                    .copied()
                    .filter(|m| m.phase == phase)
                    .collect::<Vec<_>>(),
            ),
        })
        .collect(),
    };
    let matches: Vec<_> = bee_moves
        .iter()
        .copied()
        .filter(|m| {
            m.centipawn_loss.is_some_and(|cp| {
                filter
                    .over_cp
                    .is_none_or(|min| i64::from(cp) > i64::from(min))
            }) && filter.phase.is_none_or(|phase| m.phase == phase)
                && (!filter.losses_only || is_loss(m, games.get(&m.game_id)))
        })
        .collect();
    let matching_moves = matches.len();
    let moves = matches
        .into_iter()
        .skip(filter.offset)
        .take(filter.limit)
        .map(|m| review(m, games.get(&m.game_id)))
        .collect();
    let mut loss_totals = HashMap::<&str, i64>::new();
    for m in &bee_moves {
        if is_loss(m, games.get(&m.game_id)) {
            *loss_totals.entry(&m.game_id).or_default() += i64::from(m.centipawn_loss.unwrap_or(0));
        }
    }
    // analyzed_moves is already CPL-descending with stable game/ply tie breaks.
    let mut seen = HashSet::new();
    let losses = bee_moves
        .into_iter()
        .filter(|m| {
            m.centipawn_loss.is_some()
                && is_loss(m, games.get(&m.game_id))
                && seen.insert(m.game_id.as_str())
        })
        .take(10)
        .map(|m| {
            let total = loss_totals[m.game_id.as_str()];
            LossReview {
                worst_move: review(m, games.get(&m.game_id)),
                total_cpl: total,
                worst_move_share: (total > 0)
                    .then(|| f64::from(m.centipawn_loss.unwrap()) / total as f64),
            }
        })
        .collect();
    Ok(Some(StoredAnalysisReport {
        run,
        summary,
        matching_moves,
        moves,
        losses,
    }))
}

/// Move lookup is scoped to a completed game and the selected analysis run.
pub fn review_move(catalog: &GameCatalog, run_id: i64, move_id: i64) -> Result<Option<ReviewMove>> {
    let Some(m) = catalog
        .analyzed_moves(run_id, true, None)?
        .into_iter()
        .find(|m| m.id == move_id)
    else {
        return Ok(None);
    };
    let game = catalog.game(&m.game_id)?;
    Ok(Some(review(&m, game.as_ref())))
}

fn stats(moves: &[&MoveAnalysisRecord]) -> MoveStats {
    let mut stats = MoveStats {
        bee_moves: moves.len(),
        ..Default::default()
    };
    let mut total = 0i64;
    for m in moves {
        if m.mate_before.is_some() || m.mate_after.is_some() {
            stats.mate_moves += 1;
        }
        if let Some(cp) = m.centipawn_loss {
            stats.score_disagreements +=
                usize::from(cp > 0 && m.best_move.as_deref() == Some(m.played_move.as_str()));
            stats.cp_moves += 1;
            total += i64::from(cp);
            stats.over_100 += usize::from(cp > 100);
            stats.over_200 += usize::from(cp > 200);
            stats.over_400 += usize::from(cp > 400);
        }
    }
    stats.avg_cpl = (stats.cp_moves > 0).then(|| total as f64 / stats.cp_moves as f64);
    stats
}

fn is_loss(m: &MoveAnalysisRecord, game: Option<&GameRecord>) -> bool {
    matches!(
        (m.mover_color, game.and_then(|g| g.result.as_deref())),
        (Some(Color::White), Some("0-1")) | (Some(Color::Black), Some("1-0"))
    )
}

fn review(m: &MoveAnalysisRecord, game: Option<&GameRecord>) -> ReviewMove {
    ReviewMove {
        analysis: m.clone(),
        white: game.and_then(|g| g.white.clone()),
        black: game.and_then(|g| g.black.clone()),
        result: game.and_then(|g| g.result.clone()),
        original_game_url: game
            .filter(|g| {
                g.source == "lichess"
                    && !g.id.is_empty()
                    && g.id.chars().all(|c| c.is_ascii_alphanumeric())
            })
            .map(|g| format!("https://lichess.org/{}#{}", g.id, m.ply + 1)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analysis::{NewAnalysisRun, NewGameAnalysis, NewMoveAnalysis};

    fn add_game(
        c: &GameCatalog,
        run: i64,
        id: &str,
        bee: Color,
        result: &str,
        losses: &[Option<i32>],
    ) {
        let game = GameRecord {
            id: id.into(),
            source: "lichess".into(),
            played_at: None,
            white: Some(
                if bee == Color::White {
                    "Bee"
                } else {
                    "Opponent"
                }
                .into(),
            ),
            black: Some(
                if bee == Color::Black {
                    "Bee"
                } else {
                    "Opponent"
                }
                .into(),
            ),
            white_rating: None,
            black_rating: None,
            result: Some(result.into()),
            termination: None,
            time_control: None,
            rated: None,
            variant: Some("standard".into()),
            moves: Some(vec!["e4"; losses.len()].join(" ")),
            raw_pgn: None,
            imported_at: 0,
        };
        c.upsert_game(&game).unwrap();
        let moves: Vec<_> = losses
            .iter()
            .enumerate()
            .map(|(ply, cp)| {
                let mover = if ply % 2 == 0 {
                    Color::White
                } else {
                    Color::Black
                };
                NewMoveAnalysis {
                    analysis_run_id: run,
                    game_id: id.into(),
                    ply: ply as u32,
                    fen_before: bee_chess_core::Position::startpos().to_fen(),
                    played_move: "e2e4".into(),
                    best_move: Some("d2d4".into()),
                    eval_before_cp: Some(0),
                    eval_after_cp: cp.map(|c| -c),
                    centipawn_loss: *cp,
                    mate_before: None,
                    mate_after: cp.is_none().then_some(-1),
                    phase: match ply {
                        0 => GamePhase::Opening,
                        1 => GamePhase::Middlegame,
                        _ => GamePhase::Endgame,
                    },
                    mover_color: Some(mover),
                    is_bee: Some(mover == bee),
                    pv: Some("d2d4 d7d5".into()),
                }
            })
            .collect();
        c.record_complete_game_analysis(
            &moves,
            &NewGameAnalysis {
                analysis_run_id: run,
                game_id: id.into(),
                bee_color: bee,
                avg_centipawn_loss: None,
                worst_move_cp_loss: None,
                inaccuracies: 0,
                mistakes: 0,
                blunders: 0,
                opening_avg_loss: None,
                middlegame_avg_loss: None,
                endgame_avg_loss: None,
            },
        )
        .unwrap();
    }

    fn fixture() -> GameCatalog {
        let c = GameCatalog::open_in_memory().unwrap();
        let run = NewAnalysisRun {
            engine: "SF".into(),
            nodes_per_position: Some(100_000),
            multipv: Some(1),
            schema_version: 2,
            created_at: 0,
            configuration: None,
        };
        let first = c.record_analysis_run(&run).unwrap();
        let second = c.record_analysis_run(&run).unwrap();
        add_game(
            &c,
            first,
            "A",
            Color::White,
            "0-1",
            &[Some(100), Some(999), Some(401)],
        );
        add_game(&c, first, "B", Color::Black, "1-0", &[Some(999), Some(201)]);
        add_game(&c, first, "C", Color::White, "1-0", &[Some(101)]);
        add_game(&c, first, "D", Color::White, "1/2-1/2", &[None]);
        add_game(&c, second, "E", Color::White, "0-1", &[Some(9999)]);
        c
    }

    #[test]
    fn weighted_summary_excludes_opponents_other_runs_and_mates_from_cp() {
        let c = fixture();
        let r = report(&c, 1, &ReportFilter::default()).unwrap().unwrap();
        assert_eq!(r.summary.games_analyzed, 4);
        let s = &r.summary.stats;
        assert_eq!((s.bee_moves, s.cp_moves, s.mate_moves), (5, 4, 1));
        assert_eq!(s.avg_cpl, Some(803.0 / 4.0));
        assert_eq!((s.over_100, s.over_200, s.over_400), (3, 2, 1));
        assert_eq!(r.summary.phases[0].stats.avg_cpl, Some(100.5));
        assert_eq!(
            r.moves
                .iter()
                .map(|m| m.analysis.centipawn_loss)
                .collect::<Vec<_>>(),
            [Some(401), Some(201), Some(101), Some(100)]
        );
        assert_eq!(r.losses.len(), 2);
        assert_eq!(r.losses[0].worst_move.analysis.game_id, "A");
        assert_eq!(r.losses[0].total_cpl, 501);
        assert_eq!(r.losses[0].worst_move_share, Some(401.0 / 501.0));
        assert_eq!(r.losses[1].worst_move_share, Some(1.0));
        assert!(report(&c, 999, &ReportFilter::default()).unwrap().is_none());
    }

    #[test]
    fn filters_and_pagination_do_not_change_the_run_summary() {
        let c = fixture();
        let filter = ReportFilter {
            phase: Some(GamePhase::Middlegame),
            over_cp: Some(200),
            losses_only: true,
            ..Default::default()
        };
        let r = report(&c, 1, &filter).unwrap().unwrap();
        assert_eq!(r.matching_moves, 1);
        assert_eq!(r.moves[0].analysis.game_id, "B");
        assert_eq!(r.summary.stats.bee_moves, 5);
        let r = report(
            &c,
            1,
            &ReportFilter {
                offset: 2,
                limit: 1,
                ..Default::default()
            },
        )
        .unwrap()
        .unwrap();
        assert_eq!(r.matching_moves, 4);
        assert_eq!(r.moves[0].analysis.game_id, "C");
        let r = report(
            &c,
            1,
            &ReportFilter {
                over_cp: Some(401),
                ..Default::default()
            },
        )
        .unwrap()
        .unwrap();
        assert!(r.moves.is_empty());
    }

    #[test]
    fn review_links_and_lookups_are_scoped_to_the_run_and_bee() {
        let c = fixture();
        let r = report(&c, 1, &ReportFilter::default()).unwrap().unwrap();
        let id = r.moves[0].analysis.id;
        let m = review_move(&c, 1, id).unwrap().unwrap();
        assert_eq!(
            m.original_game_url.as_deref(),
            Some("https://lichess.org/A#3")
        );
        assert_eq!(m.analysis.pv.as_deref(), Some("d2d4 d7d5"));
        assert!(review_move(&c, 2, id).unwrap().is_none());
        let opponent = c.move_analyses(1, "A").unwrap()[1].id;
        assert!(review_move(&c, 1, opponent).unwrap().is_none());
    }

    #[test]
    fn read_only_catalog_cannot_write_or_create_files() {
        let path =
            std::env::temp_dir().join(format!("bee-report-readonly-{}", uuid::Uuid::new_v4()));
        assert!(GameCatalog::open_read_only(&path).is_err());
        assert!(!path.exists());
        drop(GameCatalog::open(&path).unwrap());
        let c = GameCatalog::open_read_only(&path).unwrap();
        assert!(c
            .record_analysis_run(&NewAnalysisRun {
                engine: "SF".into(),
                nodes_per_position: None,
                multipv: None,
                schema_version: 2,
                created_at: 0,
                configuration: None
            })
            .is_err());
        drop(c);
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn best_move_score_disagreements_remain_in_raw_totals_but_are_flagged() {
        let c = fixture();
        let mut m = c.move_analyses(1, "A").unwrap().remove(0);
        m.best_move = Some(m.played_move.clone());
        let s = stats(&[&m]);
        assert_eq!(s.score_disagreements, 1);
        assert_eq!(s.avg_cpl, Some(100.0));
        m.centipawn_loss = Some(0);
        assert_eq!(stats(&[&m]).score_disagreements, 0);
    }
}
