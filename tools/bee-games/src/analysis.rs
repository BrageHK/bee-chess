use std::path::PathBuf;

use bee_game_catalog::analysis::{GamePhase, MoveAnalysisRecord};
use bee_game_catalog::analyzer::{self, AnalyzeConfig, LossStats, Progress};
use bee_game_catalog::GameCatalog;

fn positive<T: std::str::FromStr + PartialEq + Default>(
    value: &str,
    flag: &str,
) -> Result<T, String> {
    value
        .parse::<T>()
        .ok()
        .filter(|v| *v != T::default())
        .ok_or_else(|| format!("{flag} must be a positive integer"))
}

fn parse_analyze(args: &[String]) -> Result<(AnalyzeConfig, usize), String> {
    let mut config = AnalyzeConfig {
        stockfish: super::repo_root().join("external/stockfish/src/stockfish"),
        players: vec![],
        nodes_per_position: 100_000,
        limit: None,
    };
    let mut top = 20;
    let mut args = args.iter();
    while let Some(flag) = args.next() {
        let value = args
            .next()
            .ok_or_else(|| format!("{flag} requires a value"))?;
        match flag.as_str() {
            "--stockfish" => config.stockfish = PathBuf::from(value),
            "--player" => config.players.push(value.clone()),
            "--nodes" => config.nodes_per_position = positive(value, flag)?,
            "--limit" => config.limit = Some(positive(value, flag)?),
            "--top" => top = positive(value, flag)?,
            _ => return Err(format!("unknown analyze flag: {flag}")),
        }
    }
    if config.players.is_empty() || config.players.iter().any(|p| p.trim().is_empty()) {
        return Err("at least one nonempty --player is required".into());
    }
    Ok((config, top))
}

pub fn analyze(catalog: &GameCatalog, args: &[String]) -> Result<(), String> {
    let (config, top) = parse_analyze(args)?;
    let result = analyzer::analyze(catalog, &config, |event| match event {
        Progress::Started {
            run_id,
            games,
            already_analyzed,
        } => eprintln!("run {run_id}: {games} selected games, {already_analyzed} already analyzed"),
        Progress::Completed { game_id, plies } => eprintln!("analyzed {game_id}: {plies} plies"),
        Progress::Rejected { game_id, reason } => eprintln!("rejected {game_id}: {reason}"),
    })
    .map_err(|e| e.to_string())?;
    println!("run {}: analyzed {} games / {} plies; skipped {} already-analyzed games; {} searches; {} rejected games",
        result.run_id, result.games_analyzed, result.plies_analyzed, result.games_already_analyzed,
        result.searches, result.rejected.len());
    print_report(catalog, result.run_id, top, None)?;
    if !result.rejected.is_empty() {
        return Err(format!(
            "{} games could not be analyzed; see rejection reasons above",
            result.rejected.len()
        ));
    }
    Ok(())
}

pub fn report(catalog: &GameCatalog, args: &[String]) -> Result<(), String> {
    let mut run = None;
    let mut top = 20;
    let mut phase = None;
    let mut args = args.iter();
    while let Some(flag) = args.next() {
        let value = args
            .next()
            .ok_or_else(|| format!("{flag} requires a value"))?;
        match flag.as_str() {
            "--run" => run = Some(positive::<u32>(value, flag)? as i64),
            "--top" => top = positive(value, flag)?,
            "--phase" => {
                phase = Some(match value.as_str() {
                    "opening" => GamePhase::Opening,
                    "middlegame" => GamePhase::Middlegame,
                    "endgame" => GamePhase::Endgame,
                    _ => return Err("unknown phase".into()),
                })
            }
            _ => return Err(format!("unknown report flag: {flag}")),
        }
    }
    print_report(catalog, run.ok_or("--run is required")?, top, phase)
}

fn print_report(
    catalog: &GameCatalog,
    run_id: i64,
    top: usize,
    phase: Option<GamePhase>,
) -> Result<(), String> {
    let run = catalog
        .analysis_run(run_id)
        .map_err(|e| e.to_string())?
        .ok_or("analysis run not found")?;
    println!(
        "engine: {}; nodes/position: {}; MultiPV: {}; analysis version: {}",
        run.engine,
        run.nodes_per_position
            .map(|n| n.to_string())
            .unwrap_or("unspecified".into()),
        run.multipv
            .map(|n| n.to_string())
            .unwrap_or("unspecified".into()),
        run.schema_version
    );
    println!(
        "configuration: {}",
        run.configuration.as_deref().unwrap_or("legacy/unspecified")
    );
    let moves = catalog
        .analyzed_moves(run_id, false, phase)
        .map_err(|e| e.to_string())?;
    let games: std::collections::HashSet<_> = moves.iter().map(|m| &m.game_id).collect();
    println!(
        "{} games / {} analyzed plies in report",
        games.len(),
        moves.len()
    );
    println!("side      phase        moves  cp moves    ACPL  >200cp  mate moves");
    for (name, bee) in [("Bee", true), ("Opponent", false)] {
        for phase in [
            None,
            Some(GamePhase::Opening),
            Some(GamePhase::Middlegame),
            Some(GamePhase::Endgame),
        ] {
            let mut stats = LossStats::default();
            for m in moves
                .iter()
                .filter(|m| m.is_bee == Some(bee) && phase.is_none_or(|p| m.phase == p))
            {
                stats.add(m.centipawn_loss, m.mate_before, m.mate_after);
            }
            println!(
                "{name:9} {:12} {:5} {:9} {:>7} {:7} {:11}",
                phase_label(phase),
                stats.moves,
                stats.cp_moves,
                stats
                    .acpl()
                    .map(|n| format!("{n:.1}"))
                    .unwrap_or("n/a".into()),
                stats.over_200,
                stats.mate_moves
            );
        }
    }
    println!("Highest-CPL Bee moves (mate scores excluded):");
    println!("game         ply phase       played best     before   after     CPL");
    for m in moves
        .iter()
        .filter(|m| m.is_bee == Some(true) && m.centipawn_loss.is_some())
        .take(top)
    {
        println!(
            "{:<12} {:3} {:11} {:6} {:6} {:>8} {:>7} {:7}",
            m.game_id,
            m.ply,
            phase_label(Some(m.phase)),
            m.played_move,
            m.best_move.as_deref().unwrap_or("-"),
            eval(m, true),
            eval(m, false),
            m.centipawn_loss.unwrap()
        );
        println!(
            "  FEN: {}\n  PV: {}",
            m.fen_before,
            m.pv.as_deref().unwrap_or("-")
        );
    }
    Ok(())
}

fn phase_label(phase: Option<GamePhase>) -> &'static str {
    match phase {
        None => "all",
        Some(GamePhase::Opening) => "opening",
        Some(GamePhase::Middlegame) => "middlegame",
        Some(GamePhase::Endgame) => "endgame",
    }
}

fn eval(m: &MoveAnalysisRecord, before: bool) -> String {
    let (cp, mate) = if before {
        (m.eval_before_cp, m.mate_before)
    } else {
        (m.eval_after_cp, m.mate_after)
    };
    if let Some(cp) = cp {
        format!("{cp:+}")
    } else if let Some(mate) = mate {
        format!("M{mate:+}")
    } else {
        "n/a".into()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn args(values: &[&str]) -> Vec<String> {
        values.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn requires_an_identity_and_positive_fixed_budget() {
        for values in [
            &[][..],
            &["--player", ""],
            &["--player", "Bee", "--nodes", "0"],
            &["--player", "Bee", "--nodes", "-1"],
            &["--player", "Bee", "--movetime", "10"],
            &["--player", "Bee", "--limit", "0"],
            &["--player"],
        ] {
            assert!(parse_analyze(&args(values)).is_err(), "{values:?}");
        }
        let (config, top) = parse_analyze(&args(&[
            "--player", "Bee", "--nodes", "250000", "--limit", "100",
        ]))
        .unwrap();
        assert_eq!(config.nodes_per_position, 250000);
        assert_eq!(config.limit, Some(100));
        assert_eq!(top, 20);
    }
}
