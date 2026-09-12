//! Structural validation of manually reviewed catalog fixtures, not an engine
//! strength test. Every move and variation is replayed with bee-chess-core.

use super::*;
use serde::Deserialize;

#[derive(Deserialize)]
struct Fixture {
    schema_version: u32,
    id: String,
    category: String,
    fen: String,
    history_uci: Vec<String>,
    bee_move: String,
    stockfish_candidate: String,
    expected_property: String,
    source: Source,
    analysis: Evaluation,
}

#[derive(Deserialize)]
struct Source {
    game_id: String,
    ply: usize,
    analysis_method: i64,
    nodes_per_position: u64,
    configuration: serde_json::Value,
}

#[derive(Deserialize)]
struct Evaluation {
    best_cp: i32,
    played_cp: i32,
    regret_cp: i32,
    best_pv: String,
    played_pv: String,
}

fn play(position: &mut Position, uci: &str) {
    let mv = position
        .generate_legal_moves()
        .into_iter()
        .find(|m| move_uci(*m) == uci)
        .unwrap_or_else(|| panic!("illegal fixture move {uci} in {}", position.to_fen()));
    position.make_move(mv);
}

#[test]
fn regression_corpus_replays_exact_history_and_both_root_variations() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../regressions");
    let mut ids = HashSet::new();
    for group in ["tactical", "endgame", "positional"] {
        for entry in std::fs::read_dir(root.join(group)).unwrap() {
            let path = entry.unwrap().path();
            if path.extension().is_none_or(|ext| ext != "json") {
                continue;
            }
            let f: Fixture = serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
            assert_eq!(f.schema_version, 1, "{}", f.id);
            assert!(ids.insert(f.id.clone()), "duplicate fixture {}", f.id);
            assert_eq!(f.id, format!("{}-{}", f.source.game_id, f.source.ply));
            assert!(matches!(
                f.category.as_str(),
                "tactical_miss"
                    | "bad_exchange"
                    | "hanging_loose_piece"
                    | "king_safety"
                    | "passed_pawn"
                    | "rook_activity"
                    | "pawn_structure"
                    | "endgame_technique"
            ));
            assert!(!f.expected_property.trim().is_empty());
            assert_eq!(f.source.analysis_method, 3);
            assert!(f.source.nodes_per_position > 0);
            assert_eq!(f.source.configuration["move_loss"], "same-root-searchmoves");
            assert_eq!(
                f.source.configuration["engine_sha256"]
                    .as_str()
                    .unwrap()
                    .len(),
                64
            );
            assert_eq!(f.history_uci.len(), f.source.ply);
            let mut position = Position::startpos();
            for mv in &f.history_uci {
                play(&mut position, mv);
            }
            assert_eq!(position.to_fen(), f.fen, "{}", f.id);
            assert_ne!(f.bee_move, f.stockfish_candidate);
            assert!(f.analysis.regret_cp > 200);
            assert_eq!(
                f.analysis.regret_cp,
                f.analysis
                    .best_cp
                    .saturating_sub(f.analysis.played_cp)
                    .max(0)
            );
            for (first, pv) in [
                (&f.stockfish_candidate, &f.analysis.best_pv),
                (&f.bee_move, &f.analysis.played_pv),
            ] {
                assert_eq!(pv.split_whitespace().next(), Some(first.as_str()));
                let mut cursor = position.clone();
                for mv in pv.split_whitespace() {
                    play(&mut cursor, mv);
                }
            }
        }
    }
    assert!(
        !ids.is_empty(),
        "the reviewed corpus must contain positions"
    );
}
