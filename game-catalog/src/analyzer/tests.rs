use super::*;

fn game(id: &str, moves: &str, bee: Color) -> GameRecord {
    GameRecord {
        id: id.into(),
        source: "test".into(),
        played_at: Some(1),
        white: Some(
            if bee == Color::White {
                "Bee"
            } else {
                "opponent"
            }
            .into(),
        ),
        black: Some(
            if bee == Color::Black {
                "Bee"
            } else {
                "opponent"
            }
            .into(),
        ),
        white_rating: None,
        black_rating: None,
        result: Some("0-1".into()),
        termination: None,
        time_control: None,
        rated: None,
        variant: Some("standard".into()),
        moves: Some(moves.into()),
        raw_pgn: None,
        imported_at: 1,
    }
}

fn config() -> AnalyzeConfig {
    AnalyzeConfig {
        stockfish: "unused".into(),
        players: vec!["Bee".into()],
        nodes_per_position: 100_000,
        limit: None,
    }
}

fn run(catalog: &GameCatalog) -> i64 {
    catalog
        .record_analysis_run(&NewAnalysisRun {
            engine: "fake".into(),
            nodes_per_position: Some(100_000),
            multipv: Some(1),
            schema_version: ANALYSIS_VERSION,
            created_at: 0,
            configuration: None,
        })
        .unwrap()
}

struct FakeEngine {
    scores: Vec<Score>,
    histories: Vec<Vec<String>>,
    fail_at: Option<usize>,
}

impl FakeEngine {
    fn new(scores: Vec<Score>) -> Self {
        Self {
            scores,
            histories: vec![],
            fail_at: None,
        }
    }
}

impl SearchEngine for FakeEngine {
    fn search(
        &mut self,
        history: &[String],
        position: &Position,
        nodes: u64,
    ) -> Result<Search, AnalysisError> {
        assert_eq!(nodes, 100_000);
        if self.fail_at == Some(history.len()) {
            return Err(AnalysisError::Engine("injected failure".into()));
        }
        self.histories.push(history.to_vec());
        let best = position
            .generate_legal_moves()
            .first()
            .copied()
            .map(move_uci);
        Ok(Search {
            score: self.scores[history.len()],
            pv: best.iter().cloned().collect(),
            best_move: best,
        })
    }
}

#[test]
fn normalizes_both_colors_and_rolls_up_only_bee() {
    for bee in [Color::White, Color::Black] {
        let catalog = GameCatalog::open_in_memory().unwrap();
        catalog.upsert_game(&game("g", "e4 e5 Nf3", bee)).unwrap();
        let run_id = run(&catalog);
        let mut engine = FakeEngine::new(vec![
            Score::Cp(40),
            Score::Cp(80),
            Score::Cp(-30),
            Score::Cp(100),
        ]);
        let result = analyze_with_engine(
            &catalog,
            &config(),
            &["bee".into()],
            run_id,
            &mut engine,
            |_| {},
        )
        .unwrap();
        assert_eq!(
            (
                result.games_analyzed,
                result.plies_analyzed,
                result.searches
            ),
            (1, 3, 4)
        );
        assert_eq!(engine.histories.last().unwrap(), &["e2e4", "e7e5", "g1f3"]);
        let moves = catalog.move_analyses(run_id, "g").unwrap();
        assert_eq!(
            moves.iter().map(|m| m.centipawn_loss).collect::<Vec<_>>(),
            [Some(120), Some(50), Some(70)]
        );
        assert_eq!(moves[0].eval_before_cp, Some(40));
        assert_eq!(moves[0].eval_after_cp, Some(-80));
        assert_eq!(moves[1].eval_before_cp, Some(80));
        assert_eq!(moves[1].eval_after_cp, Some(30));
        assert_eq!(moves[0].fen_before, Position::startpos().to_fen());
        assert_eq!(moves[1].mover_color, Some(Color::Black));
        assert_eq!(moves[1].is_bee, Some(bee == Color::Black));
        assert!(moves.iter().all(|m| m.pv.is_some()));
        let summary = catalog.game_analysis(run_id, "g").unwrap().unwrap();
        assert_eq!(
            summary.avg_centipawn_loss,
            Some(if bee == Color::White { 95.0 } else { 50.0 })
        );
        assert_eq!(
            summary.worst_move_cp_loss,
            Some(if bee == Color::White { 120 } else { 50 })
        );
        let worst = catalog
            .analyzed_moves(run_id, true, Some(GamePhase::Opening))
            .unwrap();
        assert_eq!(worst[0].ply, if bee == Color::White { 0 } else { 1 });
    }
}

#[test]
fn mate_transitions_do_not_invent_centipawns_and_negative_loss_is_clamped() {
    let catalog = GameCatalog::open_in_memory().unwrap();
    catalog
        .upsert_game(&game("g", "f3 e5 g4 Qh4#", Color::White))
        .unwrap();
    let run_id = run(&catalog);
    let mut engine = FakeEngine::new(vec![
        Score::Cp(20),
        Score::Cp(-40),
        Score::Cp(0),
        Score::Mate(1),
        Score::Mate(0),
    ]);
    analyze_with_engine(
        &catalog,
        &config(),
        &["bee".into()],
        run_id,
        &mut engine,
        |_| {},
    )
    .unwrap();
    let moves = catalog.move_analyses(run_id, "g").unwrap();
    assert_eq!(moves[0].centipawn_loss, Some(0));
    assert_eq!(
        (
            moves[2].mate_before,
            moves[2].mate_after,
            moves[2].centipawn_loss
        ),
        (None, Some(-1), None)
    );
    assert_eq!(
        (
            moves[3].mate_before,
            moves[3].mate_after,
            moves[3].centipawn_loss
        ),
        (Some(1), Some(0), None)
    );
    assert_eq!(
        catalog
            .game_analysis(run_id, "g")
            .unwrap()
            .unwrap()
            .avg_centipawn_loss,
        Some(0.0)
    );
}

#[test]
fn restart_skips_all_100_completed_games_without_searching_and_config_is_scoped() {
    let path = std::env::temp_dir().join(format!("bee-analysis-{}.sqlite3", uuid::Uuid::new_v4()));
    let catalog = GameCatalog::open(&path).unwrap();
    for i in 0..100 {
        catalog
            .upsert_game(&game(&format!("g{i:03}"), "e4 e5", Color::White))
            .unwrap();
    }
    let run_id = run(&catalog);
    let mut engine = FakeEngine::new(vec![Score::Cp(0); 3]);
    let first = analyze_with_engine(
        &catalog,
        &config(),
        &["bee".into()],
        run_id,
        &mut engine,
        |_| {},
    )
    .unwrap();
    assert_eq!(
        (first.games_analyzed, first.plies_analyzed, first.searches),
        (100, 200, 300)
    );
    let saved = catalog.analyzed_moves(run_id, false, None).unwrap();
    drop(catalog);
    let catalog = GameCatalog::open(&path).unwrap();
    let mut engine = FakeEngine::new(vec![]); // Any search would panic.
    let second = analyze_with_engine(
        &catalog,
        &config(),
        &["bee".into()],
        run_id,
        &mut engine,
        |_| {},
    )
    .unwrap();
    assert_eq!(
        (
            second.games_analyzed,
            second.games_already_analyzed,
            second.searches
        ),
        (0, 100, 0)
    );
    assert_eq!(catalog.analyzed_moves(run_id, false, None).unwrap(), saved);
    let other_run = run(&catalog);
    let mut engine = FakeEngine::new(vec![Score::Cp(0); 3]);
    let third = analyze_with_engine(
        &catalog,
        &AnalyzeConfig {
            limit: Some(1),
            ..config()
        },
        &["bee".into()],
        other_run,
        &mut engine,
        |_| {},
    )
    .unwrap();
    assert_eq!(third.games_analyzed, 1);
    let fourth = analyze_with_engine(
        &catalog,
        &AnalyzeConfig {
            limit: Some(1),
            ..config()
        },
        &["bee".into()],
        other_run,
        &mut engine,
        |_| {},
    )
    .unwrap();
    assert_eq!(fourth.searches, 0);
    drop(catalog);
    std::fs::remove_file(path).unwrap();
}

#[test]
fn engine_failure_leaves_no_partial_game_and_retry_succeeds() {
    let catalog = GameCatalog::open_in_memory().unwrap();
    catalog
        .upsert_game(&game("g", "e4 e5 Nf3", Color::White))
        .unwrap();
    let run_id = run(&catalog);
    let mut engine = FakeEngine::new(vec![Score::Cp(0); 4]);
    engine.fail_at = Some(2);
    assert!(analyze_with_engine(
        &catalog,
        &config(),
        &["bee".into()],
        run_id,
        &mut engine,
        |_| {}
    )
    .is_err());
    assert!(catalog.move_analyses(run_id, "g").unwrap().is_empty());
    assert!(catalog.game_analysis(run_id, "g").unwrap().is_none());
    engine.fail_at = None;
    analyze_with_engine(
        &catalog,
        &config(),
        &["bee".into()],
        run_id,
        &mut engine,
        |_| {},
    )
    .unwrap();
    assert_eq!(catalog.move_analyses(run_id, "g").unwrap().len(), 3);
}

#[test]
fn rejects_bad_games_before_searching_and_keeps_processing_valid_games() {
    let catalog = GameCatalog::open_in_memory().unwrap();
    let mut bad = game("variant", "e4", Color::White);
    bad.variant = Some("chess960".into());
    catalog.upsert_game(&bad).unwrap();
    bad.id = "setup".into();
    bad.variant = Some("standard".into());
    bad.raw_pgn = Some("[SetUp \"1\"]\n[FEN \"anything\"]".into());
    catalog.upsert_game(&bad).unwrap();
    catalog
        .upsert_game(&game("broken", "e4 e5 nonsense", Color::White))
        .unwrap();
    catalog
        .upsert_game(&game("empty", "", Color::White))
        .unwrap();
    catalog
        .upsert_game(&game("valid", "e4 e5", Color::White))
        .unwrap();
    let run_id = run(&catalog);
    let mut engine = FakeEngine::new(vec![Score::Cp(0); 3]);
    let result = analyze_with_engine(
        &catalog,
        &config(),
        &["bee".into()],
        run_id,
        &mut engine,
        |_| {},
    )
    .unwrap();
    assert_eq!(result.rejected.len(), 4);
    assert_eq!(result.searches, 3);
    assert_eq!(
        catalog.analyzed_moves(run_id, false, None).unwrap().len(),
        2
    );
}

#[test]
fn phase_boundaries_and_weighted_totals_are_explicit() {
    assert_eq!(
        classify_phase(&Position::startpos(), 19),
        GamePhase::Opening
    );
    assert_eq!(
        classify_phase(&Position::startpos(), 20),
        GamePhase::Middlegame
    );
    let endgame = Position::from_fen("4k3/8/8/8/8/8/8/R3K2R w KQ - 0 1").unwrap();
    assert_eq!(classify_phase(&endgame, 0), GamePhase::Endgame);
    let mut stats = LossStats::default();
    for cp in [0, 49, 50, 99, 100, 199, 200, 201] {
        stats.add(Some(cp), None, None);
    }
    stats.add(None, None, Some(-1));
    assert_eq!((stats.moves, stats.cp_moves, stats.mate_moves), (9, 8, 1));
    assert_eq!(
        (
            stats.inaccuracies,
            stats.mistakes,
            stats.blunders,
            stats.over_200
        ),
        (2, 2, 2, 1)
    );
    assert_eq!(stats.acpl(), Some(898.0 / 8.0));
}

#[test]
fn replay_handles_castling_en_passant_and_promotion() {
    let castle = replay(&game("g", "e4 e5 Nf3 Nc6 Bc4 Bc5 O-O", Color::White)).unwrap();
    assert_eq!(castle.moves.last().unwrap(), "e1g1");
    let ep = replay(&game("g", "e4 a6 e5 d5 exd6", Color::White)).unwrap();
    assert_eq!(ep.moves.last().unwrap(), "e5d6");
    let promo = replay(&game(
        "g",
        "a4 h5 a5 h4 a6 h3 axb7 hxg2 bxa8=Q",
        Color::White,
    ))
    .unwrap();
    assert_eq!(promo.moves.last().unwrap(), "b7a8q");
}

#[test]
#[ignore = "requires a local Stockfish binary (BEE_TEST_STOCKFISH or external/stockfish/src/stockfish)"]
fn real_stockfish_is_reproducible_and_restart_does_no_searches() {
    let path = std::env::var_os("BEE_TEST_STOCKFISH")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            Path::new(env!("CARGO_MANIFEST_DIR")).join("../external/stockfish/src/stockfish")
        });
    let config = AnalyzeConfig {
        stockfish: path,
        ..config()
    };
    let catalog = GameCatalog::open_in_memory().unwrap();
    let sample = game("g", "f3 e5 g4 Qh4#", Color::White);
    catalog.upsert_game(&sample).unwrap();
    let first = analyze(&catalog, &config, |_| {}).unwrap();
    let moves = catalog.move_analyses(first.run_id, "g").unwrap();
    assert_eq!(moves[3].mate_after, Some(0));
    assert_eq!(moves[2].mate_after, Some(-1));
    let second = analyze(&catalog, &config, |_| {}).unwrap();
    assert_eq!((second.run_id, second.searches), (first.run_id, 0));
    let other = GameCatalog::open_in_memory().unwrap();
    other.upsert_game(&sample).unwrap();
    let third = analyze(&other, &config, |_| {}).unwrap();
    assert_eq!(other.move_analyses(third.run_id, "g").unwrap(), moves);
}
