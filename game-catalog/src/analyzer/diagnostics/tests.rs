use super::*;

fn judged(best: Score, played: Score, mv: &str) -> Judgment {
    let reference = Reference {
        engine: "Stockfish test".into(),
        best_move: "e2e4".into(),
        score: best.into(),
        pv: vec!["e2e4".into()],
    };
    judge(
        &reference,
        Search {
            score: played,
            best_move: Some(mv.into()),
            pv: vec![mv.into()],
        },
        50,
    )
}

#[test]
fn acceptance_uses_same_player_regret_and_does_not_require_exact_candidate() {
    let alternative = judged(Score::Cp(80), Score::Cp(30), "d2d4");
    assert_eq!(alternative.acceptable, Some(true));
    assert_eq!(alternative.regret_cp, Some(50));
    assert_eq!(
        judged(Score::Cp(80), Score::Cp(29), "d2d4").acceptable,
        Some(false)
    );
    assert_eq!(
        judged(Score::Cp(-230), Score::Cp(-290), "d2d4").regret_cp,
        Some(60)
    );
    let identical = judged(Score::Cp(80), Score::Cp(80), "e2e4");
    assert_eq!(identical.regret_cp, Some(0));
    assert_eq!(identical.acceptable, Some(true));
    let inconsistent = judged(Score::Cp(0), Score::Cp(100), "d2d4");
    assert!(inconsistent.reference_inconsistent);
    assert_eq!(inconsistent.acceptable, None);
}

#[test]
fn mates_are_not_fabricated_centipawns_and_losing_mate_distances_are_unassessed() {
    let mate_lost = judged(Score::Mate(5), Score::Cp(700), "d2d4");
    assert_eq!(mate_lost.regret_cp, None);
    assert_eq!(mate_lost.acceptable, Some(false));
    assert_eq!(
        judged(Score::Mate(5), Score::Mate(9), "d2d4").acceptable,
        Some(true)
    );
    assert_eq!(
        judged(Score::Cp(0), Score::Mate(-8), "d2d4").acceptable,
        Some(false)
    );
    assert_eq!(
        judged(Score::Mate(-8), Score::Mate(-4), "d2d4").acceptable,
        None
    );
    assert_eq!(
        judged(Score::Mate(-8), Score::Cp(-800), "d2d4").acceptable,
        None
    );
    assert_eq!(
        judged(Score::Mate(-8), Score::Mate(-8), "e2e4").regret_cp,
        Some(0)
    );
}

#[test]
fn separates_already_losing_and_preserves_state_threshold_boundaries() {
    assert_eq!(bucket(-200), "preventable");
    assert_eq!(bucket(-201), "already_losing");
    assert_eq!(state(-200), "equal");
    assert_eq!(state(200), "equal");
    assert_eq!(state(201), "winning");
    assert_eq!(state(-201), "losing");
    let (fixtures, _) =
        load_fixtures(&Path::new(env!("CARGO_MANIFEST_DIR")).join("../regressions")).unwrap();
    assert_eq!(fixtures.len(), 28);
    assert_eq!(
        fixtures
            .iter()
            .filter(|f| bucket(f.analysis.best_cp) == "already_losing")
            .count(),
        10
    );
    for f in fixtures.iter().filter(|f| f.category == "king_safety") {
        assert!(f.review["subcategory"].as_str().is_some());
    }
}

#[cfg(unix)]
mod process_tests {
    use super::*;
    use std::os::unix::fs::PermissionsExt;

    struct Directory(PathBuf);
    impl Directory {
        fn new() -> Self {
            let path =
                std::env::temp_dir().join(format!("bee-diagnostics-{}", uuid::Uuid::new_v4()));
            std::fs::create_dir_all(&path).unwrap();
            Self(path)
        }
        fn script(&self, name: &str, contents: &str) -> PathBuf {
            let path = self.0.join(name);
            std::fs::write(&path, contents).unwrap();
            std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o700)).unwrap();
            path
        }
    }
    impl Drop for Directory {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    fn fake_bee(dir: &Directory, on_go: &str) -> PathBuf {
        let options = variant_options("baseline")
            .unwrap()
            .into_iter()
            .map(|(k, v)| {
                let spec = if v == "true" {
                    "check default true".into()
                } else if v == "0" {
                    "spin default 0 min 0 max 7".into()
                } else {
                    format!("combo default {v} var {v}")
                };
                format!("echo 'option name {k} type {spec}'")
            })
            .collect::<Vec<_>>()
            .join("\n");
        dir.script(
            "bee",
            &format!(
                r#"#!/bin/sh
lmr=true
while IFS= read -r line; do
  printf '%s\n' "$line" >> '{}/bee.log'
  case "$line" in
    uci) echo 'id name bee-chess mock'; {options}; echo uciok ;;
    'setoption name UseLMR value false') lmr=false ;;
    isready) echo readyok ;;
    position*) position="$line" ;;
    'go depth '*) depth="${{line##* }}"; {on_go} ;;
  esac
done
"#,
                dir.0.display()
            ),
        )
    }

    #[test]
    fn malformed_pv_depth_mismatch_and_crashes_are_errors_but_deadlines_are_samples() {
        let dir = Directory::new();
        for on_go in [
            "exit 1",
            "echo 'bestmove e2e4'",
            "echo 'info depth 3 score cp 1 nodes 20 pv e2e4'; echo 'bestmove e2e4'",
            "echo 'info depth 4 score cp 1 nodes 20 pv e2e4 e7e4'; echo 'bestmove e2e4'",
            "echo 'info depth 4 score cp 1 nodes 20 pv e2e4'; echo 'bestmove d2d4'",
        ] {
            let path = fake_bee(&dir, on_go);
            assert!(
                bee::search(
                    &path,
                    &[],
                    &Position::startpos(),
                    "baseline",
                    &variant_options("baseline").unwrap(),
                    4,
                    Duration::from_secs(1)
                )
                .is_err(),
                "{on_go}"
            );
        }
        let path = fake_bee(&dir, "echo 'info depth 4 score cp 1 nodes 20 pv e2e4'");
        let sample = bee::search(
            &path,
            &[],
            &Position::startpos(),
            "baseline",
            &variant_options("baseline").unwrap(),
            4,
            Duration::from_millis(30),
        )
        .unwrap();
        assert_eq!(sample.status, "timed_out");
        assert!(sample.best_move.is_none());
        assert!(sample.nodes.is_none());
    }

    #[test]
    fn complete_pipeline_preserves_history_options_and_resumes_without_engine_searches() {
        let dir = Directory::new();
        let fixtures = dir.0.join("fixtures");
        for group in ["tactical", "positional", "endgame"] {
            std::fs::create_dir_all(fixtures.join(group)).unwrap();
        }
        let original =
            Path::new(env!("CARGO_MANIFEST_DIR")).join("../regressions/positional/Xtg1JApX-5.json");
        std::fs::copy(original, fixtures.join("positional/Xtg1JApX-5.json")).unwrap();
        let (loaded, _) = load_fixtures(&fixtures).unwrap();
        let f = &loaded[0];
        let bee = fake_bee(
            &dir,
            &format!(
                r#"if [ "$position" != 'position startpos moves {}' ]; then exit 3; fi
mv={}; if [ "$depth" != 4 ] || [ "$lmr" = false ]; then mv={}; fi
echo "info depth $depth score cp 44 nodes 100 pv $mv"
echo "bestmove $mv""#,
                f.history_uci.join(" "),
                f.bee_move,
                f.stockfish_candidate
            ),
        );
        let sf = dir.script("stockfish", &format!(r#"#!/bin/sh
while IFS= read -r line; do
  printf '%s\n' "$line" >> '{}/sf.log'
  case "$line" in
    uci)
      echo 'id name Stockfish mock'
      for option in Threads Hash MultiPV Ponder 'Skill Level' UCI_LimitStrength UCI_Chess960 SyzygyProbeLimit nodestime; do echo "option name $option type string"; done
      echo uciok ;;
    isready) echo readyok ;;
    position*) position="$line" ;;
    'go nodes '* )
      if [ "$position" != 'position startpos moves {}' ]; then exit 3; fi
      mv={}; cp=100
      case "$line" in *searchmoves*) mv="${{line##* }}" ;; esac
      if [ "$mv" = '{}' ]; then cp=-200; fi
      echo "info depth 10 score cp $cp pv $mv"
      echo "bestmove $mv" ;;
  esac
done
"#, dir.0.display(), f.history_uci.join(" "), f.stockfish_candidate, f.bee_move));
        let mut config = Config {
            bee,
            stockfish: sf,
            fixtures,
            output: dir.0.join("out"),
            depths: vec![4, 6],
            variants: vec!["baseline".into(), "no-lmr".into()],
            reference_nodes: 100_000,
            acceptable_cp: 50,
            timeout_seconds: 1,
            jobs: 2,
        };
        let report = run(&config, |_| {}).unwrap();
        let baseline = &report["positions"][0]["variants"][0];
        assert_eq!(baseline["first_acceptable_depth"], 6);
        assert_eq!(baseline["search_recovery"], true);
        assert_eq!(
            report["positions"][0]["same_depth_toggle_recoveries"][0]["depth"],
            4
        );
        let logs = ["bee.log", "sf.log"].map(|p| std::fs::read_to_string(dir.0.join(p)).unwrap());
        assert_eq!(logs[0].matches("uci\n").count(), 4);
        assert_eq!(
            logs[0].matches("setoption name UseLMR value false").count(),
            2
        );
        assert_eq!(logs[0].matches("ucinewgame").count(), 4);
        assert!(logs[1].contains(&format!("go nodes 100000 searchmoves {}", f.bee_move)));
        assert_eq!(run(&config, |_| {}).unwrap(), report);
        for (i, p) in ["bee.log", "sf.log"].iter().enumerate() {
            assert_eq!(std::fs::read_to_string(dir.0.join(p)).unwrap(), logs[i]);
        }
        let path = config.output.join("Xtg1JApX-5.json");
        let mut checkpoint: FixtureResult = read_json(&path).unwrap();
        checkpoint.samples.pop();
        write_json(&path, &checkpoint).unwrap();
        assert_eq!(run(&config, |_| {}).unwrap(), report);
        assert_eq!(
            std::fs::read_to_string(dir.0.join("bee.log"))
                .unwrap()
                .matches("go depth")
                .count(),
            5
        );
        assert_eq!(
            std::fs::read_to_string(dir.0.join("sf.log"))
                .unwrap()
                .matches("go nodes")
                .count(),
            logs[1].matches("go nodes").count()
        );
        let mut checkpoint: FixtureResult = read_json(&path).unwrap();
        checkpoint.samples[0].best_move = Some(f.stockfish_candidate.clone());
        checkpoint.samples[1].best_move = Some(f.bee_move.clone());
        let report = summary(&[checkpoint.clone()], &config.depths, &config.variants);
        assert_eq!(
            report["positions"][0]["variants"][0]["first_acceptable_depth"],
            4
        );
        assert_eq!(
            report["positions"][0]["variants"][0]["deepest_acceptable"],
            false
        );
        assert_eq!(
            report["positions"][0]["variants"][0]["later_regression"],
            true
        );
        checkpoint.reference.score = Score::Cp(-300).into();
        let report = summary(&[checkpoint.clone()], &config.depths, &config.variants);
        assert_eq!(report["positions"][0]["bucket"], "preventable");
        assert_eq!(report["positions"][0]["reference_bucket"], "already_losing");
        checkpoint.samples.push(checkpoint.samples[0].clone());
        assert!(validate_cached(&checkpoint, f, &config).is_err());
        config.acceptable_cp = 51;
        assert!(run(&config, |_| {}).unwrap_err().contains("changed"));
    }
}
