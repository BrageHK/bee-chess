use bee_engine::chess::Position;
use bee_engine::engine::Engine;
use bee_engine::search::{mate_in_plies, StopSignal};
use bee_engine::tablebase::{Wdl, TABLEBASE_WIN};
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

const DRAW: &str = "k7/P7/2K5/8/8/8/8/8 w - - 0 1";
const WIN: &str = "7k/8/8/8/8/8/8/3QK3 w - - 0 1";
const FIXTURES: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/syzygy");

struct WdlOnly(PathBuf);
impl WdlOnly {
    fn new(names: &[&str]) -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let path = std::env::temp_dir().join(format!(
            "bee syzygy {} {}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir_all(&path).unwrap();
        for name in names {
            std::fs::copy(PathBuf::from(FIXTURES).join(name), path.join(name)).unwrap();
        }
        Self(path)
    }
}
impl Drop for WdlOnly {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn engine(fen: &str, path: &str) -> Engine {
    let mut engine = Engine::new();
    engine.set_position(Position::from_fen(fen).unwrap());
    engine.set_syzygy_path(path).unwrap();
    engine
}

#[test]
fn wdl_only_draw_overrides_positive_eval_and_preserves_the_draw() {
    let files = WdlOnly::new(&[
        "KPvK.rtbw",
        "KQvK.rtbw",
        "KRvK.rtbw",
        "KBvK.rtbw",
        "KNvK.rtbw",
    ]);
    let mut bee = engine(DRAW, files.0.to_str().unwrap());
    let before = bee.position().clone();
    let result = bee.search(4);
    assert_eq!(result.score, 0);
    assert_eq!(result.nodes, 1);
    assert!(result.tablebase.root_resolved);
    assert_eq!(result.tablebase.root_hit.unwrap().wdl, Wdl::Draw);
    assert!(result.tablebase.root_hit.unwrap().eval_cp > 0);
    assert_eq!(bee.position(), &before);
    let mv = result.best_move.unwrap();
    bee.apply_move(mv.from(), mv.to(), mv.flag().promotion_kind())
        .unwrap();
    assert_eq!(bee.search(2).score, 0);
}

#[test]
fn missing_material_and_disabling_tables_restore_baseline_search() {
    let files = WdlOnly::new(&["KRvK.rtbw"]);
    let mut bee = engine(WIN, files.0.to_str().unwrap());
    let baseline = engine(WIN, "").search(2);
    let miss = bee.search(2);
    assert_eq!(
        (miss.score, miss.best_move, miss.nodes),
        (baseline.score, baseline.best_move, baseline.nodes)
    );
    assert!(miss.tablebase.probes > 0);
    assert_eq!(miss.tablebase.hits, 0);
    bee.set_syzygy_path(FIXTURES).unwrap();
    assert_eq!(bee.search(2).score, TABLEBASE_WIN);
    for path in ["", "/bee/nonexistent/syzygy"] {
        bee.set_syzygy_path(path).unwrap();
        let result = bee.search(2);
        assert_eq!(
            (result.score, result.best_move, result.nodes),
            (baseline.score, baseline.best_move, baseline.nodes)
        );
        assert_eq!(result.tablebase.probes, 0);
    }
}

#[test]
fn limit_changes_clear_cached_knowledge_and_defaults_stay_disabled() {
    let mut bee = Engine::new();
    assert_eq!(bee.syzygy_path(), "");
    assert_eq!(bee.syzygy_probe_limit(), 6);
    bee.set_position(Position::from_fen(DRAW).unwrap());
    let baseline = bee.search(3);
    bee.set_syzygy_path(FIXTURES).unwrap();
    assert_eq!(bee.search(3).score, 0);
    bee.set_syzygy_probe_limit(0);
    let result = bee.search(3);
    assert_eq!(
        (result.score, result.best_move, result.nodes),
        (baseline.score, baseline.best_move, baseline.nodes)
    );
    assert_eq!(result.tablebase.probes, 0);
}

#[test]
fn dtz_resolves_win_loss_and_fifty_move_draw_without_reporting_a_mate() {
    for (fen, score) in [
        (WIN, TABLEBASE_WIN),
        ("7k/8/8/8/8/8/8/3QK3 b - - 0 1", -TABLEBASE_WIN),
        ("7k/8/8/8/8/8/8/3QK3 w - - 99 1", 0),
    ] {
        let mut bee = engine(fen, FIXTURES);
        let result = bee.search(4);
        assert_eq!(result.score, score, "{fen}");
        assert!(result.tablebase.root_resolved);
        assert!(result.tablebase.root_hit.unwrap().dtz.is_some());
        assert_eq!(mate_in_plies(result.score), None);
    }
}

#[test]
fn terminal_and_claimable_draws_take_precedence() {
    for (fen, score, has_move) in [
        ("7k/6Q1/5K2/8/8/8/8/8 b - - 100 1", -30_000, false),
        ("7k/5K2/6Q1/8/8/8/8/8 b - - 0 1", 0, false),
        ("7k/8/8/8/8/8/8/3QK3 w - - 100 1", 0, true),
    ] {
        let result = engine(fen, FIXTURES).search(2);
        assert_eq!(result.score, score);
        assert_eq!(result.best_move.is_some(), has_move);
        assert_eq!(result.tablebase.probes, 0);
    }
}

#[test]
fn timed_search_stops_after_resolving_the_root_and_resets_stats() {
    let mut bee = engine(DRAW, FIXTURES);
    let mut iterations = 0;
    let result = bee.search_for_time(Duration::from_secs(30), StopSignal::new(), |_| {
        iterations += 1
    });
    assert_eq!(iterations, 1);
    assert_eq!(result.score, 0);
    let again = bee.search(2);
    assert_eq!(result.tablebase.probes, again.tablebase.probes);
    assert_eq!(result.tablebase.hits, again.tablebase.hits);
    let stop = StopSignal::new();
    stop.request_stop();
    let result = bee.search_for_time(Duration::from_secs(30), stop, |_| panic!("already stopped"));
    assert!(result.best_move.is_some());
    assert_eq!(result.tablebase.probes, 0);
}

#[test]
fn independent_instances_reload_their_own_table_paths() {
    let files = WdlOnly::new(&["KRvK.rtbw"]);
    let mut full = engine(DRAW, FIXTURES);
    let mut partial = engine(DRAW, files.0.to_str().unwrap());
    for _ in 0..3 {
        assert!(full.search(2).tablebase.hits > 0);
        // K vs K is a built-in draw in Fathom; the missing KPvK root must
        // still miss, even if a later pawn capture reaches that built-in.
        assert!(partial.search(2).tablebase.root_hit.is_none());
    }
}

#[test]
fn wdl_hit_after_capture_changes_search_without_dtz() {
    let files = WdlOnly::new(&["KQvK.rtbw"]);
    let fen = "7k/8/8/8/3r4/8/8/3QK3 w - - 0 1";
    let mut bee = engine(fen, files.0.to_str().unwrap());
    let before = bee.position().clone();
    let result = bee.search(1);
    assert_eq!(result.score, TABLEBASE_WIN);
    assert!(result.tablebase.root_hit.is_none());
    assert!(result.tablebase.hits > 0);
    let mv = result.best_move.unwrap();
    assert_eq!(format!("{}{}", mv.from(), mv.to()), "d1d4");
    assert_eq!(bee.position(), &before);
}

#[test]
fn uci_options_telemetry_and_empty_path_work_without_debug_mode() {
    let files = WdlOnly::new(&[
        "KPvK.rtbw",
        "KQvK.rtbw",
        "KRvK.rtbw",
        "KBvK.rtbw",
        "KNvK.rtbw",
    ]);
    let input = format!("uci\nsetoption name SyzygyPath value {}\nsetoption name syzygyprobelimit value 7\nisready\nposition fen {DRAW}\ngo depth 2\nsetoption name SyzygyProbeLimit value 8\nsetoption name SyzygyProbeLimit value -1\nsetoption name SyzygyProbeLimit value nonsense\nsetoption name SyzygyPath value\nposition fen {DRAW}\ngo depth 2\nquit\n", files.0.display());
    let mut bee = Engine::new();
    let mut output = Vec::new();
    bee_engine::uci::run(input.as_bytes(), &mut output, &mut bee).unwrap();
    let output = String::from_utf8(output).unwrap();
    assert!(output.contains("option name SyzygyPath type string default <empty>"));
    assert!(output.contains("option name SyzygyProbeLimit type spin default 6 min 0 max 7"));
    assert!(output.contains("readyok"));
    assert!(output.contains("info string tb hit pieces=3 wdl=draw exact=true root=true eval_cp="));
    assert!(output.contains("info string tb probes="));
    assert!(output.contains(" tbhits 0"));
    assert_eq!(output.matches("bestmove ").count(), 2);
    assert_eq!(bee.syzygy_path(), "");
    assert_eq!(bee.syzygy_probe_limit(), 7);
}

#[test]
fn repeated_history_bypasses_dtz_and_threefold_remains_a_draw() {
    let mut bee = engine(WIN, FIXTURES);
    for cycle in 0..2 {
        for uci in ["e1e2", "h8h7", "e2e1", "h7h8"] {
            let mv = bee_engine::uci::UciMove::parse(uci).unwrap();
            bee.apply_move(mv.from, mv.to, mv.promotion).unwrap();
        }
        let result = bee.search(2);
        assert!(!result.tablebase.root_resolved);
        if cycle == 0 {
            assert!(result.tablebase.root_hit.unwrap().dtz.is_none());
        } else {
            assert!(bee.is_threefold_repetition());
            assert_eq!(result.score, 0);
            assert_eq!(result.tablebase.probes, 0);
        }
    }
}

#[test]
fn lichess_vnfytgny_stays_out_of_tablebase_range_and_finishes_by_fifty_moves() {
    // https://lichess.org/VNfytGnY: after 156.Bxd2 and after 206.Be5.
    // The game never reached <=7 pieces. This is a fallback/rule-draw
    // regression, not a claim that Syzygy proves its earlier endgame drawn.
    let after_capture = "8/2k5/b3p1p1/5p1p/5P1P/4KP2/3B4/8 b - - 0 156";
    let final_position = "8/8/2b1p1p1/3kBp1p/5P1P/4KP2/8/8 b - - 100 206";
    for fen in [after_capture, final_position] {
        assert_eq!(
            bee_engine::tablebase::piece_count(&Position::from_fen(fen).unwrap()),
            11
        );
        let mut bee = engine(fen, FIXTURES);
        bee.set_syzygy_probe_limit(7);
        let result = bee.search(2);
        let baseline = engine(fen, "").search(2);
        assert_eq!(
            (result.score, result.best_move, result.nodes),
            (baseline.score, baseline.best_move, baseline.nodes)
        );
        assert_eq!(result.tablebase.probes, 0);
        if fen == final_position {
            assert_eq!(result.score, 0);
        }
    }
}
