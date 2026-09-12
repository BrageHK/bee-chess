//! Offline search diagnostics over reviewed, history-preserving fixtures.
//! No catalog writes, engine tuning, or automatic chess-motif classification.
mod bee;
mod report;
#[cfg(test)]
mod tests;

use std::collections::{BTreeMap, HashSet, VecDeque};
use std::path::{Path, PathBuf};
use std::sync::{mpsc, Mutex};
use std::time::Duration;

use bee_chess_core::Position;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use super::{binary_digest, move_uci, uci::Score, uci::Stockfish, Search, SearchEngine};
pub use bee::Sample;
pub use report::summary;

#[derive(Debug, Clone)]
pub struct Config {
    pub bee: PathBuf,
    pub stockfish: PathBuf,
    pub fixtures: PathBuf,
    pub output: PathBuf,
    pub depths: Vec<u32>,
    pub variants: Vec<String>,
    pub reference_nodes: u64,
    pub acceptable_cp: i32,
    pub timeout_seconds: u64,
    pub jobs: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Fixture {
    pub schema_version: u32,
    pub id: String,
    pub category: String,
    pub fen: String,
    pub history_uci: Vec<String>,
    pub bee_move: String,
    pub stockfish_candidate: String,
    pub expected_property: String,
    pub review: Value,
    pub source: Value,
    pub analysis: HistoricalAnalysis,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HistoricalAnalysis {
    pub best_cp: i32,
    pub played_cp: i32,
    pub regret_cp: i32,
    pub best_pv: String,
    pub played_pv: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Evaluation {
    pub cp: Option<i32>,
    pub mate_plies: Option<i32>,
}

impl From<Score> for Evaluation {
    fn from(score: Score) -> Self {
        Self {
            cp: score.cp(),
            mate_plies: score.mate(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Reference {
    pub engine: String,
    pub best_move: String,
    pub score: Evaluation,
    pub pv: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Judgment {
    pub score: Evaluation,
    pub pv: Vec<String>,
    pub regret_cp: Option<i32>,
    /// None means incomparable mate losses or an inconsistent reference.
    pub acceptable: Option<bool>,
    pub reference_inconsistent: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FixtureResult {
    pub fixture: Fixture,
    pub bucket: String,
    pub state_transition: String,
    pub reference: Reference,
    pub judgments: BTreeMap<String, Judgment>,
    pub samples: Vec<Sample>,
}

/// All effective options are explicit; no book/tablebase or previous search state.
pub fn variant_options(variant: &str) -> Result<BTreeMap<String, String>, String> {
    let mut options: BTreeMap<_, _> = [
        ("Evaluator", "Positional"),
        ("TTReuse", "PerSearch"),
        ("OpeningBook", "None"),
        ("SyzygyProbeLimit", "0"),
        ("UseTT", "true"),
        ("UseQuiescence", "true"),
        ("UseLMR", "true"),
        ("UseNullMove", "true"),
        ("UseAdaptiveNullMove", "true"),
        ("UseDeltaPruning", "true"),
        ("UseEnhancedQuiescence", "true"),
        ("UseSee", "true"),
        ("UseMobility", "true"),
        ("UseKingSafety", "true"),
    ]
    .into_iter()
    .map(|(k, v)| (k.into(), v.into()))
    .collect();
    let disabled = match variant {
        "baseline" => None,
        "no-lmr" => Some("UseLMR"),
        "no-null-move" => Some("UseNullMove"),
        "no-see" => Some("UseSee"),
        "no-king-safety" => Some("UseKingSafety"),
        _ => return Err(format!("unknown variant {variant}")),
    };
    if let Some(option) = disabled {
        options.insert(option.into(), "false".into());
    }
    Ok(options)
}

fn play(position: &mut Position, token: &str) -> Result<(), String> {
    let mv = position
        .generate_legal_moves()
        .into_iter()
        .find(|m| move_uci(*m) == token)
        .ok_or_else(|| format!("illegal move {token} in {}", position.to_fen()))?;
    position.make_move(mv);
    Ok(())
}

fn position(fixture: &Fixture) -> Result<Position, String> {
    let mut position = Position::startpos();
    for mv in &fixture.history_uci {
        play(&mut position, mv)?;
    }
    if position.to_fen() != fixture.fen {
        return Err(format!("{}: history/FEN mismatch", fixture.id));
    }
    Ok(position)
}

fn load_fixtures(root: &Path) -> Result<(Vec<Fixture>, BTreeMap<String, String>), String> {
    let mut fixtures = Vec::new();
    let mut digests = BTreeMap::new();
    for group in ["tactical", "positional", "endgame"] {
        for entry in std::fs::read_dir(root.join(group)).map_err(|e| e.to_string())? {
            let path = entry.map_err(|e| e.to_string())?.path();
            if path.extension().is_none_or(|e| e != "json") {
                continue;
            }
            let f: Fixture = read_json(&path)?;
            if f.schema_version != 1
                || f.id.is_empty()
                || !f.id.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'-')
                || f.source["ply"].as_u64() != Some(f.history_uci.len() as u64)
                || f.source["analysis_method"] != 3
                || f.expected_property.trim().is_empty()
                || f.analysis.regret_cp <= 200
                || f.analysis.regret_cp
                    != f.analysis
                        .best_cp
                        .saturating_sub(f.analysis.played_cp)
                        .max(0)
            {
                return Err(format!(
                    "invalid version/provenance/regret in {}",
                    path.display()
                ));
            }
            let root_position = position(&f)?;
            for (first, pv) in [
                (&f.bee_move, &f.analysis.played_pv),
                (&f.stockfish_candidate, &f.analysis.best_pv),
            ] {
                if pv.split_whitespace().next() != Some(first) {
                    return Err(format!("{}: PV/choice mismatch", f.id));
                }
                let mut cursor = root_position.clone();
                for mv in pv.split_whitespace() {
                    play(&mut cursor, mv)?;
                }
            }
            if digests
                .insert(
                    f.id.clone(),
                    binary_digest(&path).map_err(|e| e.to_string())?,
                )
                .is_some()
            {
                return Err(format!("duplicate fixture {}", f.id));
            }
            fixtures.push(f);
        }
    }
    if fixtures.is_empty() {
        return Err("no regression fixtures found".into());
    }
    fixtures.sort_by(|a, b| a.id.cmp(&b.id));
    Ok((fixtures, digests))
}

pub fn bucket(before: i32) -> &'static str {
    if before < -200 {
        "already_losing"
    } else {
        "preventable"
    }
}

pub fn state(cp: i32) -> &'static str {
    if cp > 200 {
        "winning"
    } else if cp < -200 {
        "losing"
    } else {
        "equal"
    }
}

fn judge(reference: &Reference, search: Search, threshold: i32) -> Judgment {
    let score = Evaluation::from(search.score);
    let identical = search.best_move.as_deref() == Some(&reference.best_move);
    let regret = reference
        .score
        .cp
        .zip(score.cp)
        .map(|(b, p)| b.saturating_sub(p).max(0));
    let inconsistent = !identical
        && match (
            reference.score.cp,
            score.cp,
            reference.score.mate_plies,
            score.mate_plies,
        ) {
            (Some(b), Some(p), _, _) => p.saturating_sub(b) > threshold,
            (Some(_), _, _, Some(p)) => p > 0,
            (_, Some(_), Some(b), _) => b <= 0,
            (_, _, Some(b), Some(p)) => b <= 0 && p > 0,
            _ => false,
        };
    let acceptable = if identical {
        Some(true)
    } else if inconsistent {
        None
    } else if let Some(cp) = regret {
        Some(cp <= threshold)
    } else {
        match (reference.score.mate_plies, score.mate_plies) {
            (Some(b), Some(p)) if b > 0 && p > 0 => Some(true),
            (Some(b), Some(p)) if b <= 0 && p <= 0 => None,
            (Some(b), _) if b > 0 => Some(false),
            (_, Some(p)) if p <= 0 => Some(false),
            _ => None,
        }
    };
    Judgment {
        score,
        pv: search.pv,
        regret_cp: if identical { Some(0) } else { regret },
        acceptable,
        reference_inconsistent: inconsistent,
    }
}

fn assess_move(
    engine: &mut Stockfish,
    result: &mut FixtureResult,
    position: &Position,
    mv: &str,
    config: &Config,
) -> Result<(), String> {
    if result.judgments.contains_key(mv) {
        return Ok(());
    }
    let search = if result.reference.best_move == mv {
        Search {
            score: match (result.reference.score.cp, result.reference.score.mate_plies) {
                (Some(v), _) => Score::Cp(v),
                (_, Some(v)) => Score::Mate(v),
                _ => return Err("missing reference score".into()),
            },
            best_move: Some(mv.into()),
            pv: result.reference.pv.clone(),
        }
    } else {
        engine
            .search(
                &result.fixture.history_uci,
                position,
                config.reference_nodes,
                Some(mv),
            )
            .map_err(|e| e.to_string())?
    };
    result.judgments.insert(
        mv.into(),
        judge(&result.reference, search, config.acceptable_cp),
    );
    Ok(())
}

fn diagnose_fixture(
    config: &Config,
    fixture: &Fixture,
    progress: &mpsc::Sender<String>,
) -> Result<FixtureResult, String> {
    let path = config.output.join(format!("{}.json", fixture.id));
    let mut cached: Option<FixtureResult> = if path.exists() {
        Some(read_json(&path)?)
    } else {
        None
    };
    if let Some(result) = &cached {
        validate_cached(result, fixture, config)?;
        if result.samples.len() == config.depths.len() * config.variants.len() {
            let _ = progress.send(format!(
                "{}: resumed {} searches",
                fixture.id,
                result.samples.len()
            ));
            return Ok(cached.take().unwrap());
        }
    }
    let position = position(fixture)?;
    let mut engine =
        Stockfish::spawn(&config.stockfish, Duration::from_secs(120)).map_err(|e| e.to_string())?;
    let mut result = match cached {
        Some(result) => result,
        None => {
            let root = engine
                .search(
                    &fixture.history_uci,
                    &position,
                    config.reference_nodes,
                    None,
                )
                .map_err(|e| e.to_string())?;
            FixtureResult {
                fixture: fixture.clone(),
                bucket: bucket(fixture.analysis.best_cp).into(),
                state_transition: format!(
                    "{}_to_{}",
                    state(fixture.analysis.best_cp),
                    state(fixture.analysis.played_cp)
                ),
                reference: Reference {
                    engine: engine.name.clone(),
                    best_move: root.best_move.ok_or("terminal fixture")?,
                    score: root.score.into(),
                    pv: root.pv,
                },
                judgments: BTreeMap::new(),
                samples: Vec::new(),
            }
        }
    };
    // Also recheck the recorded candidate and historical mistake against the
    // deeper reference; a changed reference must be visible in the report.
    for mv in [
        &fixture.bee_move,
        &fixture.stockfish_candidate,
        &result.reference.best_move.clone(),
    ] {
        assess_move(&mut engine, &mut result, &position, mv, config)?;
    }
    write_json(&path, &result)?;
    for variant in &config.variants {
        let options = variant_options(variant)?;
        for &depth in &config.depths {
            if result
                .samples
                .iter()
                .any(|s| s.variant == *variant && s.requested_depth == depth)
            {
                continue;
            }
            let sample = bee::search(
                &config.bee,
                &fixture.history_uci,
                &position,
                variant,
                &options,
                depth,
                Duration::from_secs(config.timeout_seconds),
            )?;
            if let Some(mv) = &sample.best_move {
                assess_move(&mut engine, &mut result, &position, mv, config)?;
            }
            let _ = progress.send(format!(
                "{} {variant} depth {depth}: {} {} nodes={}",
                fixture.id,
                sample.status,
                sample.best_move.as_deref().unwrap_or("—"),
                sample.nodes.map_or("—".into(), |n| n.to_string())
            ));
            result.samples.push(sample);
            write_json(&path, &result)?;
        }
    }
    Ok(result)
}

fn validate_cached(
    result: &FixtureResult,
    fixture: &Fixture,
    config: &Config,
) -> Result<(), String> {
    if serde_json::to_value(&result.fixture).map_err(|e| e.to_string())?
        != serde_json::to_value(fixture).map_err(|e| e.to_string())?
    {
        return Err(format!("{}: checkpoint fixture changed", fixture.id));
    }
    let mut seen = HashSet::new();
    for s in &result.samples {
        if !config.variants.contains(&s.variant)
            || !config.depths.contains(&s.requested_depth)
            || !seen.insert((&s.variant, s.requested_depth))
            || !matches!(s.status.as_str(), "completed" | "timed_out")
            || (s.status == "completed"
                && (s.depth != Some(s.requested_depth)
                    || s.nodes.is_none()
                    || s.score.is_none()
                    || s.best_move
                        .as_ref()
                        .is_none_or(|m| !result.judgments.contains_key(m))))
        {
            return Err(format!("{}: invalid checkpoint sample", fixture.id));
        }
    }
    Ok(())
}

/// Atomic checkpoints after every search. Identical restarts skip completed
/// samples (including recorded timeouts); use a new output for changed limits.
pub fn run(config: &Config, mut progress: impl FnMut(String)) -> Result<Value, String> {
    if config.depths.is_empty()
        || config.depths.iter().any(|d| *d == 0 || *d > 64)
        || config.depths.windows(2).any(|w| w[0] >= w[1])
        || config.variants.is_empty()
        || config.variants.iter().collect::<HashSet<_>>().len() != config.variants.len()
        || config.reference_nodes == 0
        || config.acceptable_cp < 0
        || config.timeout_seconds == 0
        || config.timeout_seconds > 86400
        || config.jobs == 0
        || config.jobs > 32
    {
        return Err("invalid depths (strictly increasing 1..64), variants, nodes, threshold, timeout (1..86400), or jobs (1..32)".into());
    }
    let variants: BTreeMap<_, _> = config
        .variants
        .iter()
        .map(|v| Ok((v.clone(), variant_options(v)?)))
        .collect::<Result<_, String>>()?;
    let (fixtures, digests) = load_fixtures(&config.fixtures)?;
    let manifest = json!({
        "schema_version": 1, "method_version": 1,
        "bee_sha256": binary_digest(&config.bee).map_err(|e| e.to_string())?,
        "stockfish_sha256": binary_digest(&config.stockfish).map_err(|e| e.to_string())?,
        "fixture_sha256": digests, "depths": config.depths, "variants": variants,
        "reference_nodes": config.reference_nodes, "acceptable_cp": config.acceptable_cp,
        "timeout_seconds": config.timeout_seconds, "jobs": config.jobs,
        "platform": format!("{}-{}", std::env::consts::OS, std::env::consts::ARCH),
        "history": "startpos-with-all-moves", "bee_reset": "fresh-process-per-search",
        "scores": "mover-pov; mate distances in signed plies",
        "stockfish": {"threads": 1, "hash_mb": 16, "multipv": 1, "ponder": false,
            "skill_level": 20, "limit_strength": false, "chess960": false,
            "syzygy_probe_limit": 0, "nodestime": 0, "reset": "ucinewgame-per-search",
            "score_selection": "last-exact-primary-pv", "regret": "same-root-searchmoves"},
        "state_threshold_cp": 200,
    });
    std::fs::create_dir_all(&config.output).map_err(|e| e.to_string())?;
    let lock = std::fs::OpenOptions::new()
        .create(true)
        .truncate(false)
        .write(true)
        .open(config.output.join(".lock"))
        .map_err(|e| e.to_string())?;
    lock.try_lock()
        .map_err(|e| format!("diagnostic output is already in use: {e}"))?;
    let manifest_path = config.output.join("manifest.json");
    if manifest_path.exists() {
        let previous: Value = read_json(&manifest_path)?;
        if previous != manifest {
            return Err("configuration/binary/corpus changed; use a new --output directory".into());
        }
    } else {
        write_json(&manifest_path, &manifest)?;
    }
    let queue = Mutex::new(VecDeque::from(fixtures));
    let (updates, messages) = mpsc::channel();
    let (finished, results) = mpsc::channel();
    std::thread::scope(|scope| {
        for _ in 0..config.jobs {
            let (queue, updates, finished) = (&queue, updates.clone(), finished.clone());
            scope.spawn(move || loop {
                let Some(fixture) = queue.lock().expect("fixture queue").pop_front() else {
                    break;
                };
                let result = diagnose_fixture(config, &fixture, &updates)
                    .map_err(|e| format!("{}: {e}", fixture.id));
                if let Err(error) = &result {
                    let _ = updates.send(format!("error: {error}"));
                }
                let _ = finished.send(result);
            });
        }
        drop(updates);
        drop(finished);
        for message in messages {
            progress(message);
        }
    });
    let mut results: Vec<_> = results.into_iter().collect::<Result<_, _>>()?;
    results.sort_by(|a, b| a.fixture.id.cmp(&b.fixture.id));
    let report = summary(&results, &config.depths, &config.variants);
    write_json(&config.output.join("summary.json"), &report)?;
    Ok(report)
}

fn read_json<T: serde::de::DeserializeOwned>(path: &Path) -> Result<T, String> {
    serde_json::from_slice(&std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?)
        .map_err(|e| format!("{}: {e}", path.display()))
}

fn write_json(path: &Path, value: &impl Serialize) -> Result<(), String> {
    let temporary = path.with_extension("json.tmp");
    let mut file = std::fs::File::create(&temporary).map_err(|e| e.to_string())?;
    serde_json::to_writer_pretty(&mut file, value).map_err(|e| e.to_string())?;
    file.sync_all().map_err(|e| e.to_string())?;
    std::fs::rename(temporary, path).map_err(|e| e.to_string())
}
