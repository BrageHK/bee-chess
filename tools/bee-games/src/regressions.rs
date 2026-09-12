use bee_game_catalog::analyzer::diagnostics::{self, Config};
use std::path::PathBuf;

pub fn diagnose(args: &[String]) -> Result<(), String> {
    let root = super::repo_root();
    let mut config = Config {
        bee: root.join("target/release/bee"),
        stockfish: root.join("external/stockfish/src/stockfish"),
        fixtures: root.join("regressions"),
        output: root.join("data/games/regression-diagnostics"),
        depths: vec![4, 6, 8, 10, 12],
        variants: [
            "baseline",
            "no-lmr",
            "no-null-move",
            "no-see",
            "no-king-safety",
        ]
        .map(String::from)
        .to_vec(),
        reference_nodes: 1_000_000,
        acceptable_cp: 50,
        timeout_seconds: 30,
        jobs: 4,
    };
    let mut args = args.iter();
    while let Some(flag) = args.next() {
        let value = args
            .next()
            .ok_or_else(|| format!("{flag} requires a value"))?;
        match flag.as_str() {
            "--bee" => config.bee = PathBuf::from(value),
            "--stockfish" => config.stockfish = PathBuf::from(value),
            "--fixtures" => config.fixtures = PathBuf::from(value),
            "--output" => config.output = PathBuf::from(value),
            "--depths" => {
                config.depths = value
                    .split(',')
                    .map(|d| d.parse().map_err(|_| "invalid --depths".to_string()))
                    .collect::<Result<_, _>>()?
            }
            "--variants" => config.variants = value.split(',').map(String::from).collect(),
            "--reference-nodes" => {
                config.reference_nodes = value.parse().map_err(|_| "invalid --reference-nodes")?
            }
            "--acceptable-cp" => {
                config.acceptable_cp = value.parse().map_err(|_| "invalid --acceptable-cp")?
            }
            "--timeout-seconds" => {
                config.timeout_seconds = value.parse().map_err(|_| "invalid --timeout-seconds")?
            }
            "--jobs" => config.jobs = value.parse().map_err(|_| "invalid --jobs")?,
            _ => return Err(format!("unrecognized flag: {flag}")),
        }
    }
    let report = diagnostics::run(&config, |message| eprintln!("{message}"))?;
    println!(
        "{} fixtures ({} unique FENs): {} preventable, {} already losing",
        report["fixtures"], report["unique_fens"], report["preventable"], report["already_losing"]
    );
    println!(
        "Bucket          Variant          Depth   Acceptable / Completed   Unassessed   Timeouts"
    );
    for row in report["totals"].as_array().ok_or("invalid summary")? {
        if row["category"] != "all" {
            continue;
        }
        println!(
            "{:<15} {:<16} {:>5} {:>12} / {:<9} {:>10} {:>10}",
            row["bucket"].as_str().unwrap_or(""),
            row["variant"].as_str().unwrap_or(""),
            row["depth"].as_u64().unwrap_or(0),
            row["acceptable"].as_u64().unwrap_or(0),
            row["completed"].as_u64().unwrap_or(0),
            row["unassessed"].as_u64().unwrap_or(0),
            row["timed_out"].as_u64().unwrap_or(0)
        );
    }
    println!(
        "Results and resumable checkpoints: {}",
        config.output.display()
    );
    Ok(())
}
