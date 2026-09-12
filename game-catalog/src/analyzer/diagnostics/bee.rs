//! Bee-specific UCI boundary. A fresh process makes fixed-depth searches cold
//! and independent of fixture order, other variants, and resumed runs.
use std::collections::BTreeMap;
use std::io::{BufRead, BufReader, Write};
use std::path::Path;
use std::process::{Child, ChildStdin, Command, Stdio};
use std::sync::mpsc::{self, Receiver, RecvTimeoutError};
use std::time::{Duration, Instant};

use bee_chess_core::Position;
use serde::{Deserialize, Serialize};

use super::{play, Evaluation};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Sample {
    pub variant: String,
    pub requested_depth: u32,
    pub status: String,
    pub best_move: Option<String>,
    pub score: Option<Evaluation>,
    pub depth: Option<u32>,
    pub nodes: Option<u64>,
    pub elapsed_ms: u64,
    pub pv: Vec<String>,
    pub counters: BTreeMap<String, u64>,
}

struct Process {
    child: Child,
    input: ChildStdin,
    output: Receiver<std::io::Result<String>>,
}

impl Drop for Process {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

impl Process {
    fn send(&mut self, command: &str) -> Result<(), String> {
        writeln!(self.input, "{command}").map_err(|e| e.to_string())?;
        self.input.flush().map_err(|e| e.to_string())
    }

    // None is exclusively a deadline, never EOF or malformed protocol output.
    fn receive(&self, deadline: Instant) -> Result<Option<String>, String> {
        let Some(remaining) = deadline.checked_duration_since(Instant::now()) else {
            return Ok(None);
        };
        match self.output.recv_timeout(remaining) {
            Ok(line) => line.map(Some).map_err(|e| e.to_string()),
            Err(RecvTimeoutError::Timeout) => Ok(None),
            Err(RecvTimeoutError::Disconnected) => Err("Bee exited before bestmove/readyok".into()),
        }
    }
}

pub(super) fn search(
    path: &Path,
    history: &[String],
    position: &Position,
    variant: &str,
    options: &BTreeMap<String, String>,
    depth: u32,
    timeout: Duration,
) -> Result<Sample, String> {
    let mut child = Command::new(path)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit())
        .spawn()
        .map_err(|e| format!("starting Bee: {e}"))?;
    let input = child.stdin.take().expect("piped stdin");
    let output = child.stdout.take().expect("piped stdout");
    let (sender, receiver) = mpsc::channel();
    std::thread::spawn(move || {
        for line in BufReader::new(output).lines() {
            if sender.send(line).is_err() {
                break;
            }
        }
    });
    let mut engine = Process {
        child,
        input,
        output: receiver,
    };
    let startup_deadline = Instant::now() + Duration::from_secs(10);
    engine.send("uci")?;
    let mut name = String::new();
    let mut advertised = BTreeMap::new();
    loop {
        let line = engine
            .receive(startup_deadline)?
            .ok_or("Bee handshake timed out")?;
        if line == "uciok" {
            break;
        }
        if let Some(value) = line.strip_prefix("id name ") {
            name = value.into();
        }
        if let Some(value) = line.strip_prefix("option name ") {
            if let Some((name, spec)) = value.split_once(" type ") {
                advertised.insert(name.to_string(), spec.to_string());
            }
        }
    }
    if !name.starts_with("bee-chess") {
        return Err(format!("expected Bee, received {name:?}"));
    }
    for (name, value) in options {
        let spec = advertised
            .get(name)
            .ok_or_else(|| format!("Bee lacks option {name}"))?;
        let valid = match spec.split_whitespace().next() {
            Some("check") => value == "true" || value == "false",
            Some("combo") => spec.split(" var ").skip(1).any(|v| v == value),
            Some("spin") => value.parse::<u32>().is_ok(),
            _ => false,
        };
        if !valid {
            return Err(format!("invalid Bee option {name}={value}"));
        }
        engine.send(&format!("setoption name {name} value {value}"))?;
    }
    engine.send("ucinewgame")?;
    engine.send("isready")?;
    while engine
        .receive(startup_deadline)?
        .ok_or("Bee readiness timed out")?
        != "readyok"
    {}
    engine.send(&format!("position startpos moves {}", history.join(" ")))?;
    let started = Instant::now();
    engine.send(&format!("go depth {depth}"))?;
    let mut sample = Sample {
        variant: variant.into(),
        requested_depth: depth,
        status: "timed_out".into(),
        best_move: None,
        score: None,
        depth: None,
        nodes: None,
        elapsed_ms: 0,
        pv: Vec::new(),
        counters: BTreeMap::new(),
    };
    let mut info = None;
    while let Some(line) = engine.receive(started + timeout)? {
        if let Some(parsed) = parse_info(&line) {
            info = Some(parsed);
        }
        if let Some(rest) = line.strip_prefix("bestmove ") {
            let best = rest.split_whitespace().next().ok_or("empty Bee bestmove")?;
            let parsed = info.ok_or("Bee bestmove without complete exact search info")?;
            if parsed.depth != depth || parsed.pv.first().map(String::as_str) != Some(best) {
                return Err(
                    "Bee did not complete the requested depth or bestmove disagrees with PV".into(),
                );
            }
            let mut cursor = position.clone();
            for mv in &parsed.pv {
                play(&mut cursor, mv)?;
            }
            sample.status = "completed".into();
            sample.best_move = Some(best.into());
            sample.score = Some(parsed.score);
            sample.depth = Some(parsed.depth);
            sample.nodes = Some(parsed.nodes);
            sample.pv = parsed.pv;
            sample.counters = parsed.counters;
            break;
        }
    }
    sample.elapsed_ms = started.elapsed().as_millis() as u64;
    Ok(sample)
}

struct Info {
    score: Evaluation,
    depth: u32,
    nodes: u64,
    pv: Vec<String>,
    counters: BTreeMap<String, u64>,
}

fn parse_info(line: &str) -> Option<Info> {
    let tokens: Vec<_> = line.split_whitespace().collect();
    if tokens.first() != Some(&"info")
        || tokens.get(1) == Some(&"string")
        || tokens.contains(&"lowerbound")
        || tokens.contains(&"upperbound")
    {
        return None;
    }
    let value = |key| {
        tokens
            .iter()
            .position(|t| *t == key)
            .and_then(|i| tokens.get(i + 1))
            .copied()
    };
    if value("multipv").is_some_and(|v| v != "1") {
        return None;
    }
    let i = tokens.iter().position(|t| *t == "score")?;
    let number = tokens.get(i + 2)?.parse::<i32>().ok()?;
    // Bee currently emits signed PLIES in its UCI mate field, unlike Stockfish.
    let score = match *tokens.get(i + 1)? {
        "cp" => Evaluation {
            cp: Some(number),
            mate_plies: None,
        },
        "mate" => Evaluation {
            cp: None,
            mate_plies: Some(number),
        },
        _ => return None,
    };
    let pv_start = tokens.iter().position(|t| *t == "pv")?;
    let mut counters = BTreeMap::new();
    for key in [
        "lmr_attempts",
        "lmr_fail_lows",
        "lmr_researches",
        "nmp_attempts",
        "nmp_cutoffs",
        "delta_attempts",
        "delta_pruned",
        "see_attempts",
        "see_pruned",
        "tbhits",
    ] {
        if let Some(v) = value(key).and_then(|v| v.parse().ok()) {
            counters.insert(key.into(), v);
        }
    }
    Some(Info {
        score,
        depth: value("depth")?.parse().ok()?,
        nodes: value("nodes")?.parse().ok()?,
        pv: tokens[pv_start + 1..]
            .iter()
            .map(|t| t.to_string())
            .collect(),
        counters,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn preserves_bee_mate_plies_and_ignores_bounds_and_secondary_pvs() {
        let line = "info depth 6 score mate -4 nodes 234 lmr_attempts 9 pv e2e4 e7e5";
        let info = parse_info(line).unwrap();
        assert_eq!(info.score.mate_plies, Some(-4));
        assert_eq!(info.nodes, 234);
        assert_eq!(info.counters["lmr_attempts"], 9);
        assert!(parse_info(&format!("{line} upperbound")).is_none());
        assert!(parse_info(&format!("{line} multipv 2")).is_none());
        assert!(parse_info("info string depth 6 score cp 80 nodes 20 pv e2e4").is_none());
    }
}
