//! Small synchronous UCI adapter for offline Stockfish analysis.
//! Each operation has a deadline; Drop kills and reaps even a broken engine.

use std::io::{BufRead, BufReader, Write};
use std::path::Path;
use std::process::{Child, ChildStdin, Command, Stdio};
use std::sync::mpsc::{self, Receiver};
use std::time::{Duration, Instant};

use bee_chess_core::Position;

use super::{move_uci, AnalysisError, Search, SearchEngine};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Score {
    Cp(i32),
    /// Signed plies: positive means the side to move mates, negative means
    /// it is mated. Zero is an already checkmated side to move.
    Mate(i32),
}

impl Score {
    pub fn cp(self) -> Option<i32> {
        match self {
            Self::Cp(v) => Some(v),
            _ => None,
        }
    }

    pub fn mate(self) -> Option<i32> {
        match self {
            Self::Mate(v) => Some(v),
            _ => None,
        }
    }

    pub fn negated(self) -> Self {
        match self {
            Self::Cp(v) => Self::Cp(-v),
            Self::Mate(v) => Self::Mate(-v),
        }
    }
}

pub(super) struct Stockfish {
    child: Child,
    input: ChildStdin,
    output: Receiver<std::io::Result<String>>,
    timeout: Duration,
    pub name: String,
}

impl Stockfish {
    pub fn spawn(path: &Path, timeout: Duration) -> Result<Self, AnalysisError> {
        let mut child = Command::new(path)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()?;
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
        let mut engine = Self {
            child,
            input,
            output: receiver,
            timeout,
            name: String::new(),
        };
        engine.send("uci")?;
        let deadline = Instant::now() + timeout;
        let mut options = Vec::new();
        loop {
            let line = engine.receive(deadline)?;
            if line == "uciok" {
                break;
            }
            if let Some(name) = line.strip_prefix("id name ") {
                engine.name = name.into();
            }
            if let Some(option) = line.strip_prefix("option name ") {
                if let Some((name, _)) = option.split_once(" type ") {
                    options.push(name.to_string());
                }
            }
        }
        if !engine.name.to_lowercase().contains("stockfish") {
            return Err(AnalysisError::Engine(
                "expected a Stockfish UCI engine".into(),
            ));
        }
        for (name, value) in [
            ("Threads", "1"),
            ("Hash", "16"),
            ("MultiPV", "1"),
            ("Ponder", "false"),
            ("Skill Level", "20"),
            ("UCI_LimitStrength", "false"),
            ("UCI_Chess960", "false"),
            ("SyzygyProbeLimit", "0"),
            ("nodestime", "0"),
        ] {
            if !options.iter().any(|option| option == name) {
                return Err(AnalysisError::Engine(format!(
                    "missing required UCI option {name}"
                )));
            }
            engine.send(&format!("setoption name {name} value {value}"))?;
        }
        engine.ready()?;
        Ok(engine)
    }

    fn send(&mut self, command: &str) -> Result<(), AnalysisError> {
        writeln!(self.input, "{command}")?;
        self.input.flush()?;
        Ok(())
    }

    fn receive(&self, deadline: Instant) -> Result<String, AnalysisError> {
        let remaining = deadline
            .checked_duration_since(Instant::now())
            .ok_or_else(|| AnalysisError::Engine("Stockfish timed out".into()))?;
        match self.output.recv_timeout(remaining) {
            Ok(Ok(line)) => Ok(line),
            Ok(Err(e)) => Err(e.into()),
            Err(e) => Err(AnalysisError::Engine(format!("waiting for Stockfish: {e}"))),
        }
    }

    fn ready(&mut self) -> Result<(), AnalysisError> {
        self.send("isready")?;
        let deadline = Instant::now() + self.timeout;
        while self.receive(deadline)? != "readyok" {}
        Ok(())
    }
}

impl SearchEngine for Stockfish {
    fn search(
        &mut self,
        history: &[String],
        position: &Position,
        nodes: u64,
    ) -> Result<Search, AnalysisError> {
        // Resets hash and search heuristics, so prior games and resume order
        // cannot change the analysis. Keep the full history for repetition.
        self.send("ucinewgame")?;
        self.ready()?;
        let command = if history.is_empty() {
            "position startpos".into()
        } else {
            format!("position startpos moves {}", history.join(" "))
        };
        self.send(&command)?;
        self.send(&format!("go nodes {nodes}"))?;
        let deadline = Instant::now() + self.timeout;
        let mut latest = None;
        loop {
            let line = self.receive(deadline)?;
            if let Some(info) = parse_info(&line) {
                latest = Some(info);
            }
            if let Some(rest) = line.strip_prefix("bestmove ") {
                let reported_best = rest
                    .split_whitespace()
                    .next()
                    .filter(|m| *m != "(none)" && *m != "0000")
                    .map(str::to_string);
                let (score, pv) = latest.ok_or_else(|| {
                    AnalysisError::Engine("bestmove without an exact score".into())
                })?;
                let legal = position.generate_legal_moves();
                // At a node cutoff Stockfish can emit a final bound and choose
                // a different bestmove from that unfinished iteration. Keep the
                // best move AND score of the last exact primary PV together.
                let best = pv.first().cloned();
                let valid = |m: &Option<String>| match m {
                    Some(m) => legal.iter().any(|mv| move_uci(*mv) == *m),
                    None => legal.is_empty(),
                };
                if !valid(&reported_best) || !valid(&best) {
                    return Err(AnalysisError::Engine(format!(
                        "invalid bestmove/PV after {} plies: reported={reported_best:?}, score={score:?}, pv={pv:?}, FEN={}",
                        history.len(), position.to_fen()
                    )));
                }
                // Never persist a corrupt or truncated UCI PV as trustworthy data.
                let mut cursor = position.clone();
                for token in &pv {
                    let mv = cursor
                        .generate_legal_moves()
                        .into_iter()
                        .find(|m| move_uci(*m) == *token)
                        .ok_or_else(|| AnalysisError::Engine(format!("illegal PV move {token}")))?;
                    cursor.make_move(mv);
                }
                return Ok(Search {
                    score,
                    best_move: best,
                    pv,
                });
            }
        }
    }
}

impl Drop for Stockfish {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

/// Bounds and secondary PVs must never be treated as exact evaluations.
fn parse_info(line: &str) -> Option<(Score, Vec<String>)> {
    let tokens: Vec<_> = line.split_whitespace().collect();
    if tokens.first() != Some(&"info")
        || tokens.get(1) == Some(&"string")
        || tokens.contains(&"lowerbound")
        || tokens.contains(&"upperbound")
    {
        return None;
    }
    if let Some(i) = tokens.iter().position(|t| *t == "multipv") {
        if tokens.get(i + 1) != Some(&"1") {
            return None;
        }
    }
    let i = tokens.iter().position(|t| *t == "score")?;
    let value = tokens.get(i + 2)?.parse::<i32>().ok()?;
    let score = match *tokens.get(i + 1)? {
        "cp" if value != i32::MIN => Score::Cp(value),
        "mate" => {
            let plies = value.checked_mul(2)?;
            Score::Mate(if value > 0 {
                plies.checked_sub(1)?
            } else {
                plies
            })
        }
        _ => return None,
    };
    if score == Score::Mate(i32::MIN) {
        return None;
    }
    let pv = tokens
        .iter()
        .position(|t| *t == "pv")
        .map(|i| tokens[i + 1..].iter().map(|t| t.to_string()).collect())
        .unwrap_or_default();
    Some((score, pv))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_exact_primary_scores_and_mate_distances() {
        assert_eq!(
            parse_info("info depth 12 multipv 1 score cp -37 nodes 1000 pv e2e4 e7e5"),
            Some((Score::Cp(-37), vec!["e2e4".into(), "e7e5".into()]))
        );
        assert_eq!(
            parse_info("info depth 4 score mate 3 pv e2e4").unwrap().0,
            Score::Mate(5)
        );
        assert_eq!(
            parse_info("info depth 4 score mate -3 pv e2e4").unwrap().0,
            Score::Mate(-6)
        );
        assert_eq!(
            parse_info("info depth 0 score mate 0"),
            Some((Score::Mate(0), vec![]))
        );
        for line in [
            "info string score cp 100",
            "info score cp 10 lowerbound pv e2e4",
            "info score cp 10 upperbound pv e2e4",
            "info multipv 2 score cp 20 pv e2e4",
            "info nodes 1",
            "info score cp",
            "info score mate 2147483647",
        ] {
            assert_eq!(parse_info(line), None, "{line}");
        }
    }

    #[cfg(unix)]
    fn fake_process(on_go: &str) -> Stockfish {
        use std::os::unix::fs::PermissionsExt;
        let path =
            std::env::temp_dir().join(format!("bee-fake-stockfish-{}", uuid::Uuid::new_v4()));
        let script = format!(
            r#"#!/bin/sh
while IFS= read -r line; do
  case "$line" in
    uci)
      echo 'id name Stockfish fake'
      for option in Threads Hash MultiPV Ponder 'Skill Level' UCI_LimitStrength UCI_Chess960 SyzygyProbeLimit nodestime; do
        echo "option name $option type string"
      done
      echo uciok ;;
    isready) echo readyok ;;
    'go nodes 100000') {on_go} ;;
  esac
done
"#
        );
        std::fs::write(&path, script).unwrap();
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o700)).unwrap();
        let process = Stockfish::spawn(&path, Duration::from_secs(2)).unwrap();
        std::fs::remove_file(path).unwrap();
        process
    }

    #[test]
    #[cfg(unix)]
    fn node_cutoff_keeps_the_last_exact_score_and_its_own_pv_together() {
        let mut process = fake_process("echo 'info depth 11 score cp -85 pv e2e4 e7e5'; echo 'info depth 12 score cp -92 lowerbound pv d2d4'; echo 'bestmove d2d4'");
        let result = process.search(&[], &Position::startpos(), 100_000).unwrap();
        assert_eq!(result.score, Score::Cp(-85));
        assert_eq!(result.best_move.as_deref(), Some("e2e4"));
        assert_eq!(result.pv, ["e2e4", "e7e5"]);
    }

    #[test]
    #[cfg(unix)]
    fn engine_exit_missing_scores_and_illegal_pv_are_errors() {
        for on_go in [
            "exit 0",
            "echo 'bestmove e2e4'",
            "echo 'info score cp 0 pv e2e4 e7e4'; echo 'bestmove e2e4'",
            "echo 'info score cp 0 pv e2e4'; echo 'bestmove e2e5'",
        ] {
            let mut process = fake_process(on_go);
            assert!(
                process.search(&[], &Position::startpos(), 100_000).is_err(),
                "{on_go}"
            );
        }
    }

    #[test]
    #[cfg(unix)]
    fn missing_bestmove_times_out_even_when_info_was_received() {
        let mut process = fake_process("echo 'info score cp 0 pv e2e4'");
        process.timeout = Duration::from_millis(50);
        assert!(process.search(&[], &Position::startpos(), 100_000).is_err());
    }
}
