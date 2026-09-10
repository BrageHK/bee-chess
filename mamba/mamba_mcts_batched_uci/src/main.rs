//! UCI-speaking wrapper around `search::search_with_stats` -- point a UCI
//! GUI or lichess-bot's `homemade`/`uci` engine config at this binary (via
//! `run_uci.sh`, which sets up the ROCm/libtorch environment first).
//!
//! Same wire protocol as `training/src/bee_training/chess_mamba/play_mcts.py`
//! (uci/isready/ucinewgame/setoption/position/go/quit, same `Simulations`/
//! `BatchSize` UCI spin options), so either one drops into the same GUI
//! config -- this one just runs the NN inference natively in Rust instead of
//! through a Python subprocess (see `search.rs`'s module docstring).

mod search;

use std::io::{self, BufRead, Write};
use std::str::FromStr;

use chess::{Board, ChessMove, Piece};
use search::{Model, SearchConfig, search_with_stats};
use tch::{Cuda, Device};

const ENGINE_NAME: &str = "Bee-Mamba-BatchedMCTS";
const ENGINE_AUTHOR: &str = "bee-chess";

const DEFAULT_SIMULATIONS: u32 = 800;
const DEFAULT_BATCH_SIZE: u32 = 64;

struct Game {
    board: Board,
    halfmove_clock: u32,
    fullmove_number: u32,
}

impl Game {
    fn new() -> Self {
        Self { board: Board::default(), halfmove_clock: 0, fullmove_number: 1 }
    }

    fn set_fen(&mut self, fen_fields: &[&str]) {
        let board_fen = fen_fields[..4].join(" ");
        self.board = Board::from_str(&board_fen).expect("valid FEN from a UCI position command");
        self.halfmove_clock = fen_fields.get(4).and_then(|s| s.parse().ok()).unwrap_or(0);
        self.fullmove_number = fen_fields.get(5).and_then(|s| s.parse().ok()).unwrap_or(1);
    }

    fn push_uci(&mut self, uci: &str) {
        let mv = ChessMove::from_str(uci).expect("legal engine-supplied UCI move");
        let is_pawn_move = self.board.piece_on(mv.get_source()) == Some(Piece::Pawn);
        let is_capture = self.board.piece_on(mv.get_dest()).is_some();
        self.halfmove_clock = if is_pawn_move || is_capture { 0 } else { self.halfmove_clock + 1 };
        if self.board.side_to_move() == chess::Color::Black {
            self.fullmove_number += 1;
        }
        self.board = self.board.make_move_new(mv);
    }

    fn fen(&self) -> String {
        let board_fen = self.board.to_string();
        let mut fields: Vec<&str> = board_fen.split_whitespace().collect();
        fields.truncate(4);
        format!("{} {} {}", fields.join(" "), self.halfmove_clock, self.fullmove_number)
    }
}

fn apply_position_command(game: &mut Game, tokens: &[&str]) {
    let (fen_tokens, rest) = if tokens[0] == "startpos" {
        (
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1".split_whitespace().collect::<Vec<_>>(),
            &tokens[1..],
        )
    } else {
        assert_eq!(tokens[0], "fen");
        (tokens[1..7].to_vec(), &tokens[7..])
    };
    game.set_fen(&fen_tokens);
    if rest.first() == Some(&"moves") {
        for uci in &rest[1..] {
            game.push_uci(uci);
        }
    }
}

fn handle_setoption(rest: &[&str], simulations: &mut u32, batch_size: &mut u32) {
    if rest.len() < 4 || rest[0] != "name" || rest[2] != "value" {
        return;
    }
    let (name, value) = (rest[1], rest[3]);
    match name {
        "Simulations" => {
            if let Ok(v) = value.parse() {
                *simulations = v;
            }
        }
        "BatchSize" => {
            if let Ok(v) = value.parse() {
                *batch_size = v;
            }
        }
        _ => {}
    }
}

fn default_device() -> Device {
    if Cuda::is_available() { Device::Cuda(0) } else { Device::Cpu }
}

fn main() {
    tch::set_num_threads(1);

    let device = default_device();
    eprintln!("[mamba_mcts_batched_uci] loading model on {device:?}");
    let model = Model::load_embedded(device);

    let mut simulations: u32 =
        std::env::var("SIMULATIONS").ok().and_then(|s| s.parse().ok()).unwrap_or(DEFAULT_SIMULATIONS);
    let mut batch_size: u32 =
        std::env::var("BATCH_SIZE").ok().and_then(|s| s.parse().ok()).unwrap_or(DEFAULT_BATCH_SIZE);

    let mut game = Game::new();
    let stdin = io::stdin();
    let mut stdout = io::stdout();

    for line in stdin.lock().lines() {
        let line = line.expect("stdin readable");
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let tokens: Vec<&str> = line.split_whitespace().collect();

        match tokens[0] {
            "uci" => {
                writeln!(stdout, "id name {ENGINE_NAME}").unwrap();
                writeln!(stdout, "id author {ENGINE_AUTHOR}").unwrap();
                writeln!(stdout, "option name Simulations type spin default {DEFAULT_SIMULATIONS} min 1 max 100000")
                    .unwrap();
                writeln!(stdout, "option name BatchSize type spin default {DEFAULT_BATCH_SIZE} min 1 max 512").unwrap();
                writeln!(stdout, "uciok").unwrap();
            }
            "isready" => writeln!(stdout, "readyok").unwrap(),
            "ucinewgame" => game = Game::new(),
            "setoption" => handle_setoption(&tokens[1..], &mut simulations, &mut batch_size),
            "position" => apply_position_command(&mut game, &tokens[1..]),
            "go" => {
                let cfg = SearchConfig {
                    simulations: simulations as usize,
                    batch_size: batch_size as usize,
                    ..SearchConfig::default()
                };
                let t0 = std::time::Instant::now();
                let outcome = search_with_stats(&model, &game.fen(), &cfg);
                let nodes_per_sec = simulations as f64 / t0.elapsed().as_secs_f64();
                let cache_hit_rate =
                    100.0 * outcome.cache_hits as f64 / (outcome.cache_hits + outcome.cache_misses).max(1) as f64;
                eprintln!(
                    "[mamba_mcts_batched_uci] {nodes_per_sec:.0} nodes/s, {} collisions, {cache_hit_rate:.1}% cache hit",
                    outcome.collisions
                );
                match outcome.best_move {
                    Some(mv) => writeln!(stdout, "bestmove {mv}").unwrap(),
                    None => writeln!(stdout, "bestmove (none)").unwrap(),
                }
            }
            "quit" => return,
            _ => {}
        }
        stdout.flush().unwrap();
    }
}
