//! PyO3 extension: lc0-style PUCT MCTS (tree/PUCT/FPU/virtual-loss/legal-move
//! generation, plus the board-plane encoding) entirely in Rust, calling back
//! into Python only for the batched NN forward pass itself.
//!
//! Two speedups on top of the first working version:
//!   1. Batching: each "wave" collects `batch_size` pending leaves (staking
//!      virtual loss on each as they're selected, so a wave doesn't just
//!      re-pick the same top leaf `batch_size` times) and evaluates them in
//!      ONE Python call -- one real batched NN forward pass instead of
//!      `batch_size` separate ones. This is what lets a GPU (PyTorch's MPS)
//!      win at all: every earlier single-leaf GPU path (burn/wgpu,
//!      burn/metal, onnxruntime's CoreML EP) lost badly to CPU.
//!   2. No FEN round-trip: the first batched version passed FEN *strings*
//!      to Python, which re-parsed each one into a `chess.Board` and
//!      re-encoded it into planes -- pure waste, since Rust already has the
//!      `Board` in hand from its own tree walk. This version encodes planes
//!      directly in Rust (`encode_board`, ported from bee-chess's
//!      `encode.py` / MuZero-rs's `chess_mamba_bot.rs`) into one flat
//!      buffer per wave, handed to Python as a numpy array; results come
//!      back as numpy arrays too, read via a raw slice instead of boxing
//!      every float as a Python object through `.tolist()`.
//!
//! Still no OS threading -- with batching doing the real work, a
//! single-threaded wave loop is simpler and already keeps the NN saturated;
//! the Python callback holds the GIL only for the Rust<->Python call
//! itself, releasing it during the actual tensor computation (both
//! onnxruntime and PyTorch do this).
//!
//! 3. NN evaluation cache (per lc0's own docs: "turning off lc0's cache
//!    makes NPS plummet" -- this is one of its biggest real levers, and
//!    was missing here entirely until now). Keyed by `Board::get_hash()`
//!    (a Zobrist hash -- doesn't fold in the halfmove clock, which
//!    `Board` doesn't track anyway; two positions differing only in that
//!    one auxiliary input scalar are close enough to treat as the same
//!    cache entry, same approximation real engines make for the 50-move
//!    counter). A hit skips the NN call for that leaf entirely: no
//!    encoding, no batch slot, just an immediate expand+backup. How much
//!    this helps depends entirely on how many transpositions a given
//!    search tree actually contains -- measured, not assumed, via the
//!    `cache_hits`/`cache_misses` fields on `SearchStats`.

use std::collections::HashMap;
use std::str::FromStr;

use chess::{Board, BoardStatus, ChessMove, Color, MoveGen, Piece, ALL_SQUARES};
use numpy::{PyArray1, PyArrayMethods, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

const CP_CLIP: f32 = 1000.0;
const N_VALUE_BINS: usize = 128;
const BIN_WIDTH: f32 = 2.0 * CP_CLIP / N_VALUE_BINS as f32;

const CPUCT_INIT: f32 = 1.745;
const CPUCT_BASE: f32 = 38739.0;
const CPUCT_FACTOR: f32 = 3.894;
const FPU_REDUCTION: f32 = 0.33;
const VIRTUAL_LOSS: f32 = 1.0;

// Must match bee-chess's encode.py / MuZero-rs's chess_mamba_bot.rs exactly.
const N_PIECE_TYPES: usize = 12;
const N_AUX: usize = 8;
const IN_DIM: usize = N_PIECE_TYPES + N_AUX;

fn bin_center(i: usize) -> f32 {
    -CP_CLIP + (i as f32 + 0.5) * BIN_WIDTH
}

fn softmax_in_place(xs: &mut [f32]) {
    let max = xs.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for x in xs.iter_mut() {
        *x = (*x - max).exp();
        sum += *x;
    }
    for x in xs.iter_mut() {
        *x /= sum;
    }
}

/// `board` (+ the halfmove clock `chess::Board` doesn't track itself) ->
/// flat (64 * IN_DIM) row-major (square, channel) plane data.
fn encode_board(board: &Board, halfmove_clock: u32, out: &mut [f32]) {
    debug_assert_eq!(out.len(), 64 * IN_DIM);
    out.fill(0.0);
    for square in ALL_SQUARES {
        if let Some(piece) = board.piece_on(square) {
            let color_offset = if board.color_on(square) == Some(Color::White) {
                0
            } else {
                6
            };
            out[square.to_index() * IN_DIM + piece.to_index() + color_offset] = 1.0;
        }
    }
    let white_castle = board.castle_rights(Color::White);
    let black_castle = board.castle_rights(Color::Black);
    let aux = [
        f32::from(white_castle.has_kingside()),
        f32::from(white_castle.has_queenside()),
        f32::from(black_castle.has_kingside()),
        f32::from(black_castle.has_queenside()),
        f32::from(board.en_passant().is_some()),
        halfmove_clock as f32 / 100.0,
        f32::from(board.side_to_move() == Color::White),
        0.0, // reserved, unused -- see encode.py
    ];
    for square in ALL_SQUARES {
        let base = square.to_index() * IN_DIM + N_PIECE_TYPES;
        out[base..base + N_AUX].copy_from_slice(&aux);
    }
}

struct Node {
    prior: f32,
    n: u32,
    w: f32,
    vln: u32,
    vlw: f32,
    children: Vec<(ChessMove, usize)>,
}

impl Node {
    fn q(&self) -> f32 {
        let total = self.n + self.vln;
        if total == 0 {
            0.0
        } else {
            (self.w + self.vlw) / total as f32
        }
    }
    fn expanded(&self) -> bool {
        !self.children.is_empty()
    }
}

fn cpuct(parent_n: u32) -> f32 {
    CPUCT_INIT + CPUCT_FACTOR * ((parent_n as f32 + CPUCT_BASE) / CPUCT_BASE).ln()
}

fn visited_policy_mass(arena: &[Node], node: &Node) -> f32 {
    node.children
        .iter()
        .filter(|&&(_, idx)| arena[idx].n > 0)
        .map(|&(_, idx)| arena[idx].prior)
        .sum()
}

fn select_child(arena: &[Node], node_idx: usize) -> usize {
    let node = &arena[node_idx];
    let parent_total = node.n + node.vln;
    let c = cpuct(parent_total);
    let fpu = -node.q() - FPU_REDUCTION * visited_policy_mass(arena, node).sqrt();
    let sqrt_total = (parent_total.max(1) as f32).sqrt();

    let mut best_score = f32::NEG_INFINITY;
    let mut best_idx = node.children[0].1;
    for &(_, child_idx) in &node.children {
        let child = &arena[child_idx];
        let child_total = child.n + child.vln;
        let u = c * child.prior * sqrt_total / (1.0 + child_total as f32);
        let q_term = if child_total > 0 { -child.q() } else { fpu };
        let score = q_term + u;
        if score > best_score {
            best_score = score;
            best_idx = child_idx;
        }
    }
    best_idx
}

fn advance_halfmove_clock(board: &Board, mv: ChessMove, prev: u32) -> u32 {
    let is_pawn_move = board.piece_on(mv.get_source()) == Some(Piece::Pawn);
    let is_capture = board.piece_on(mv.get_dest()).is_some();
    if is_pawn_move || is_capture {
        0
    } else {
        prev + 1
    }
}

fn select_path(
    arena: &mut Vec<Node>,
    root_board: &Board,
    root_clock: u32,
) -> (Vec<usize>, Board, u32) {
    let mut idx = 0usize;
    let mut board = *root_board;
    let mut clock = root_clock;
    let mut path = vec![0usize];

    arena[0].vln += 1;
    arena[0].vlw += VIRTUAL_LOSS;

    while arena[idx].expanded() {
        let next = select_child(arena, idx);
        let mv = arena[idx]
            .children
            .iter()
            .find(|&&(_, c)| c == next)
            .unwrap()
            .0;
        clock = advance_halfmove_clock(&board, mv, clock);
        board = board.make_move_new(mv);
        idx = next;
        arena[idx].vln += 1;
        arena[idx].vlw += VIRTUAL_LOSS;
        path.push(idx);
    }
    (path, board, clock)
}

fn unstake_and_backup(arena: &mut [Node], path: &[usize], leaf_value: f32) {
    let mut value = leaf_value;
    for &idx in path.iter().rev() {
        let node = &mut arena[idx];
        node.vln -= 1;
        node.vlw -= VIRTUAL_LOSS;
        node.n += 1;
        node.w += value;
        value = -value;
    }
}

fn expand(arena: &mut Vec<Node>, leaf_idx: usize, priors: Vec<(ChessMove, f32)>) {
    if arena[leaf_idx].expanded() || priors.is_empty() {
        return;
    }
    let mut children = Vec::with_capacity(priors.len());
    for (mv, prior) in priors {
        let child_idx = arena.len();
        arena.push(Node {
            prior,
            n: 0,
            w: 0.0,
            vln: 0,
            vlw: 0.0,
            children: Vec::new(),
        });
        children.push((mv, child_idx));
    }
    arena[leaf_idx].children = children;
}

fn legal_moves_with_indices(board: &Board) -> Vec<(ChessMove, usize)> {
    MoveGen::new_legal(board)
        .filter(|mv| !mv.get_promotion().is_some_and(|p| p != Piece::Queen))
        .map(|mv| {
            (
                mv,
                mv.get_source().to_index() * 64 + mv.get_dest().to_index(),
            )
        })
        .collect()
}

/// Turns one leaf's raw (policy_logits: len 4096, value_logits: len
/// N_VALUE_BINS) network output into (legal-move priors, value in [-1,1]
/// from the side-to-move's perspective).
fn postprocess(
    board: &Board,
    policy_logits: &[f32],
    value_logits: &[f32],
) -> (Vec<(ChessMove, f32)>, f32) {
    let legal = legal_moves_with_indices(board);
    let mut scores: Vec<f32> = legal.iter().map(|&(_, idx)| policy_logits[idx]).collect();
    softmax_in_place(&mut scores);
    let priors = legal
        .into_iter()
        .zip(scores)
        .map(|((mv, _), p)| (mv, p))
        .collect();

    let mut value_probs = value_logits.to_vec();
    softmax_in_place(&mut value_probs);
    let expected_cp: f32 = value_probs
        .iter()
        .enumerate()
        .map(|(i, p)| p * bin_center(i))
        .sum();
    let value = (expected_cp / 400.0).tanh();
    (priors, value)
}

/// Encodes `boards` into one (n, 64, IN_DIM) numpy array and calls
/// `evaluate_batch(planes) -> (policy: ndarray[n, 4096], value: ndarray[n,
/// N_VALUE_BINS])`, returning the two result arrays' raw contiguous slices'
/// owned copies (so they outlive the `Bound` borrow).
fn call_evaluate_batch(
    py: Python<'_>,
    evaluate_batch: &PyObject,
    boards: &[Board],
    clocks: &[u32],
) -> PyResult<(Vec<f32>, Vec<f32>)> {
    let n = boards.len();
    let mut planes = vec![0f32; n * 64 * IN_DIM];
    for (i, (board, &clock)) in boards.iter().zip(clocks).enumerate() {
        encode_board(
            board,
            clock,
            &mut planes[i * 64 * IN_DIM..(i + 1) * 64 * IN_DIM],
        );
    }
    let planes_arr = PyArray1::from_vec(py, planes).reshape([n, 64, IN_DIM])?;

    let result = evaluate_batch.call1(py, (planes_arr,))?;
    let (policy_obj, value_obj): (PyObject, PyObject) = result.extract(py)?;
    let policy_arr: PyReadonlyArray2<f32> = policy_obj.extract(py)?;
    let value_arr: PyReadonlyArray2<f32> = value_obj.extract(py)?;
    Ok((
        policy_arr.as_slice()?.to_vec(),
        value_arr.as_slice()?.to_vec(),
    ))
}

#[pyclass]
#[derive(Clone, Copy, Default)]
pub struct SearchStats {
    #[pyo3(get)]
    pub cache_hits: u64,
    #[pyo3(get)]
    pub cache_misses: u64,
}

fn search_impl(
    py: Python<'_>,
    fen: &str,
    evaluate_batch: &PyObject,
    simulations: usize,
    batch_size: usize,
) -> PyResult<(Option<String>, SearchStats)> {
    let fields: Vec<&str> = fen.split_whitespace().collect();
    if fields.len() < 4 {
        return Err(PyValueError::new_err("malformed FEN"));
    }
    let root_board = Board::from_str(&fields[..4].join(" "))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let root_clock: u32 = fields.get(4).and_then(|s| s.parse().ok()).unwrap_or(0);

    let mut arena = vec![Node {
        prior: 0.0,
        n: 0,
        w: 0.0,
        vln: 0,
        vlw: 0.0,
        children: Vec::new(),
    }];
    let mut cache: HashMap<u64, (Vec<(ChessMove, f32)>, f32)> = HashMap::new();
    let mut stats = SearchStats::default();

    // Root eval: a batch of 1, same call path as every other batch.
    {
        let (policy_flat, value_flat) =
            call_evaluate_batch(py, evaluate_batch, &[root_board], &[root_clock])?;
        let policy_dim = policy_flat.len();
        let value_dim = value_flat.len();
        let (priors, _v) = postprocess(
            &root_board,
            &policy_flat[..policy_dim],
            &value_flat[..value_dim],
        );
        if priors.is_empty() {
            return Ok((None, stats)); // no legal moves
        }
        expand(&mut arena, 0, priors.clone());
        cache.insert(root_board.get_hash(), (priors, _v));
    }

    let mut done = 0usize;
    while done < simulations {
        let wave = batch_size.min(simulations - done);

        let mut paths: Vec<Vec<usize>> = Vec::with_capacity(wave);
        let mut boards: Vec<Board> = Vec::with_capacity(wave);
        let mut clocks: Vec<u32> = Vec::with_capacity(wave);
        for _ in 0..wave {
            let (path, board, clock) = select_path(&mut arena, &root_board, root_clock);
            paths.push(path);
            boards.push(board);
            clocks.push(clock);
        }

        let mut pending_indices = Vec::with_capacity(wave);
        let mut pending_boards = Vec::with_capacity(wave);
        let mut pending_clocks = Vec::with_capacity(wave);
        let mut leaf_values = vec![0.0f32; wave];
        for i in 0..wave {
            if boards[i].status() != BoardStatus::Ongoing {
                leaf_values[i] = if boards[i].status() == BoardStatus::Checkmate {
                    -1.0
                } else {
                    0.0
                };
                continue;
            }
            match cache.get(&boards[i].get_hash()) {
                Some((priors, value)) => {
                    stats.cache_hits += 1;
                    let leaf_idx = *paths[i].last().unwrap();
                    expand(&mut arena, leaf_idx, priors.clone());
                    leaf_values[i] = *value;
                }
                None => {
                    stats.cache_misses += 1;
                    pending_indices.push(i);
                    pending_boards.push(boards[i]);
                    pending_clocks.push(clocks[i]);
                }
            }
        }

        if !pending_boards.is_empty() {
            let (policy_flat, value_flat) =
                call_evaluate_batch(py, evaluate_batch, &pending_boards, &pending_clocks)?;
            let policy_stride = policy_flat.len() / pending_boards.len();
            let value_stride = value_flat.len() / pending_boards.len();
            for (k, &i) in pending_indices.iter().enumerate() {
                let policy_row = &policy_flat[k * policy_stride..(k + 1) * policy_stride];
                let value_row = &value_flat[k * value_stride..(k + 1) * value_stride];
                let (priors, value) = postprocess(&boards[i], policy_row, value_row);
                let leaf_idx = *paths[i].last().unwrap();
                expand(&mut arena, leaf_idx, priors.clone());
                cache.insert(boards[i].get_hash(), (priors, value));
                leaf_values[i] = value;
            }
        }

        for i in 0..wave {
            unstake_and_backup(&mut arena, &paths[i], leaf_values[i]);
        }
        done += wave;
    }

    let best = arena[0]
        .children
        .iter()
        .max_by_key(|&&(_, idx)| arena[idx].n)
        .map(|&(mv, _)| mv.to_string());
    Ok((best, stats))
}

#[pyfunction]
#[pyo3(signature = (fen, evaluate_batch, simulations=800, batch_size=16))]
fn search(
    py: Python<'_>,
    fen: &str,
    evaluate_batch: PyObject,
    simulations: usize,
    batch_size: usize,
) -> PyResult<Option<String>> {
    search_impl(py, fen, &evaluate_batch, simulations, batch_size).map(|(mv, _stats)| mv)
}

/// Like `search`, plus an NN-cache hit/miss breakdown (see the module
/// docstring's point 3) so callers can see how much the transposition
/// cache is actually buying on a given position/simulation budget.
#[pyfunction]
#[pyo3(signature = (fen, evaluate_batch, simulations=800, batch_size=16))]
fn search_with_stats(
    py: Python<'_>,
    fen: &str,
    evaluate_batch: PyObject,
    simulations: usize,
    batch_size: usize,
) -> PyResult<(Option<String>, SearchStats)> {
    search_impl(py, fen, &evaluate_batch, simulations, batch_size)
}

#[pymodule]
fn mamba_mcts_native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<SearchStats>()?;
    m.add_function(wrap_pyfunction!(search, m)?)?;
    m.add_function(wrap_pyfunction!(search_with_stats, m)?)?;
    Ok(())
}
