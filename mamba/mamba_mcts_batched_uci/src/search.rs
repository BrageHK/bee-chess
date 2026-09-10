//! Wave-batched PUCT MCTS through a *dynamic-batch* TorchScript module,
//! ported from MuZero-rs's `mz-web` crate (`chess_mamba_mcts_batched.rs`) so
//! bee-chess has a native UCI engine at the same nodes/sec, with no
//! dependency on that other repo at build or run time.
//!
//! Why this exists next to `mamba_mcts_native` (this repo's PyO3 crate,
//! which gets its NN inference from Python/PyTorch) instead of just using
//! that: this is a from-scratch equivalent with the NN inference *inside*
//! Rust too, via raw `tch::CModule` -- no Python process, no PyO3 GIL
//! handoff, a single native binary a UCI GUI or lichess-bot can exec
//! directly. Same wave-batching design either way (see
//! `mamba_mcts_native/src/lib.rs`'s docstring for the shared rationale:
//! batch leaves per wave, one NN forward call per wave, single-threaded,
//! plus a Zobrist-keyed NN-eval cache).
//!
//! `models/chess_mamba_dynamic.pt` is `checkpoints/ThisTimeForSure/best.pt`
//! traced to TorchScript with a dynamic batch axis via plain
//! `torch.jit.trace` (forcing `scan_backend="sequential"`, same as
//! `export_onnx.py`'s own choice, for the same tracing-safety reason).
//! Verified to match eager output exactly across batch sizes never seen
//! while tracing. Regenerate after retraining with:
//! ```python
//! # from bee-chess/training, with its own .venv active
//! import torch
//! from bee_training.chess_mamba.train import TrainConfig, build_model
//! ckpt = torch.load("checkpoints/ThisTimeForSure/best.pt", map_location="cpu", weights_only=False)
//! config = TrainConfig(**ckpt["config"]); config.scan_backend = "sequential"
//! model = build_model(config); model.load_state_dict(ckpt["model_state"]); model.eval()
//! dummy = torch.randn(5, 64, 20)  # batch size 5 is arbitrary, just not 1
//! traced = torch.jit.trace(model, (dummy,))
//! traced.save("../mamba_mcts_batched_uci/models/chess_mamba_dynamic.pt")
//! ```

use core::str::FromStr;
use std::collections::HashMap;
use std::io::Cursor;

use chess::{ALL_SQUARES, Board, BoardStatus, ChessMove, Color, MoveGen, Piece};
use tch::{CModule, Device, IValue, Kind, Tensor};

// Must match bee-chess's encode.py / mamba_mcts_native's encode_board
// exactly: 12 one-hot piece planes (6 piece types x 2 colors) + 8 auxiliary
// scalars broadcast to every square.
const N_PIECE_TYPES: usize = 12;
const N_AUX: usize = 8;
const IN_DIM: usize = N_PIECE_TYPES + N_AUX;

const CP_CLIP: f32 = 1000.0;
const N_VALUE_BINS: usize = 128;
const BIN_WIDTH: f32 = 2.0 * CP_CLIP / N_VALUE_BINS as f32;

// Same lc0-inspired defaults as mamba_mcts_native/src/lib.rs.
const CPUCT_INIT: f32 = 1.745;
const CPUCT_BASE: f32 = 38739.0;
const CPUCT_FACTOR: f32 = 3.894;
const FPU_REDUCTION: f32 = 0.33;

fn encode_board(board: &Board, halfmove_clock: u32, out: &mut [f32]) {
    debug_assert_eq!(out.len(), 64 * IN_DIM);
    out.fill(0.0);
    for square in ALL_SQUARES {
        if let Some(piece) = board.piece_on(square) {
            let color_offset = if board.color_on(square) == Some(Color::White) { 0 } else { 6 };
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
        0.0,
    ];
    for square in ALL_SQUARES {
        let base = square.to_index() * IN_DIM + N_PIECE_TYPES;
        out[base..base + N_AUX].copy_from_slice(&aux);
    }
}

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

fn advance_halfmove_clock(board: &Board, mv: ChessMove, prev: u32) -> u32 {
    let is_pawn_move = board.piece_on(mv.get_source()) == Some(Piece::Pawn);
    let is_capture = board.piece_on(mv.get_dest()).is_some();
    if is_pawn_move || is_capture { 0 } else { prev + 1 }
}

#[derive(Debug, Clone, Copy)]
pub struct SearchConfig {
    pub simulations: usize,
    pub batch_size: usize,
    pub virtual_loss: f32,
    pub cpuct_init: f32,
    pub cpuct_base: f32,
    pub cpuct_factor: f32,
    pub fpu_reduction: f32,
}

impl Default for SearchConfig {
    fn default() -> Self {
        Self {
            simulations: 800,
            batch_size: 64,
            virtual_loss: 1.0,
            cpuct_init: CPUCT_INIT,
            cpuct_base: CPUCT_BASE,
            cpuct_factor: CPUCT_FACTOR,
            fpu_reduction: FPU_REDUCTION,
        }
    }
}

// The distro's default linker enables `--as-needed`, which -- unlike the
// build this was ported from (MuZero-rs's `mz-web`, linked with `lld`) --
// drops `libtorch_hip.so`/`libtorch_cuda.so` (and their own `libc10_*`
// dependency) from this binary's NEEDED entries entirely: nothing in tch's
// own generated bindings creates an actual undefined-symbol reference into
// either library (ATen's device dispatch is a runtime registry populated
// by those libraries' own static initializers, invisible to link-time
// as-needed analysis), so without this, `Cuda::is_available()` silently
// reports `false` even on a correct GPU libtorch install with a real GPU
// present (verified on ROCm: same torch install, `python -c 'import torch;
// torch.cuda.is_available()'` correctly returns `True` in the same shell).
// Referencing (never calling) one real exported symbol forces the linker
// to keep the library as NEEDED, which pulls in its own `libc10_*`
// dependency transitively and lets the GPU backend's static initializers
// run at process startup, same as they do inside the Python interpreter.
// `build.rs` sets `has_torch_hip`/`has_torch_cuda` by checking which
// library actually exists in the resolved libtorch dir, so a CPU-only or
// Mac (MPS lives inside `libtorch_cpu.so` already, no separate library)
// build compiles neither module. PyTorch's ATen keeps the `at::cuda`
// C++ namespace for both real CUDA and hipified ROCm builds (that's what
// lets one source tree target both), so the same mangled symbol name is
// expected to exist in `libtorch_cuda.so` too -- verified on ROCm; not yet
// verified on a real CUDA machine.
#[cfg(has_torch_hip)]
#[allow(non_snake_case, dead_code)]
mod force_link_gpu_backend {
    #[link(name = "torch_hip")]
    extern "C" {
        #[link_name = "_ZN2at4cuda3jit21initializeCudaContextEv"]
        fn force_link_symbol();
    }
    #[used]
    static FORCE_LINK: unsafe extern "C" fn() = force_link_symbol;
}

#[cfg(all(has_torch_cuda, not(has_torch_hip)))]
#[allow(non_snake_case, dead_code)]
mod force_link_gpu_backend {
    #[link(name = "torch_cuda")]
    extern "C" {
        #[link_name = "_ZN2at4cuda3jit21initializeCudaContextEv"]
        fn force_link_symbol();
    }
    #[used]
    static FORCE_LINK: unsafe extern "C" fn() = force_link_symbol;
}

pub struct Model {
    module: CModule,
    device: Device,
}

impl Model {
    pub fn load_embedded(device: Device) -> Self {
        let bytes: &[u8] = include_bytes!("../models/chess_mamba_dynamic.pt");
        let module = CModule::load_data_on_device(&mut Cursor::new(bytes), device)
            .expect("embedded chess_mamba_dynamic.pt failed to load");
        Self { module, device }
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
        if total == 0 { 0.0 } else { (self.w + self.vlw) / total as f32 }
    }

    fn expanded(&self) -> bool {
        !self.children.is_empty()
    }
}

fn cpuct(parent_n: u32, cfg: &SearchConfig) -> f32 {
    cfg.cpuct_init + cfg.cpuct_factor * ((parent_n as f32 + cfg.cpuct_base) / cfg.cpuct_base).ln()
}

fn visited_policy_mass(arena: &[Node], node: &Node) -> f32 {
    node.children.iter().filter(|&&(_, idx)| arena[idx].n > 0).map(|&(_, idx)| arena[idx].prior).sum()
}

fn select_child(arena: &[Node], node_idx: usize, cfg: &SearchConfig) -> usize {
    let node = &arena[node_idx];
    let parent_total = node.n + node.vln;
    let c = cpuct(parent_total, cfg);
    let fpu = -node.q() - cfg.fpu_reduction * visited_policy_mass(arena, node).sqrt();
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

struct Tree {
    arena: Vec<Node>,
    collisions: u64,
}

fn select_path(tree: &mut Tree, root_board: &Board, root_halfmove_clock: u32, cfg: &SearchConfig) -> (Vec<usize>, Board, u32) {
    let mut idx = 0usize;
    let mut board = *root_board;
    let mut halfmove_clock = root_halfmove_clock;
    let mut path = vec![0usize];

    {
        let node = &mut tree.arena[0];
        node.vln += 1;
        node.vlw += cfg.virtual_loss;
    }

    while tree.arena[idx].expanded() {
        let next = select_child(&tree.arena, idx, cfg);
        let mv = tree.arena[idx].children.iter().find(|&&(_, c)| c == next).unwrap().0;
        halfmove_clock = advance_halfmove_clock(&board, mv, halfmove_clock);
        board = board.make_move_new(mv);
        idx = next;
        let node = &mut tree.arena[idx];
        node.vln += 1;
        node.vlw += cfg.virtual_loss;
        path.push(idx);
    }

    if tree.arena[idx].vln > 1 {
        tree.collisions += 1;
    }
    (path, board, halfmove_clock)
}

fn unstake_and_backup(tree: &mut Tree, path: &[usize], leaf_value: f32, virtual_loss: f32) {
    let mut value = leaf_value;
    for &idx in path.iter().rev() {
        let node = &mut tree.arena[idx];
        node.vln -= 1;
        node.vlw -= virtual_loss;
        node.n += 1;
        node.w += value;
        value = -value;
    }
}

fn expand(tree: &mut Tree, leaf_idx: usize, priors: Vec<(ChessMove, f32)>) {
    if tree.arena[leaf_idx].expanded() || priors.is_empty() {
        return;
    }
    let mut children = Vec::with_capacity(priors.len());
    for (mv, prior) in priors {
        let child_idx = tree.arena.len();
        tree.arena.push(Node { prior, n: 0, w: 0.0, vln: 0, vlw: 0.0, children: Vec::new() });
        children.push((mv, child_idx));
    }
    tree.arena[leaf_idx].children = children;
}

fn best_move_by_visits(tree: &Tree) -> Option<ChessMove> {
    tree.arena[0].children.iter().max_by_key(|&&(_, idx)| tree.arena[idx].n).map(|&(mv, _)| mv)
}

fn evaluate_batch(model: &Model, boards: &[(Board, u32)]) -> Vec<(Vec<(ChessMove, f32)>, f32)> {
    let n = boards.len();
    let mut planes_flat = vec![0f32; n * 64 * IN_DIM];
    let mut legal_moves_per: Vec<Vec<ChessMove>> = Vec::with_capacity(n);
    for (i, (board, halfmove_clock)) in boards.iter().enumerate() {
        encode_board(board, *halfmove_clock, &mut planes_flat[i * 64 * IN_DIM..(i + 1) * 64 * IN_DIM]);
        let legal_moves: Vec<ChessMove> = MoveGen::new_legal(board)
            .filter(|mv| !mv.get_promotion().is_some_and(|p| p != Piece::Queen))
            .collect();
        legal_moves_per.push(legal_moves);
    }

    let input = Tensor::from_slice(&planes_flat)
        .to_device(model.device)
        .reshape([n as i64, 64, IN_DIM as i64]);
    let output = model.module.forward_is(&[IValue::Tensor(input)]).expect("chess_mamba_dynamic forward failed");
    let IValue::Tuple(mut outputs) = output else { panic!("expected a (policy, value) tuple output") };
    assert_eq!(outputs.len(), 2, "expected exactly 2 outputs");
    let IValue::Tensor(value_logits) = outputs.pop().unwrap() else { panic!("expected a value tensor") };
    let IValue::Tensor(policy_logits) = outputs.pop().unwrap() else { panic!("expected a policy tensor") };

    let policy_flat = policy_logits.reshape([n as i64, 64 * 64]).to_kind(Kind::Float).to_device(Device::Cpu);
    let value_flat = value_logits.to_kind(Kind::Float).to_device(Device::Cpu);
    let policy_rows: Vec<Vec<f32>> = (&policy_flat).try_into().expect("policy tensor -> Vec<Vec<f32>>");
    let value_rows: Vec<Vec<f32>> = (&value_flat).try_into().expect("value tensor -> Vec<Vec<f32>>");

    boards
        .iter()
        .zip(legal_moves_per)
        .zip(policy_rows)
        .zip(value_rows)
        .map(|(((_, legal_moves), policy), mut value_probs)| {
            if legal_moves.is_empty() {
                return (Vec::new(), 0.0);
            }
            let mut scores: Vec<f32> =
                legal_moves.iter().map(|mv| policy[mv.get_source().to_index() * 64 + mv.get_dest().to_index()]).collect();
            softmax_in_place(&mut scores);
            let priors: Vec<(ChessMove, f32)> = legal_moves.into_iter().zip(scores).collect();

            softmax_in_place(&mut value_probs);
            let expected_cp: f32 = value_probs.iter().enumerate().map(|(i, p)| p * bin_center(i)).sum();
            let value = (expected_cp / 400.0).tanh();
            (priors, value)
        })
        .collect()
}

pub struct SearchOutcome {
    pub best_move: Option<String>,
    pub collisions: u64,
    pub cache_hits: u64,
    pub cache_misses: u64,
}

pub fn search_with_stats(model: &Model, fen: &str, cfg: &SearchConfig) -> SearchOutcome {
    let Some(root_board) = Board::from_str(fen).ok() else {
        return SearchOutcome { best_move: None, collisions: 0, cache_hits: 0, cache_misses: 0 };
    };
    let root_halfmove_clock: u32 = fen.split_whitespace().nth(4).and_then(|s| s.parse().ok()).unwrap_or(0);

    let (root_priors, _root_value) = {
        let mut result = evaluate_batch(model, &[(root_board, root_halfmove_clock)]);
        result.pop().unwrap()
    };
    if root_priors.is_empty() {
        return SearchOutcome { best_move: None, collisions: 0, cache_hits: 0, cache_misses: 0 };
    }

    let mut tree =
        Tree { arena: vec![Node { prior: 0.0, n: 0, w: 0.0, vln: 0, vlw: 0.0, children: Vec::new() }], collisions: 0 };
    expand(&mut tree, 0, root_priors);

    let mut eval_cache: HashMap<u64, (Vec<(ChessMove, f32)>, f32)> = HashMap::new();
    let mut cache_hits = 0u64;
    let mut cache_misses = 0u64;

    let mut remaining = cfg.simulations;
    while remaining > 0 {
        let wave_size = remaining.min(cfg.batch_size);
        remaining -= wave_size;

        let mut paths = Vec::with_capacity(wave_size);
        let mut nn_leaves: Vec<(Board, u32)> = Vec::new();
        let mut nn_path_indices = Vec::new();
        for _ in 0..wave_size {
            let (path, leaf_board, leaf_clock) = select_path(&mut tree, &root_board, root_halfmove_clock, cfg);
            let leaf_idx = *path.last().unwrap();
            if leaf_board.status() != BoardStatus::Ongoing {
                let value = if leaf_board.status() == BoardStatus::Checkmate { -1.0 } else { 0.0 };
                unstake_and_backup(&mut tree, &path, value, cfg.virtual_loss);
            } else if let Some((priors, value)) = eval_cache.get(&leaf_board.get_hash()) {
                cache_hits += 1;
                expand(&mut tree, leaf_idx, priors.clone());
                unstake_and_backup(&mut tree, &path, *value, cfg.virtual_loss);
            } else {
                cache_misses += 1;
                nn_path_indices.push((paths.len(), leaf_idx));
                paths.push((path, leaf_board.get_hash()));
                nn_leaves.push((leaf_board, leaf_clock));
            }
        }

        if !nn_leaves.is_empty() {
            let results = evaluate_batch(model, &nn_leaves);
            for ((path_idx, leaf_idx), (priors, value)) in nn_path_indices.into_iter().zip(results) {
                let (path, hash) = &paths[path_idx];
                eval_cache.insert(*hash, (priors.clone(), value));
                expand(&mut tree, leaf_idx, priors);
                unstake_and_backup(&mut tree, path, value, cfg.virtual_loss);
            }
        }
    }

    SearchOutcome {
        best_move: best_move_by_visits(&tree).map(|mv| mv.to_string()),
        collisions: tree.collisions,
        cache_hits,
        cache_misses,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plays_a_legal_move_from_the_start_position() {
        let model = Model::load_embedded(Device::Cpu);
        let cfg = SearchConfig { simulations: 16, batch_size: 8, ..SearchConfig::default() };
        let outcome = search_with_stats(&model, "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", &cfg);
        let mv = outcome.best_move.expect("start position has legal moves");
        assert!(mv.len() == 4 || mv.len() == 5);
    }

    #[test]
    fn none_when_game_is_already_over() {
        let model = Model::load_embedded(Device::Cpu);
        let cfg = SearchConfig { simulations: 16, batch_size: 8, ..SearchConfig::default() };
        let outcome =
            search_with_stats(&model, "rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3", &cfg);
        assert!(outcome.best_move.is_none());
    }
}
