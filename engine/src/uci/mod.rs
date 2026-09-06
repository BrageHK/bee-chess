//! UCI process boundary.
//!
//! This is the only module allowed to know about raw UCI text. It parses
//! stdin lines into typed commands and writes typed responses back to
//! stdout. Per ADR 0001, no UCI strings may leak below this module.
//!
//! The engine supports the handshake, evaluator selection via `setoption`,
//! position setup, and synchronous searches. The remaining pieces of a
//! full asynchronous state machine (`stop`, `ponderhit`, and concurrent
//! input handling while searching) land in a follow-up milestone.

use std::io::{BufRead, Write};
use std::time::Instant;

use crate::chess::{Color, Move, PieceKind, Position, Square};
use crate::diagnostics::DiagnosticLevel;
use crate::engine::{Engine, EvaluatorKind, OpeningBookKind};
use crate::search::{mate_in_plies, DEFAULT_MOVE_OVERHEAD_MS};

pub const ENGINE_NAME: &str = "bee-chess";
pub const ENGINE_AUTHOR: &str = "bragehk, johsol and sebasabe";

/// A single move as written in UCI move notation: `e2e4`, or `e7e8q`
/// for a promotion. This is pure protocol text turned into structured
/// data -- it does not know which of the position's legal moves (if
/// any) it actually corresponds to; matching it against legality is
/// `Engine::apply_move`'s job, since that requires board context this
/// type deliberately doesn't carry.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct UciMove {
    pub from: Square,
    pub to: Square,
    pub promotion: Option<PieceKind>,
}

impl UciMove {
    /// Parses a single UCI move token, e.g. `"e2e4"` or `"e7e8q"`.
    /// Returns `None` if `s` isn't a well-formed UCI move token; it does
    /// not check legality.
    pub fn parse(s: &str) -> Option<Self> {
        let (from, to, promotion) = match s.len() {
            4 => (&s[0..2], &s[2..4], None),
            5 => (&s[0..2], &s[2..4], Some(&s[4..5])),
            _ => return None,
        };

        let from = from.parse().ok()?;
        let to = to.parse().ok()?;
        let promotion = match promotion {
            Some(letter) => Some(match letter {
                "q" => PieceKind::Queen,
                "r" => PieceKind::Rook,
                "b" => PieceKind::Bishop,
                "n" => PieceKind::Knight,
                _ => return None,
            }),
            None => None,
        };

        Some(UciMove {
            from,
            to,
            promotion,
        })
    }
}

/// The parsed body of a `position` command: a base position (the
/// starting position, or one given by FEN) plus a sequence of moves to
/// play from it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PositionCommand {
    StartPos { moves: Vec<UciMove> },
    Fen { fen: String, moves: Vec<UciMove> },
}

impl PositionCommand {
    /// Parses the argument portion of a `position` command, i.e.
    /// everything after `"position "`. Accepts `startpos [moves ...]`,
    /// `fen <fen> [moves ...]`. Returns `None` for anything else
    /// (missing/malformed base position); a malformed individual move
    /// in the `moves` list is silently dropped along with the rest of
    /// the list from that point on, since a truncated move list is a
    /// GUI bug we can't recover a sensible position from anyway.
    pub fn parse(args: &str) -> Option<Self> {
        let args = args.trim();

        if let Some(rest) = args.strip_prefix("startpos") {
            let moves = parse_moves_suffix(rest)?;
            return Some(PositionCommand::StartPos { moves });
        }

        let rest = args.strip_prefix("fen")?;
        let rest = rest.strip_prefix(' ')?;
        let (fen, moves) = match rest.find("moves") {
            Some(index) => (rest[..index].trim().to_string(), &rest[index..]),
            None => (rest.trim().to_string(), ""),
        };
        let moves = parse_moves_suffix(moves)?;
        Some(PositionCommand::Fen { fen, moves })
    }

    /// Builds the `Position` this command describes: the base position
    /// with every move in `moves` applied in order via `Engine`'s
    /// legal-move lookup. Returns `Err` (with the base position
    /// unmodified for FEN errors, or partially-applied for an illegal
    /// move) if the FEN is invalid or a move doesn't match any legal
    /// move at the point it's applied.
    fn resolve(&self, engine: &mut Engine) -> Result<(), PositionCommandError> {
        let (base, moves): (Position, &[UciMove]) = match self {
            PositionCommand::StartPos { moves } => (Position::startpos(), moves),
            PositionCommand::Fen { fen, moves } => (
                Position::from_fen(fen).map_err(PositionCommandError::InvalidFen)?,
                moves,
            ),
        };

        engine.set_position(base);
        for &UciMove {
            from,
            to,
            promotion,
        } in moves
        {
            engine
                .apply_move(from, to, promotion)
                .map_err(PositionCommandError::IllegalMove)?;
        }
        Ok(())
    }
}

/// Why a `position` command could not be fully applied.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PositionCommandError {
    InvalidFen(crate::chess::FenError),
    IllegalMove(crate::engine::IllegalMoveError),
}

/// Parses the `[moves e2e4 e7e5 ...]` suffix of a `position` command
/// (or an empty/whitespace-only string, meaning no moves). Returns
/// `None` if a `moves` keyword is present but any move token after it
/// fails to parse.
fn parse_moves_suffix(s: &str) -> Option<Vec<UciMove>> {
    let s = s.trim();
    if s.is_empty() {
        return Some(Vec::new());
    }

    let rest = s.strip_prefix("moves")?;
    rest.split_whitespace().map(UciMove::parse).collect()
}

/// The parsed body of a `go` command. Recognizes `depth <n>` (fixed-
/// depth search, see `Engine::search`), `movetime <ms>` (time-bounded
/// iterative deepening with a single fixed budget, see
/// `Engine::search_for_time`), and the real UCI clock fields
/// (`wtime`/`btime`/`winc`/`binc`/`movestogo`, see
/// `Engine::search_with_clock`). Priority when more than one applies:
/// `movetime` first, then clock fields, then `depth`, matching the
/// dispatch order in `run` below -- movetime and the clock fields are
/// both "time-bounded search," and movetime is the more explicit
/// request when both happen to be present. `infinite`, `ponder`,
/// `stop`, and real cancellation are follow-up work; see
/// `SearchLimits` in `crate::search` for the eventual full shape.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct GoCommand {
    pub depth: Option<u32>,
    pub movetime_ms: Option<u64>,
    pub white_time_ms: Option<u64>,
    pub black_time_ms: Option<u64>,
    pub white_increment_ms: Option<u64>,
    pub black_increment_ms: Option<u64>,
    pub moves_to_go: Option<u32>,
}

impl GoCommand {
    /// Parses the argument portion of a `go` command, i.e. everything
    /// after `"go"`. Recognizes `depth <n>`, `movetime <ms>`, and
    /// `wtime`/`btime`/`winc`/`binc`/`movestogo`; any other token
    /// (`infinite`, `ponder`, ...) is ignored rather than rejected,
    /// since a real `go` line from a GUI may carry fields this
    /// milestone doesn't act on yet, and ignoring them is more useful
    /// than refusing the whole command over them.
    pub fn parse(args: &str) -> Self {
        let tokens: Vec<&str> = args.split_whitespace().collect();
        GoCommand {
            depth: go_field(&tokens, "depth"),
            movetime_ms: go_field(&tokens, "movetime"),
            white_time_ms: go_field(&tokens, "wtime"),
            black_time_ms: go_field(&tokens, "btime"),
            white_increment_ms: go_field(&tokens, "winc"),
            black_increment_ms: go_field(&tokens, "binc"),
            moves_to_go: go_field(&tokens, "movestogo"),
        }
    }

    /// Resolves this command's clock fields to `side`'s own
    /// side-relative `ClockTimeControl` (see that type's docs), or
    /// `None` if the command didn't carry a time-left field for
    /// `side` at all (e.g. `go depth 8`, `go movetime 500`, `go
    /// infinite`) -- `Engine::search_with_clock` should only be used
    /// when this returns `Some`.
    pub fn clock_for(&self, side: Color) -> Option<crate::search::ClockTimeControl> {
        let (time_left_ms, increment_ms) = match side {
            Color::White => (self.white_time_ms?, self.white_increment_ms.unwrap_or(0)),
            Color::Black => (self.black_time_ms?, self.black_increment_ms.unwrap_or(0)),
        };
        Some(crate::search::ClockTimeControl {
            time_left: std::time::Duration::from_millis(time_left_ms),
            increment: std::time::Duration::from_millis(increment_ms),
            moves_to_go: self.moves_to_go,
        })
    }
}

/// Extracts the integer value following `name` in a `go` command's
/// tokens (e.g. `go_field(tokens, "depth")` finds the `4` in `"go
/// depth 4"`), or `None` if `name` isn't present or isn't followed by
/// a valid value of type `T`.
fn go_field<T: std::str::FromStr>(tokens: &[&str], name: &str) -> Option<T> {
    tokens
        .iter()
        .position(|&token| token == name)
        .and_then(|index| tokens.get(index + 1))
        .and_then(|value| value.parse().ok())
}

/// A parsed UCI command. Only a subset of the full protocol is
/// implemented so far; unrecognized input is ignored rather than
/// erroring, per the UCI convention of tolerating unknown commands.
/// `ponder`/`ponderhit` are not implemented yet -- see `GoCommand`'s
/// docs; `stop` (see `Stop` below) and concurrent input handling while
/// searching are, via `run`'s event loop (see its own docs).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum UciCommand {
    Uci,
    IsReady,
    Debug(bool),
    SetOption {
        name: String,
        value: String,
    },
    NewGame,
    Position(PositionCommand),
    Go(GoCommand),
    /// Requests that an in-progress `go` stop as soon as possible and
    /// report its `bestmove` -- see `crate::search::StopSignal`'s
    /// docs. A `stop` with no search running is simply ignored, per
    /// normal UCI tolerance of commands that don't currently apply.
    Stop,
    Quit,
    Unknown(String),
}

impl UciCommand {
    pub fn parse(line: &str) -> Self {
        let line = line.trim();
        match line {
            "uci" => UciCommand::Uci,
            "isready" => UciCommand::IsReady,
            "ucinewgame" => UciCommand::NewGame,
            "stop" => UciCommand::Stop,
            "quit" => UciCommand::Quit,
            _ => {
                if let Some(rest) = line.strip_prefix("position") {
                    return match rest.strip_prefix(' ').and_then(PositionCommand::parse) {
                        Some(command) => UciCommand::Position(command),
                        None => UciCommand::Unknown(line.to_string()),
                    };
                }
                if let Some(rest) = line.strip_prefix("go") {
                    return UciCommand::Go(GoCommand::parse(rest));
                }
                match line.split_whitespace().collect::<Vec<_>>().as_slice() {
                    ["debug", "on"] => UciCommand::Debug(true),
                    ["debug", "off"] => UciCommand::Debug(false),
                    _ => parse_setoption(line)
                        .unwrap_or_else(|| UciCommand::Unknown(line.to_string())),
                }
            }
        }
    }
}

fn parse_setoption(line: &str) -> Option<UciCommand> {
    let rest = line.strip_prefix("setoption name ")?;
    let (name, value) = rest.split_once(" value ")?;
    if name.trim().is_empty() || value.trim().is_empty() {
        return None;
    }
    Some(UciCommand::SetOption {
        name: name.trim().to_string(),
        value: value.trim().to_string(),
    })
}

/// Parses a UCI `check`-type option's value (`true`/`false`, per the
/// protocol's own boolean spelling), case-insensitively.
fn parse_uci_check(value: &str) -> Option<bool> {
    match value.trim().to_ascii_lowercase().as_str() {
        "true" => Some(true),
        "false" => Some(false),
        _ => None,
    }
}

/// Formats a move as UCI long algebraic notation, e.g. `e2e4` or
/// `e7e8q`. The inverse of `UciMove::parse`'s promotion-letter mapping.
fn format_uci_move(mv: Move) -> String {
    match mv.flag().promotion_kind() {
        Some(kind) => {
            let letter = match kind {
                PieceKind::Queen => 'q',
                PieceKind::Rook => 'r',
                PieceKind::Bishop => 'b',
                PieceKind::Knight => 'n',
                PieceKind::Pawn | PieceKind::King => {
                    unreachable!("promotion_kind never returns Pawn or King")
                }
            };
            format!("{}{}{letter}", mv.from(), mv.to())
        }
        None => format!("{}{}", mv.from(), mv.to()),
    }
}

/// Writes one `info depth <n> score cp <n>|mate <n> nodes <n> time
/// <ms> pv ...` line for a completed search result. Real UCI `info`
/// fields, not `info string` -- `info string` is reserved for
/// diagnostics (see `crate::diagnostics`), and this is exactly the
/// structured search telemetry those fields exist for.
fn write_search_info<W: Write>(
    output: &mut W,
    result: &crate::search::SearchResult,
    elapsed: std::time::Duration,
) -> std::io::Result<()> {
    let score_field = match mate_in_plies(result.score) {
        Some(plies_to_mate) => format!("mate {plies_to_mate}"),
        None => format!("cp {}", result.score),
    };
    write!(
        output,
        "info depth {} score {score_field} nodes {} time {}",
        result.depth,
        result.nodes,
        elapsed.as_millis(),
    )?;
    if !result.pv.is_empty() {
        write!(output, " pv")?;
        for mv in &result.pv {
            write!(output, " {}", format_uci_move(*mv))?;
        }
    }
    writeln!(output)
}

/// One line of protocol output a search worker thread wants written,
/// carried back to the event-loop thread over a channel rather than
/// having the worker write to `output` itself -- see `run`'s module
/// docs on why only the event-loop thread ever touches `output`
/// (serializing all UCI text through one place avoids exactly the
/// interleaving bugs a second writer thread would risk).
#[derive(Debug)]
enum SearchEvent {
    /// One completed depth's `info depth ...` line, pre-rendered:
    /// rendering happens on the worker thread (it has the `SearchResult`
    /// and knows its own elapsed time), only the actual `write!` to
    /// `output` happens on the event-loop thread.
    Info(String),
    /// The search has produced its final answer -- exactly one of
    /// these is sent per `go`, always as the worker's last message.
    Done(crate::search::SearchResult),
}

/// Everything that can wake `run`'s event loop up: a new line of input
/// (or the input stream ending), or the active search worker producing
/// another `SearchEvent`. Both a reader thread and a search worker
/// thread send `Event`s into the *same* channel (via cloned senders),
/// so the event loop never needs to poll or `select!` between two
/// separate channels -- it just blocks on one `recv()` and reacts to
/// whichever kind of `Event` arrives first. See the module docs on
/// `run` for why this shape: it's what lets a search's natural
/// completion (with no new command having arrived) still promptly
/// produce a `bestmove`, and what lets a `stop`/new command arriving
/// mid-search be seen immediately rather than only after the current
/// line-read unblocks.
#[derive(Debug)]
enum Event {
    Line(String),
    /// The input stream ended (EOF) -- treated like an implicit `quit`
    /// once any outstanding search is stopped and joined, same as
    /// real UCI GUIs disconnecting.
    InputEnded,
    Search(SearchEvent),
}

/// A `go` running on its own thread: the join handle to reclaim
/// `Engine` and the `StopSignal` `run`'s event loop uses to cancel it
/// (on an explicit `stop`, on `quit`/EOF, or before handling any other
/// command that needs exclusive access to `Engine`). Its `SearchEvent`s
/// arrive as `Event::Search` on the event loop's own channel (see
/// `Event`'s docs), not through a channel owned by this type.
struct SearchWorker {
    handle: std::thread::JoinHandle<Engine>,
    stop: crate::search::StopSignal,
}

impl SearchWorker {
    /// Spawns `go_command` as a new worker, taking ownership of
    /// `engine` for the duration of the search (moved back out via
    /// `join`) -- this is why `run`'s event loop holds `Option<Engine>`
    /// rather than `Engine` directly: exactly one of "the main loop
    /// owns it" or "a worker owns it" is true at any moment, never
    /// both, so nothing else can call an `&mut Engine` method while a
    /// search is in flight. Every `SearchEvent` (each completed
    /// depth's info line, then exactly one final `Done`) is sent as an
    /// `Event::Search` on `events` -- the same channel the event loop
    /// already reads input lines from.
    fn spawn(
        mut engine: Engine,
        go_command: GoCommand,
        events: std::sync::mpsc::Sender<Event>,
    ) -> Self {
        let stop = crate::search::StopSignal::new();
        let worker_stop = stop.clone();

        let handle = std::thread::spawn(move || {
            let side_to_move = engine.position().side_to_move();
            let start = Instant::now();
            let on_depth_complete = {
                let events = events.clone();
                move |result: &crate::search::SearchResult| {
                    let mut line = Vec::new();
                    // A rendering failure into an in-memory `Vec` is not a
                    // realistic failure mode; silently skipping the info
                    // line rather than panicking the worker is the
                    // worst case if it somehow did happen.
                    if write_search_info(&mut line, result, start.elapsed()).is_ok() {
                        if let Ok(text) = String::from_utf8(line) {
                            let _ = events.send(Event::Search(SearchEvent::Info(
                                text.trim_end().to_string(),
                            )));
                        }
                    }
                }
            };

            let result = if let Some(movetime_ms) = go_command.movetime_ms {
                let budget = std::time::Duration::from_millis(movetime_ms);
                engine.search_for_time(budget, worker_stop, on_depth_complete)
            } else if let Some(control) = go_command.clock_for(side_to_move) {
                engine.search_with_clock(control, worker_stop, on_depth_complete)
            } else {
                // Default to a shallow depth when none of movetime/
                // wtime/btime is given, since there's no time-based
                // stopping condition (and so no `StopSignal` plumbing
                // -- `Engine::search` is a fixed-depth, uninterruptible
                // search, same as before) to fall back on instead.
                const DEFAULT_DEPTH: u32 = 4;
                let depth = go_command.depth.unwrap_or(DEFAULT_DEPTH);
                let result = engine.search(depth);
                // depth == 0 means this was a book hit (see
                // `Engine::book_move`'s docs), not a real search --
                // there's no depth/node count to report, so sending an
                // "info depth 0 ..." line for it would misrepresent it
                // as one.
                if result.depth > 0 {
                    on_depth_complete(&result);
                }
                result
            };

            let _ = events.send(Event::Search(SearchEvent::Done(result)));
            engine
        });

        SearchWorker { handle, stop }
    }

    /// Requests cancellation (see `StopSignal`'s docs) and blocks
    /// until the worker finishes, reclaiming `Engine`. Used by an
    /// explicit `stop`/`quit`/EOF, and by any other command that needs
    /// `&mut Engine` while a search is still outstanding -- `run`'s
    /// event loop always finishes the previous `go` before starting a
    /// new one or touching `Engine` any other way. The worker's final
    /// `SearchEvent::Done` (and, along the way, any `Info` lines still
    /// in flight) still arrives as an ordinary `Event::Search` on the
    /// event loop's channel -- this only blocks the *calling* thread
    /// until that has happened, it doesn't consume or short-circuit
    /// those events.
    fn stop_and_join(self) -> Engine {
        self.stop.request_stop();
        self.handle
            .join()
            .expect("search worker thread should not panic")
    }
}

/// Runs the UCI event loop against `input`/`output`, exactly like
/// `run_stdio` (see its docs for the actual async semantics: natural
/// completion, `stop`, `quit`/EOF, and every other command's join-
/// first behavior) -- the difference is purely about thread ownership,
/// not behavior.
///
/// This spawns its reader thread *scoped* to this call
/// (`std::thread::scope`), which is what lets `R` be a borrowed,
/// non-`'static` type (e.g. `some_local_string.as_bytes()`) -- the
/// large majority of this module's own tests use exactly that. The
/// real cost of a scope is that it always joins every thread it
/// spawned before returning, with no way to abandon one early: if
/// `input` never reaches EOF and this event loop returns via `quit`
/// (not EOF) while the reader is blocked inside a `read_line` syscall
/// that has nothing left to read yet, this function will hang waiting
/// for that syscall to return -- which real, never-closing process
/// stdin can never provide, since nothing here can interrupt a
/// blocking `read` from another thread. Every test in this module
/// either sends `quit` only after EOF-reaching input, or otherwise
/// lets its input naturally end, specifically to stay clear of this.
///
/// **This is why `bee.rs`'s real binary uses `run_stdio`, not this
/// function**: real stdin from a GUI/Lichess bridge does not close
/// when Bee itself decides to quit, so a hang here would be a real,
/// user-visible bug (the process never exiting after `quit`), not just
/// a testing footgun. Use this `run` directly only for tests (or any
/// other caller than knows its `input` always reaches EOF promptly);
/// use `run_stdio` for anything driven by a real, possibly-still-open
/// input stream.
pub fn run<R: BufRead + Send, W: Write>(
    input: R,
    mut output: W,
    engine: &mut Engine,
) -> std::io::Result<()> {
    let (sender, events) = std::sync::mpsc::channel::<Event>();

    std::thread::scope(|scope| {
        let reader_sender = sender.clone();
        scope.spawn(move || read_lines_into(input, reader_sender));

        run_event_loop(&sender, &events, &mut output, engine)
    })
}

/// Runs the UCI event loop against real process I/O (`bee.rs`'s actual
/// `main`) -- see `run_event_loop`'s docs for the async semantics this
/// provides (natural completion, `stop`, `quit`/EOF, and every other
/// command joining an outstanding search first).
///
/// Unlike `run`, `input`'s reader thread here is **detached**
/// (`std::thread::spawn`, not `std::thread::scope`) rather than joined
/// before this function returns: real stdin from a GUI/Lichess bridge
/// does not close just because Bee decided to `quit`, so the reader
/// thread can still be blocked inside `read_line` -- with no portable
/// way to cancel a blocking read from another thread in safe stable
/// Rust -- at the exact moment `quit` (or the event loop's own error
/// path) is ready to return. Waiting for that thread anyway (as a
/// scoped thread would force) means the process would never actually
/// exit on `quit` against real stdin, which is a real, user-visible
/// hang, not just a test inconvenience -- see `run`'s own docs for the
/// full explanation of why a scope can't selectively skip its join.
/// Abandoning a reader thread that owns nothing but its own read
/// buffer (no `Engine` access, no `output` writes -- see `Event`'s and
/// `read_lines_into`'s docs) and leaving it to be torn down when the
/// whole process exits right after `main` returns is the accepted
/// trade-off: real UCI GUIs behave this way regardless of what
/// language/threading model an engine uses.
///
/// Requires `R: 'static` (the detached thread must fully own `input`
/// for as long as the process might run, not just for this call's
/// duration) -- this is what actually forces the split from `run`:
/// most of this module's own tests build their input from a local,
/// non-`'static` byte slice/string and use `run` instead.
pub fn run_stdio<R: BufRead + Send + 'static, W: Write>(
    input: R,
    mut output: W,
    engine: &mut Engine,
) -> std::io::Result<()> {
    let (sender, events) = std::sync::mpsc::channel::<Event>();

    let reader_sender = sender.clone();
    std::thread::spawn(move || read_lines_into(input, reader_sender));

    run_event_loop(&sender, &events, &mut output, engine)
}

/// Reads `input` line by line, sending each as an `Event::Line` (or
/// `Event::InputEnded` once it ends) into `sender` -- the body shared
/// by both `run`'s scoped reader thread and `run_stdio`'s detached
/// one. Touches nothing but `input` and `sender`: no `Engine`, no
/// `output` -- see `run`'s module-level docs on why only the event-
/// loop thread itself ever writes protocol output.
fn read_lines_into<R: BufRead>(mut input: R, sender: std::sync::mpsc::Sender<Event>) {
    loop {
        let mut line = String::new();
        match input.read_line(&mut line) {
            Ok(0) => {
                let _ = sender.send(Event::InputEnded);
                return;
            }
            Ok(_) => {
                let trimmed = line.trim_end_matches(['\n', '\r']).to_string();
                if sender.send(Event::Line(trimmed)).is_err() {
                    return; // event loop already exited
                }
            }
            Err(_) => {
                let _ = sender.send(Event::InputEnded);
                return;
            }
        }
    }
}

/// Consumes `events` until `quit`/EOF, driving `engine` and writing to
/// `output` -- the actual event loop `run` wraps with its input-reader
/// thread. Kept separate (and taking a bare channel pair rather than
/// owning any input source) specifically so tests can drive the exact
/// command/search-result interleavings that matter for cancellation (a
/// `stop` arriving mid-search, `quit` while searching, natural
/// completion racing a `stop`, ...) by sending `Event`s directly,
/// deterministically, instead of depending on real thread timing
/// against a static input buffer.
///
/// `sender` is only ever used to hand a clone to each `SearchWorker`
/// this loop spawns (so it can send its own `Event::Search`s back);
/// this function never sends anything through it directly itself.
/// Whoever calls this owns `sender`'s lifetime -- `run` keeps its own
/// reader-thread clone until the reader exits, and a test typically
/// keeps its `Sender` alive for as long as it wants `events.recv()` to
/// block rather than immediately see a disconnected channel.
fn run_event_loop<W: Write>(
    sender: &std::sync::mpsc::Sender<Event>,
    events: &std::sync::mpsc::Receiver<Event>,
    output: &mut W,
    engine: &mut Engine,
) -> std::io::Result<()> {
    // Exactly one of these is ever active at a time -- see
    // `SearchWorker`'s docs on why `Engine` needs to move into (and
    // back out of) a worker rather than being shared.
    let mut active_search: Option<SearchWorker> = None;
    // Events that arrived out of order relative to a search drain --
    // see `drain_and_write_search_events`/`next_event`'s docs.
    let mut pending: std::collections::VecDeque<Event> = std::collections::VecDeque::new();

    loop {
        match next_event(events, &mut pending) {
            Ok(Event::Line(line)) => {
                if engine.debug() {
                    engine.emit_diagnostic(DiagnosticLevel::Debug, format!("received: {line}"));
                }

                let command = UciCommand::parse(&line);

                // Every command except `Stop` itself needs exclusive
                // access to `Engine` -- finish the outstanding search
                // first (see `SearchWorker::stop_and_join`'s docs).
                // `Stop` is handled inline below instead, since a
                // `stop` with nothing running is a normal, silent
                // no-op rather than something that should try to
                // "finish" a search that doesn't exist.
                if !matches!(command, UciCommand::Stop) {
                    if let Some(worker) = active_search.take() {
                        *engine = worker.stop_and_join();
                        drain_and_write_search_events(events, &mut pending, output, engine)?;
                    }
                }

                match command {
                    UciCommand::Uci => {
                        writeln!(output, "id name {ENGINE_NAME}")?;
                        writeln!(output, "id author {ENGINE_AUTHOR}")?;
                        writeln!(output, "option name Evaluator type combo default Positional var Positional var Material var Experimental")?;
                        // Experimental search feature switches -- see
                        // `SearchOptions`'s docs. These default to `true` (the
                        // normal, strongest configuration); Bee Lab's A/B
                        // experiment runner is the intended way to turn one off,
                        // not a permanent engine configuration.
                        writeln!(output, "option name UseTT type check default true")?;
                        writeln!(output, "option name UseQuiescence type check default true")?;
                        writeln!(
                            output,
                            "option name UseEnhancedQuiescence type check default true"
                        )?;
                        // See `crate::book`'s module docs -- `None` is the
                        // default (a book is an opt-in experiment), `Cow` is
                        // the first, deliberately small opening book.
                        writeln!(
                            output,
                            "option name OpeningBook type combo default None var None var Cow"
                        )?;
                        // See `crate::search::TimeManagerConfig::move_overhead`'s
                        // docs -- milliseconds reserved every move for
                        // protocol/process/network delay, never planned as
                        // thinking time. The right value depends on the
                        // deployment (a network round trip to a Lichess bridge
                        // needs more than a local GUI).
                        writeln!(
                            output,
                            "option name MoveOverhead type spin default {} min 0 max 1000",
                            DEFAULT_MOVE_OVERHEAD_MS
                        )?;
                        writeln!(output, "uciok")?;
                    }
                    UciCommand::IsReady => {
                        writeln!(output, "readyok")?;
                    }
                    UciCommand::Debug(on) => {
                        engine.set_debug(on);
                    }
                    UciCommand::SetOption { name, value } => {
                        if name.eq_ignore_ascii_case("Evaluator") {
                            if let Some(evaluator) = EvaluatorKind::parse(&value) {
                                engine.set_evaluator(evaluator);
                            } else {
                                engine.emit_diagnostic(
                                    DiagnosticLevel::Warn,
                                    format!("ignored invalid Evaluator value: {value}"),
                                );
                            }
                        } else if name.eq_ignore_ascii_case("UseTT") {
                            match parse_uci_check(&value) {
                                Some(use_tt) => engine.set_use_tt(use_tt),
                                None => engine.emit_diagnostic(
                                    DiagnosticLevel::Warn,
                                    format!("ignored invalid UseTT value: {value}"),
                                ),
                            }
                        } else if name.eq_ignore_ascii_case("UseQuiescence") {
                            match parse_uci_check(&value) {
                                Some(use_quiescence) => engine.set_use_quiescence(use_quiescence),
                                None => engine.emit_diagnostic(
                                    DiagnosticLevel::Warn,
                                    format!("ignored invalid UseQuiescence value: {value}"),
                                ),
                            }
                        } else if name.eq_ignore_ascii_case("UseEnhancedQuiescence") {
                            match parse_uci_check(&value) {
                                Some(enabled) => engine.set_use_enhanced_quiescence(enabled),
                                None => engine.emit_diagnostic(
                                    DiagnosticLevel::Warn,
                                    format!("ignored invalid UseEnhancedQuiescence value: {value}"),
                                ),
                            }
                        } else if name.eq_ignore_ascii_case("OpeningBook") {
                            if let Some(book) = OpeningBookKind::parse(&value) {
                                engine.set_opening_book(book);
                            } else {
                                engine.emit_diagnostic(
                                    DiagnosticLevel::Warn,
                                    format!("ignored invalid OpeningBook value: {value}"),
                                );
                            }
                        } else if name.eq_ignore_ascii_case("MoveOverhead") {
                            match value.trim().parse::<u64>() {
                                Ok(ms) => {
                                    engine.set_move_overhead(std::time::Duration::from_millis(ms))
                                }
                                Err(_) => engine.emit_diagnostic(
                                    DiagnosticLevel::Warn,
                                    format!("ignored invalid MoveOverhead value: {value}"),
                                ),
                            }
                        } else {
                            engine.emit_diagnostic(
                                DiagnosticLevel::Info,
                                format!("ignored unknown UCI option: {name}"),
                            );
                        }
                    }
                    UciCommand::NewGame => {
                        engine.new_game();
                    }
                    UciCommand::Position(command) => {
                        if let Err(error) = command.resolve(engine) {
                            engine.emit_diagnostic(DiagnosticLevel::Warn, format!("{error:?}"));
                        }
                    }
                    UciCommand::Go(go_command) => {
                        // Ownership of `engine` moves into the worker
                        // for the duration of the search -- `*engine`
                        // is a placeholder default until it's moved
                        // back by whatever eventually joins this
                        // worker (a later Done event below, or an
                        // explicit stop/quit/EOF/next-command join
                        // above). See `SearchWorker`'s docs.
                        let taken = std::mem::take(engine);
                        active_search =
                            Some(SearchWorker::spawn(taken, go_command, sender.clone()));
                    }
                    UciCommand::Stop => {
                        if let Some(worker) = active_search.take() {
                            *engine = worker.stop_and_join();
                            drain_and_write_search_events(events, &mut pending, output, engine)?;
                        }
                        // A stop with nothing running is a normal,
                        // silent no-op, per UCI's general tolerance of
                        // commands that don't currently apply.
                    }
                    UciCommand::Quit => {
                        if let Some(worker) = active_search.take() {
                            *engine = worker.stop_and_join();
                            drain_and_write_search_events(events, &mut pending, output, engine)?;
                        }
                        return Ok(());
                    }
                    UciCommand::Unknown(ref command) => {
                        // Unrecognized commands are ignored, per UCI
                        // convention -- but worth a diagnostic when debug is
                        // on, since a silently-ignored typo (e.g. from a
                        // hand-typed test session) is otherwise invisible.
                        engine.emit_diagnostic(
                            DiagnosticLevel::Info,
                            format!("ignored unknown UCI command: {command}"),
                        );
                    }
                }

                for diagnostic in engine.take_diagnostics() {
                    if engine.debug() {
                        writeln!(output, "info string {}", diagnostic.message)?;
                    }
                }
                output.flush()?;
            }
            Ok(Event::Search(search_event)) => {
                // A `Done` means the active worker has produced its
                // final answer and is about to exit on its own (no
                // `stop` needed) -- clear `active_search` now, or a
                // later `stop_and_join`/next-command join would
                // needlessly re-request cancellation on (and, worse,
                // block forever trying to drain non-existent further
                // events from) a worker that already finished and has
                // nothing left to send.
                let done = matches!(search_event, SearchEvent::Done(_));
                write_search_event(output, engine, search_event)?;
                if done {
                    if let Some(worker) = active_search.take() {
                        *engine = worker.stop_and_join();
                    }
                }
                output.flush()?;
            }
            Ok(Event::InputEnded) | Err(_) => {
                // EOF (a real GUI closing stdin) or the reader thread
                // is gone -- same handling as `quit`: stop and join
                // any outstanding search before exiting.
                if let Some(worker) = active_search.take() {
                    *engine = worker.stop_and_join();
                    drain_and_write_search_events(events, &mut pending, output, engine)?;
                }
                return Ok(());
            }
        }
    }
}

/// Writes one `SearchEvent`'s protocol output. A `Done` result's
/// `bestmove` line always follows immediately after any of its own
/// `Info` lines this same event carries -- but `Info`/`Done` arrive as
/// *separate* channel messages (one per completed depth, per
/// `SearchWorker::spawn`'s docs), so this only ever renders one of
/// them at a time; the ordering guarantee comes from the channel
/// itself (a single `SearchWorker` always sends its own events in
/// order) plus every event being fully written+flushed before the
/// loop reads its next one.
fn write_search_event<W: Write>(
    output: &mut W,
    engine: &mut Engine,
    event: SearchEvent,
) -> std::io::Result<()> {
    match event {
        SearchEvent::Info(line) => writeln!(output, "{line}"),
        SearchEvent::Done(result) => {
            let _ = engine; // reserved for future per-result engine bookkeeping
            match result.best_move {
                Some(mv) => writeln!(output, "bestmove {}", format_uci_move(mv)),
                // No legal moves (checkmate/stalemate): UCI's
                // convention for "no move to make" is bestmove
                // 0000 rather than omitting the response.
                None => writeln!(output, "bestmove 0000"),
            }
        }
    }
}

/// After `stop_and_join` reclaims `Engine`, any `SearchEvent`s that
/// worker sent (its remaining `Info` lines, then its final `Done`) are
/// still sitting in `events`, not yet read -- `stop_and_join` itself
/// doesn't consume them (see its docs). This drains up through that
/// worker's final `Done` and writes each `Event::Search` in order, so
/// its `bestmove` (and any `info` lines leading up to it) reach
/// `output` before whatever triggered the join is itself handled.
///
/// A worker finishing is not the only thing that can produce an event,
/// though: the reader thread's *next* line (or `InputEnded`) can be
/// sent, and received here, before every one of the worker's own
/// already-sent events has been (per-sender order is preserved, but
/// there is no ordering guarantee *between* different senders -- see
/// `Event`'s docs). Since `join()` returning guarantees the worker's
/// thread has already made all of its `send` calls, its own events are
/// guaranteed to all eventually arrive; anything else received while
/// waiting for them (a `Line`/`InputEnded` that raced ahead) is pushed
/// onto `pending` instead of being lost, for the main loop to consume
/// next via `next_event` exactly as if it had arrived normally.
fn drain_and_write_search_events<W: Write>(
    events: &std::sync::mpsc::Receiver<Event>,
    pending: &mut std::collections::VecDeque<Event>,
    output: &mut W,
    engine: &mut Engine,
) -> std::io::Result<()> {
    loop {
        match events.recv() {
            Ok(Event::Search(search_event)) => {
                let done = matches!(search_event, SearchEvent::Done(_));
                write_search_event(output, engine, search_event)?;
                if done {
                    return Ok(());
                }
            }
            Ok(other) => pending.push_back(other),
            Err(_) => return Ok(()), // channel disconnected -- nothing more to drain
        }
    }
}

/// The event loop's single point of receipt: always checks `pending`
/// (events that arrived out of order relative to a search drain -- see
/// `drain_and_write_search_events`'s docs) before blocking on `events`
/// itself, so nothing sent to the channel is ever lost or processed
/// out of the order it was actually sent in.
fn next_event(
    events: &std::sync::mpsc::Receiver<Event>,
    pending: &mut std::collections::VecDeque<Event>,
) -> Result<Event, std::sync::mpsc::RecvError> {
    if let Some(event) = pending.pop_front() {
        return Ok(event);
    }
    events.recv()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A `Read` that reads `text` in order, then blocks briefly before
    /// reporting EOF -- wrap in `BufReader::new(..)` for `run`'s `R:
    /// BufRead` bound (matching how `bee.rs`'s real `main` wraps real
    /// `Stdin` -- see its own docs). See `run`'s docs on why EOF now
    /// cancels an outstanding search exactly like `quit` does: a plain
    /// `b"go movetime 200\nquit\n"` buffer would have `quit` (or EOF,
    /// if `quit` is dropped) arrive as fast as the reader thread can
    /// read it, almost certainly well before a 200ms search finishes,
    /// cancelling it early rather than letting it complete naturally.
    /// Tests that want to observe a *complete*, uninterrupted
    /// `movetime` search use this instead of a bare byte slice, with
    /// `eof_delay` comfortably longer than the budget under test.
    struct SlowEofInput {
        remaining: std::collections::VecDeque<u8>,
        eof_delay: std::time::Duration,
    }

    impl SlowEofInput {
        fn new(text: &str, eof_delay: std::time::Duration) -> Self {
            SlowEofInput {
                remaining: text.bytes().collect(),
                eof_delay,
            }
        }
    }

    impl std::io::Read for SlowEofInput {
        fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
            if self.remaining.is_empty() {
                std::thread::sleep(self.eof_delay);
                return Ok(0);
            }
            let mut n = 0;
            while n < buf.len() {
                let Some(byte) = self.remaining.pop_front() else {
                    break;
                };
                buf[n] = byte;
                n += 1;
            }
            Ok(n)
        }
    }

    #[test]
    fn parses_known_commands() {
        assert_eq!(UciCommand::parse("uci"), UciCommand::Uci);
        assert_eq!(UciCommand::parse("isready"), UciCommand::IsReady);
        assert_eq!(UciCommand::parse("ucinewgame"), UciCommand::NewGame);
        assert_eq!(UciCommand::parse("quit"), UciCommand::Quit);
    }

    #[test]
    fn go_command_parses_depth() {
        assert_eq!(
            UciCommand::parse("go depth 1"),
            UciCommand::Go(GoCommand {
                depth: Some(1),
                ..Default::default()
            })
        );
        assert_eq!(
            UciCommand::parse("go depth 12"),
            UciCommand::Go(GoCommand {
                depth: Some(12),
                ..Default::default()
            })
        );
    }

    #[test]
    fn go_command_with_no_depth_has_none() {
        assert_eq!(
            UciCommand::parse("go"),
            UciCommand::Go(GoCommand::default())
        );
    }

    #[test]
    fn go_command_ignores_unrecognized_fields() {
        // A real GUI's `go` line carries fields (`ponder`, `infinite`,
        // ...) this milestone doesn't act on yet; they must not
        // prevent parsing the fields we do recognize.
        assert_eq!(
            UciCommand::parse("go ponder depth 3"),
            UciCommand::Go(GoCommand {
                depth: Some(3),
                ..Default::default()
            })
        );
    }

    #[test]
    fn go_command_parses_clock_fields() {
        assert_eq!(
            UciCommand::parse("go wtime 300000 btime 295000 winc 2000 binc 1000 movestogo 20"),
            UciCommand::Go(GoCommand {
                white_time_ms: Some(300_000),
                black_time_ms: Some(295_000),
                white_increment_ms: Some(2_000),
                black_increment_ms: Some(1_000),
                moves_to_go: Some(20),
                ..Default::default()
            })
        );
    }

    #[test]
    fn go_command_clock_for_resolves_the_requested_sides_own_fields() {
        let go_command = GoCommand {
            white_time_ms: Some(60_000),
            black_time_ms: Some(55_000),
            white_increment_ms: Some(1_000),
            black_increment_ms: None,
            moves_to_go: Some(10),
            ..Default::default()
        };

        let white = go_command.clock_for(Color::White).unwrap();
        assert_eq!(white.time_left, std::time::Duration::from_millis(60_000));
        assert_eq!(white.increment, std::time::Duration::from_millis(1_000));
        assert_eq!(white.moves_to_go, Some(10));

        let black = go_command.clock_for(Color::Black).unwrap();
        assert_eq!(black.time_left, std::time::Duration::from_millis(55_000));
        assert_eq!(
            black.increment,
            std::time::Duration::ZERO,
            "missing binc defaults to zero, not None/panic"
        );
    }

    #[test]
    fn go_command_clock_for_is_none_without_a_time_left_field() {
        assert_eq!(GoCommand::default().clock_for(Color::White), None);
        assert_eq!(
            GoCommand {
                depth: Some(4),
                ..Default::default()
            }
            .clock_for(Color::Black),
            None
        );
    }

    #[test]
    fn go_command_with_malformed_depth_value_has_none() {
        assert_eq!(
            UciCommand::parse("go depth notanumber"),
            UciCommand::Go(GoCommand::default())
        );
    }

    #[test]
    fn go_command_parses_movetime() {
        assert_eq!(
            UciCommand::parse("go movetime 100"),
            UciCommand::Go(GoCommand {
                movetime_ms: Some(100),
                ..Default::default()
            })
        );
    }

    #[test]
    fn go_command_parses_both_depth_and_movetime() {
        assert_eq!(
            UciCommand::parse("go depth 6 movetime 5000"),
            UciCommand::Go(GoCommand {
                depth: Some(6),
                movetime_ms: Some(5000),
                ..Default::default()
            })
        );
    }

    #[test]
    fn format_uci_move_formats_quiet_move() {
        let mv = UciMove::parse("e2e4").unwrap();
        let position_move = crate::chess::Move::new(mv.from, mv.to, crate::chess::MoveFlag::Quiet);
        assert_eq!(format_uci_move(position_move), "e2e4");
    }

    #[test]
    fn format_uci_move_formats_every_promotion_letter() {
        use crate::chess::MoveFlag;
        let from = "a7".parse().unwrap();
        let to = "a8".parse().unwrap();
        assert_eq!(
            format_uci_move(crate::chess::Move::new(from, to, MoveFlag::PromoteQueen)),
            "a7a8q"
        );
        assert_eq!(
            format_uci_move(crate::chess::Move::new(from, to, MoveFlag::PromoteRook)),
            "a7a8r"
        );
        assert_eq!(
            format_uci_move(crate::chess::Move::new(from, to, MoveFlag::PromoteBishop)),
            "a7a8b"
        );
        assert_eq!(
            format_uci_move(crate::chess::Move::new(from, to, MoveFlag::PromoteKnight)),
            "a7a8n"
        );
    }

    #[test]
    fn uci_move_parses_quiet_move() {
        let mv = UciMove::parse("e2e4").expect("should parse");
        assert_eq!(mv.from, "e2".parse().unwrap());
        assert_eq!(mv.to, "e4".parse().unwrap());
        assert_eq!(mv.promotion, None);
    }

    #[test]
    fn uci_move_parses_promotion() {
        let mv = UciMove::parse("e7e8q").expect("should parse");
        assert_eq!(mv.from, "e7".parse().unwrap());
        assert_eq!(mv.to, "e8".parse().unwrap());
        assert_eq!(mv.promotion, Some(PieceKind::Queen));
    }

    #[test]
    fn uci_move_parses_every_promotion_letter() {
        assert_eq!(
            UciMove::parse("a7a8q").unwrap().promotion,
            Some(PieceKind::Queen)
        );
        assert_eq!(
            UciMove::parse("a7a8r").unwrap().promotion,
            Some(PieceKind::Rook)
        );
        assert_eq!(
            UciMove::parse("a7a8b").unwrap().promotion,
            Some(PieceKind::Bishop)
        );
        assert_eq!(
            UciMove::parse("a7a8n").unwrap().promotion,
            Some(PieceKind::Knight)
        );
    }

    #[test]
    fn uci_move_rejects_malformed_tokens() {
        assert_eq!(UciMove::parse(""), None);
        assert_eq!(UciMove::parse("e2e"), None);
        assert_eq!(UciMove::parse("e2e4x"), None); // bad promotion letter
        assert_eq!(UciMove::parse("z2e4"), None); // bad square
        assert_eq!(UciMove::parse("e2e4qq"), None); // too long
    }

    #[test]
    fn position_command_parses_startpos_with_no_moves() {
        assert_eq!(
            PositionCommand::parse("startpos"),
            Some(PositionCommand::StartPos { moves: Vec::new() })
        );
    }

    #[test]
    fn position_command_parses_startpos_with_moves() {
        let parsed = PositionCommand::parse("startpos moves e2e4 e7e5 g1f3").unwrap();
        assert_eq!(
            parsed,
            PositionCommand::StartPos {
                moves: vec![
                    UciMove::parse("e2e4").unwrap(),
                    UciMove::parse("e7e5").unwrap(),
                    UciMove::parse("g1f3").unwrap(),
                ]
            }
        );
    }

    #[test]
    fn position_command_parses_fen_with_no_moves() {
        let fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
        assert_eq!(
            PositionCommand::parse(&format!("fen {fen}")),
            Some(PositionCommand::Fen {
                fen: fen.to_string(),
                moves: Vec::new(),
            })
        );
    }

    #[test]
    fn position_command_parses_fen_with_moves() {
        let fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
        let parsed = PositionCommand::parse(&format!("fen {fen} moves e2e4 e7e5")).unwrap();
        assert_eq!(
            parsed,
            PositionCommand::Fen {
                fen: fen.to_string(),
                moves: vec![
                    UciMove::parse("e2e4").unwrap(),
                    UciMove::parse("e7e5").unwrap()
                ],
            }
        );
    }

    #[test]
    fn position_command_rejects_unknown_base() {
        assert_eq!(PositionCommand::parse("nonsense"), None);
        assert_eq!(PositionCommand::parse(""), None);
    }

    #[test]
    fn position_command_rejects_malformed_move_in_list() {
        assert_eq!(PositionCommand::parse("startpos moves e2e4 bogus"), None);
    }

    #[test]
    fn parses_unknown_command() {
        assert_eq!(
            UciCommand::parse("bogus"),
            UciCommand::Unknown("bogus".to_string())
        );
    }

    #[test]
    fn parses_debug_on_and_off() {
        assert_eq!(UciCommand::parse("debug on"), UciCommand::Debug(true));
        assert_eq!(UciCommand::parse("debug off"), UciCommand::Debug(false));
    }

    #[test]
    fn parses_debug_with_bad_argument_as_unknown() {
        assert_eq!(
            UciCommand::parse("debug maybe"),
            UciCommand::Unknown("debug maybe".to_string())
        );
    }

    #[test]
    fn parses_setoption_with_multiword_name_and_value() {
        assert_eq!(
            UciCommand::parse("setoption name Future Evaluator value Some Model"),
            UciCommand::SetOption {
                name: "Future Evaluator".to_string(),
                value: "Some Model".to_string(),
            }
        );
    }

    #[test]
    fn uci_handshake_produces_expected_output() {
        let input = b"uci\nisready\nquit\n".as_slice();
        let mut output = Vec::new();
        let mut engine = Engine::default();
        run(input, &mut output, &mut engine).expect("run should succeed");
        let text = String::from_utf8(output).expect("output should be valid utf8");
        assert!(text.contains(&format!("id name {ENGINE_NAME}")));
        assert!(text.contains(&format!("id author {ENGINE_AUTHOR}")));
        assert!(text.contains("option name Evaluator type combo default Positional"));
        assert!(text.contains("var Experimental"));
        assert!(text.contains("option name UseTT type check default true"));
        assert!(text.contains("option name UseQuiescence type check default true"));
        assert!(text.contains("option name UseEnhancedQuiescence type check default true"));
        assert!(text.contains("option name OpeningBook type combo default None"));
        assert!(text.contains("uciok"));
        assert!(text.contains("readyok"));
    }

    #[test]
    fn setoption_changes_the_evaluator() {
        let input = b"setoption name Evaluator value Material\nquit\n".as_slice();
        let mut output = Vec::new();
        let mut engine = Engine::default();
        run(input, &mut output, &mut engine).expect("run should succeed");
        assert_eq!(engine.evaluator(), EvaluatorKind::Material);
    }

    #[test]
    fn setoption_selects_the_experimental_evaluator() {
        let input = b"setoption name Evaluator value Experimental\nquit\n".as_slice();
        let mut output = Vec::new();
        let mut engine = Engine::default();
        run(input, &mut output, &mut engine).expect("run should succeed");
        assert_eq!(engine.evaluator(), EvaluatorKind::Experimental);
    }

    #[test]
    fn setoption_disables_the_transposition_table() {
        let input = b"setoption name UseTT value false\nquit\n".as_slice();
        let mut output = Vec::new();
        let mut engine = Engine::default();
        run(input, &mut output, &mut engine).expect("run should succeed");
        assert!(!engine.search_options().use_tt);
    }

    #[test]
    fn setoption_disables_quiescence() {
        let input = b"setoption name UseQuiescence value false\nquit\n".as_slice();
        let mut output = Vec::new();
        let mut engine = Engine::default();
        run(input, &mut output, &mut engine).expect("run should succeed");
        assert!(!engine.search_options().use_quiescence);
    }

    #[test]
    fn setoption_disables_enhanced_quiescence() {
        let input = b"setoption name UseEnhancedQuiescence value false\nquit\n".as_slice();
        let mut output = Vec::new();
        let mut engine = Engine::default();
        run(input, &mut output, &mut engine).expect("run should succeed");
        assert!(!engine.search_options().use_enhanced_quiescence);
    }

    #[test]
    fn setoption_enables_the_cow_opening_book() {
        let input = b"setoption name OpeningBook value Cow\nquit\n".as_slice();
        let mut output = Vec::new();
        let mut engine = Engine::default();
        run(input, &mut output, &mut engine).expect("run should succeed");
        assert_eq!(engine.opening_book_kind(), OpeningBookKind::Cow);
    }

    #[test]
    fn invalid_opening_book_value_keeps_the_current_setting() {
        let input = b"setoption name OpeningBook value NotARealBook\nquit\n".as_slice();
        let mut output = Vec::new();
        let mut engine = Engine::default();
        run(input, &mut output, &mut engine).expect("run should succeed");
        assert_eq!(engine.opening_book_kind(), OpeningBookKind::None);
    }

    #[test]
    fn cow_opening_book_plays_e3_instantly_via_the_real_uci_protocol() {
        let input = b"setoption name OpeningBook value Cow\ngo depth 4\nquit\n".as_slice();
        let mut output = Vec::new();
        let mut engine = Engine::default();
        run(input, &mut output, &mut engine).expect("run should succeed");
        let text = String::from_utf8(output).expect("output should be valid utf8");
        assert!(text.contains("bestmove e2e3"));
        // A book hit has no real search depth -- no "info depth ..."
        // line should have been written for it.
        assert!(!text.contains("info depth"));
    }

    #[test]
    fn invalid_use_tt_value_keeps_the_current_setting() {
        let input = b"setoption name UseTT value maybe\nquit\n".as_slice();
        let mut output = Vec::new();
        let mut engine = Engine::default();
        run(input, &mut output, &mut engine).expect("run should succeed");
        assert!(
            engine.search_options().use_tt,
            "default is true; an invalid value must not change it"
        );
    }

    #[test]
    fn invalid_evaluator_keeps_the_current_evaluator() {
        let input = b"setoption name Evaluator value Unknown\nquit\n".as_slice();
        let mut output = Vec::new();
        let mut engine = Engine::default();
        run(input, &mut output, &mut engine).expect("run should succeed");
        assert_eq!(engine.evaluator(), EvaluatorKind::Positional);
    }

    #[test]
    fn debug_on_sets_engine_debug_flag() {
        let input = b"debug on\nquit\n".as_slice();
        let mut output = Vec::new();
        let mut engine = Engine::default();
        run(input, &mut output, &mut engine).expect("run should succeed");
        assert!(engine.debug());
    }

    #[test]
    fn debug_off_clears_engine_debug_flag() {
        let input = b"debug on\ndebug off\nquit\n".as_slice();
        let mut output = Vec::new();
        let mut engine = Engine::default();
        run(input, &mut output, &mut engine).expect("run should succeed");
        assert!(!engine.debug());
    }

    #[test]
    fn debug_mode_echoes_received_commands() {
        let input = b"debug on\nisready\nquit\n".as_slice();
        let mut output = Vec::new();
        let mut engine = Engine::default();
        run(input, &mut output, &mut engine).expect("run should succeed");
        let text = String::from_utf8(output).expect("output should be valid utf8");
        assert!(text.contains("info string received: isready"));
    }

    #[test]
    fn debug_off_by_default_does_not_echo_commands() {
        let input = b"isready\nquit\n".as_slice();
        let mut output = Vec::new();
        let mut engine = Engine::default();
        run(input, &mut output, &mut engine).expect("run should succeed");
        let text = String::from_utf8(output).expect("output should be valid utf8");
        assert!(!text.contains("info string"));
    }

    #[test]
    fn unknown_command_produces_diagnostic_when_debug_is_on() {
        let input = b"debug on\nbogus-command\nquit\n".as_slice();
        let mut output = Vec::new();
        let mut engine = Engine::default();
        run(input, &mut output, &mut engine).expect("run should succeed");
        let text = String::from_utf8(output).expect("output should be valid utf8");
        assert!(text.contains("info string ignored unknown UCI command: bogus-command"));
    }

    #[test]
    fn unknown_command_is_silent_when_debug_is_off() {
        let input = b"bogus-command\nquit\n".as_slice();
        let mut output = Vec::new();
        let mut engine = Engine::default();
        run(input, &mut output, &mut engine).expect("run should succeed");
        let text = String::from_utf8(output).expect("output should be valid utf8");
        assert!(!text.contains("info string"));
        assert!(!text.contains("bogus-command"));
    }

    #[test]
    fn diagnostics_do_not_accumulate_unboundedly_when_debug_is_off() {
        // Diagnostics are drained every line regardless of debug state
        // (only *writing* them is gated), so nothing should pile up in
        // the engine even across many silently-ignored commands.
        let mut input = String::new();
        for _ in 0..50 {
            input.push_str("bogus-command\n");
        }
        input.push_str("quit\n");

        let mut output = Vec::new();
        let mut engine = Engine::default();
        run(input.as_bytes(), &mut output, &mut engine).expect("run should succeed");

        assert_eq!(engine.take_diagnostics(), Vec::new());
    }

    #[test]
    fn invalid_position_command_emits_a_diagnostic_when_debug_is_on() {
        let input = b"debug on\nposition startpos moves e2e4 e2e4\nquit\n".as_slice();
        let mut output = Vec::new();
        let mut engine = Engine::default();
        run(input, &mut output, &mut engine).expect("run should succeed");
        let text = String::from_utf8(output).expect("output should be valid utf8");
        assert!(text.contains("info string IllegalMove"));
    }

    /// Applies `mv` to `position` by looking it up among the position's
    /// legal moves, exactly the way `Engine::apply_move` does -- used
    /// by tests to build the expected position independently of the
    /// engine/uci plumbing under test.
    fn apply_legal_move(position: &mut Position, mv: UciMove) {
        let legal_move = position
            .generate_legal_moves()
            .into_iter()
            .find(|m| {
                m.from() == mv.from && m.to() == mv.to && m.flag().promotion_kind() == mv.promotion
            })
            .unwrap_or_else(|| panic!("{mv:?} should be legal"));
        position.make_move(legal_move);
    }

    #[test]
    fn position_startpos_moves_produces_expected_position() {
        // The milestone's headline acceptance case: position startpos
        // moves e2e4 e7e5 g1f3, then the engine's position must be
        // exactly the position that results from playing those moves.
        let input = b"position startpos moves e2e4 e7e5 g1f3\nquit\n".as_slice();
        let mut output = Vec::new();
        let mut engine = Engine::default();
        run(input, &mut output, &mut engine).expect("run should succeed");

        let mut expected = Position::startpos();
        apply_legal_move(&mut expected, UciMove::parse("e2e4").unwrap());
        apply_legal_move(&mut expected, UciMove::parse("e7e5").unwrap());
        apply_legal_move(&mut expected, UciMove::parse("g1f3").unwrap());

        assert_eq!(engine.position(), &expected);
    }

    #[test]
    fn position_startpos_with_no_moves_is_startpos() {
        let input = b"position startpos\nquit\n".as_slice();
        let mut output = Vec::new();
        let mut engine = Engine::default();
        run(input, &mut output, &mut engine).expect("run should succeed");
        assert_eq!(engine.position(), &Position::startpos());
    }

    #[test]
    fn position_fen_moves_produces_expected_position() {
        // FEN parsing plus a move list is where parsing bugs tend to
        // show up (field boundaries, the "moves" keyword split), so
        // this is exercised directly rather than only via startpos.
        let fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
        let input = format!("position fen {fen} moves e2e4 e7e5\nquit\n");
        let mut output = Vec::new();
        let mut engine = Engine::default();
        run(input.as_bytes(), &mut output, &mut engine).expect("run should succeed");

        let mut expected = Position::from_fen(fen).expect("valid FEN");
        apply_legal_move(&mut expected, UciMove::parse("e2e4").unwrap());
        apply_legal_move(&mut expected, UciMove::parse("e7e5").unwrap());

        assert_eq!(engine.position(), &expected);
    }

    #[test]
    fn position_fen_with_no_moves_matches_from_fen() {
        let fen = "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3";
        let input = format!("position fen {fen}\nquit\n");
        let mut output = Vec::new();
        let mut engine = Engine::default();
        run(input.as_bytes(), &mut output, &mut engine).expect("run should succeed");
        assert_eq!(
            engine.position(),
            &Position::from_fen(fen).expect("valid FEN")
        );
    }

    #[test]
    fn position_fen_with_reduced_castling_rights_and_promotion() {
        // A "nasty" position: reduced castling rights plus a move list
        // that ends in a promotion.
        let fen = "8/P6k/8/8/8/8/8/7K w - - 0 1";
        let input = format!("position fen {fen} moves a7a8q\nquit\n");
        let mut output = Vec::new();
        let mut engine = Engine::default();
        run(input.as_bytes(), &mut output, &mut engine).expect("run should succeed");

        let mut expected = Position::from_fen(fen).expect("valid FEN");
        apply_legal_move(&mut expected, UciMove::parse("a7a8q").unwrap());

        assert_eq!(engine.position(), &expected);
    }

    #[test]
    fn ucinewgame_does_not_reset_a_previously_set_position() {
        // ucinewgame conceptually resets game/search-specific state; it
        // is always followed by a `position` command that establishes
        // the actual position, so it must not itself reset the board.
        let input = b"position startpos moves e2e4\nucinewgame\nquit\n".as_slice();
        let mut output = Vec::new();
        let mut engine = Engine::default();
        run(input, &mut output, &mut engine).expect("run should succeed");

        let mut expected = Position::startpos();
        apply_legal_move(&mut expected, UciMove::parse("e2e4").unwrap());

        assert_eq!(engine.position(), &expected);
    }

    #[test]
    fn illegal_move_in_list_leaves_position_at_last_legal_state() {
        let input = b"position startpos moves e2e4 e2e4\nquit\n".as_slice();
        let mut output = Vec::new();
        let mut engine = Engine::default();
        run(input, &mut output, &mut engine).expect("run should succeed");

        // e2e4 is legal once (from startpos); playing it again from the
        // resulting position is illegal (there's no white pawn on e2
        // any more), so the position should reflect exactly one move
        // applied.
        let mut expected = Position::startpos();
        apply_legal_move(&mut expected, UciMove::parse("e2e4").unwrap());

        assert_eq!(engine.position(), &expected);
    }

    /// The vertical path the milestone asks for, verbatim:
    ///
    ///     uci
    ///     isready
    ///     position startpos
    ///     go depth 1
    ///     quit
    ///
    /// asserting the returned bestmove is a move that is actually legal
    /// in the resulting position -- not merely that the output looks
    /// like a move.
    #[test]
    fn go_depth_1_from_startpos_returns_a_legal_move() {
        let input = b"uci\nisready\nposition startpos\ngo depth 1\nquit\n".as_slice();
        let mut output = Vec::new();
        let mut engine = Engine::default();
        run(input, &mut output, &mut engine).expect("run should succeed");
        let text = String::from_utf8(output).expect("output should be valid utf8");

        let bestmove_line = text
            .lines()
            .find(|line| line.starts_with("bestmove "))
            .expect("should emit a bestmove line");
        let uci_move = bestmove_line.strip_prefix("bestmove ").unwrap();

        let legal_moves = Position::startpos().generate_legal_moves();
        assert!(
            legal_moves
                .into_iter()
                .any(|mv| format_uci_move(mv) == uci_move),
            "{uci_move} is not among startpos's legal moves"
        );
    }

    #[test]
    fn go_depth_1_returns_the_only_legal_move_when_there_is_exactly_one() {
        // White king a1 in check from a black rook on e1 (checks along
        // rank 1); black king c3 covers every flight square except a2.
        // a1a2 is the position's one and only legal move.
        let fen = "8/8/8/8/8/2k5/8/K3r3 w - - 0 1";
        let position = Position::from_fen(fen).expect("valid FEN");
        assert_eq!(
            position.generate_legal_moves().len(),
            1,
            "test setup: expected exactly one legal move"
        );

        let input = format!("position fen {fen}\ngo depth 1\nquit\n");
        let mut output = Vec::new();
        let mut engine = Engine::default();
        run(input.as_bytes(), &mut output, &mut engine).expect("run should succeed");
        let text = String::from_utf8(output).expect("output should be valid utf8");

        assert!(
            text.lines().any(|line| line == "bestmove a1a2"),
            "expected `bestmove a1a2`, got: {text:?}"
        );
    }

    #[test]
    fn go_with_no_legal_moves_returns_null_move() {
        // Checkmate: no legal moves exist, so bestmove must be the UCI
        // null-move convention "0000" rather than omitting the response
        // or panicking.
        let fen = "6k1/8/8/8/8/8/5PPP/r6K w - - 0 1";
        let position = Position::from_fen(fen).expect("valid FEN");
        assert!(
            position.generate_legal_moves().is_empty(),
            "test setup: expected checkmate"
        );

        let input = format!("position fen {fen}\ngo depth 1\nquit\n");
        let mut output = Vec::new();
        let mut engine = Engine::default();
        run(input.as_bytes(), &mut output, &mut engine).expect("run should succeed");
        let text = String::from_utf8(output).expect("output should be valid utf8");

        assert!(text.lines().any(|line| line == "bestmove 0000"));
    }

    #[test]
    fn go_depth_4_reports_info_fields_before_bestmove() {
        let input = b"position startpos moves e2e4 e7e5\ngo depth 4\nquit\n".as_slice();
        let mut output = Vec::new();
        let mut engine = Engine::default();
        run(input, &mut output, &mut engine).expect("run should succeed");
        let text = String::from_utf8(output).expect("output should be valid utf8");

        let info_line = text
            .lines()
            .find(|line| line.starts_with("info depth"))
            .expect("should emit an info line");
        assert!(info_line.contains("depth 4"));
        assert!(info_line.contains("score cp") || info_line.contains("score mate"));
        assert!(info_line.contains("nodes"));
        assert!(info_line.contains("time"));

        let info_index = text.lines().position(|line| line == info_line).unwrap();
        let bestmove_index = text
            .lines()
            .position(|line| line.starts_with("bestmove"))
            .expect("should emit a bestmove line");
        assert!(
            info_index < bestmove_index,
            "info must be reported before bestmove"
        );
    }

    #[test]
    fn go_depth_4_searches_a_real_tree_and_finds_mate_in_one() {
        // The milestone's target behavior: go depth N actually searches
        // rather than returning the first legal move, and reports a
        // mate score via the proper UCI `score mate` field (not
        // `info string`).
        let fen = "6k1/5ppp/8/8/8/8/8/3QK3 w - - 0 1";
        let input = format!("position fen {fen}\ngo depth 4\nquit\n");
        let mut output = Vec::new();
        let mut engine = Engine::default();
        run(input.as_bytes(), &mut output, &mut engine).expect("run should succeed");
        let text = String::from_utf8(output).expect("output should be valid utf8");

        assert!(
            text.lines().any(|line| line == "bestmove d1d8"),
            "expected `bestmove d1d8`, got: {text:?}"
        );
        assert!(
            text.lines()
                .any(|line| line.starts_with("info") && line.contains("score mate")),
            "expected a `score mate` info line, got: {text:?}"
        );
    }

    #[test]
    fn go_with_no_depth_uses_a_default_depth() {
        let input = b"position startpos\ngo\nquit\n".as_slice();
        let mut output = Vec::new();
        let mut engine = Engine::default();
        run(input, &mut output, &mut engine).expect("run should succeed");
        let text = String::from_utf8(output).expect("output should be valid utf8");

        assert!(text.lines().any(|line| line.starts_with("bestmove")));
        assert!(text.lines().any(|line| line.starts_with("info depth")));
    }

    #[test]
    fn go_movetime_reports_one_info_line_per_completed_depth() {
        // No `quit`: EOF now cancels an outstanding search exactly
        // like `quit` does (see `run`'s docs), so this must let the
        // 200ms movetime search complete naturally before EOF arrives
        // -- see `SlowEofInput`'s docs.
        let input = std::io::BufReader::new(SlowEofInput::new(
            "position startpos\ngo movetime 200\n",
            std::time::Duration::from_millis(300),
        ));
        let mut output = Vec::new();
        let mut engine = Engine::default();
        run(input, &mut output, &mut engine).expect("run should succeed");
        let text = String::from_utf8(output).expect("output should be valid utf8");

        let info_lines: Vec<&str> = text
            .lines()
            .filter(|l| l.starts_with("info depth"))
            .collect();
        assert!(
            info_lines.len() >= 2,
            "expected multiple depths reported within 200ms, got: {info_lines:?}"
        );

        // Depths must be strictly increasing, starting at 1.
        let depths: Vec<u32> = info_lines
            .iter()
            .map(|line| {
                line.split_whitespace()
                    .nth(2)
                    .unwrap()
                    .parse()
                    .expect("depth should be a number")
            })
            .collect();
        assert_eq!(depths.first(), Some(&1));
        for pair in depths.windows(2) {
            assert_eq!(pair[1], pair[0] + 1);
        }

        assert!(text.lines().any(|line| line.starts_with("bestmove")));
    }

    #[test]
    fn go_wtime_btime_drives_a_real_time_bounded_search() {
        // White to move, plenty of time -- should behave like any
        // other iterative-deepening search: multiple increasing
        // depths, then a bestmove. No `quit`: see `SlowEofInput`'s
        // docs on why EOF must be delayed for this to observe a
        // naturally-completed search rather than one cancelled early.
        let input = std::io::BufReader::new(SlowEofInput::new(
            "position startpos\ngo wtime 5000 btime 5000\n",
            std::time::Duration::from_millis(300),
        ));
        let mut output = Vec::new();
        let mut engine = Engine::default();
        run(input, &mut output, &mut engine).expect("run should succeed");
        let text = String::from_utf8(output).expect("output should be valid utf8");

        assert!(
            text.lines().any(|line| line.starts_with("info depth")),
            "expected at least one real search depth, got: {text:?}"
        );
        assert!(text.lines().any(|line| line.starts_with("bestmove")));
    }

    #[test]
    fn go_wtime_uses_whites_own_clock_not_blacks() {
        // White has almost no time, Black has plenty -- if `go`
        // resolved the wrong side's clock, this would search for
        // seconds instead of returning almost immediately.
        let input = b"position startpos\ngo wtime 20 btime 300000\nquit\n".as_slice();
        let mut output = Vec::new();
        let mut engine = Engine::default();

        let start = Instant::now();
        run(input, &mut output, &mut engine).expect("run should succeed");
        let elapsed = start.elapsed();

        assert!(
            elapsed < std::time::Duration::from_millis(500),
            "go should have used White's own (tiny) clock, took {elapsed:?}"
        );
        let text = String::from_utf8(output).expect("output should be valid utf8");
        assert!(text.lines().any(|line| line.starts_with("bestmove")));
    }

    #[test]
    fn go_with_near_zero_time_left_still_returns_a_legal_bestmove() {
        // An extreme time control (e.g. lagging badly on Lichess):
        // Engine must fall back to a legal move rather than ever
        // omitting `bestmove` or crashing.
        let input = b"position startpos\ngo wtime 1 btime 1\nquit\n".as_slice();
        let mut output = Vec::new();
        let mut engine = Engine::default();
        run(input, &mut output, &mut engine).expect("run should succeed");
        let text = String::from_utf8(output).expect("output should be valid utf8");

        let bestmove_line = text
            .lines()
            .find(|line| line.starts_with("bestmove"))
            .expect("bestmove line should always be present");
        assert_ne!(bestmove_line, "bestmove 0000");
    }

    #[test]
    fn setoption_move_overhead_is_reflected_in_a_tighter_time_budget() {
        // Indirect but real end-to-end check: a huge MoveOverhead
        // eating the entire clock must make Engine fall back to the
        // first legal move (depth 0, no search) rather than run any
        // real search at all.
        let input =
            b"setoption name MoveOverhead value 10000\nposition startpos\ngo wtime 100 btime 100\nquit\n"
                .as_slice();
        let mut output = Vec::new();
        let mut engine = Engine::default();
        run(input, &mut output, &mut engine).expect("run should succeed");
        let text = String::from_utf8(output).expect("output should be valid utf8");

        assert!(
            !text.lines().any(|line| line.starts_with("info depth")),
            "a MoveOverhead larger than the whole clock should leave zero usable time, got: {text:?}"
        );
        assert!(text.lines().any(|line| line.starts_with("bestmove")));
    }

    #[test]
    fn go_movetime_finds_mate_in_one_via_iterative_deepening() {
        let fen = "6k1/5ppp/8/8/8/8/8/3QK3 w - - 0 1";
        // No `quit`: see `SlowEofInput`'s docs.
        let input = std::io::BufReader::new(SlowEofInput::new(
            &format!("position fen {fen}\ngo movetime 500\n"),
            std::time::Duration::from_millis(300),
        ));
        let mut output = Vec::new();
        let mut engine = Engine::default();
        run(input, &mut output, &mut engine).expect("run should succeed");
        let text = String::from_utf8(output).expect("output should be valid utf8");

        assert!(
            text.lines().any(|line| line == "bestmove d1d8"),
            "expected `bestmove d1d8`, got: {text:?}"
        );
        assert!(text
            .lines()
            .any(|line| line.starts_with("info") && line.contains("score mate")));
    }

    #[test]
    fn go_movetime_info_line_includes_pv() {
        // No `quit`: see `SlowEofInput`'s docs.
        let input = std::io::BufReader::new(SlowEofInput::new(
            "position startpos\ngo movetime 100\n",
            std::time::Duration::from_millis(300),
        ));
        let mut output = Vec::new();
        let mut engine = Engine::default();
        run(input, &mut output, &mut engine).expect("run should succeed");
        let text = String::from_utf8(output).expect("output should be valid utf8");

        assert!(
            text.lines()
                .any(|line| line.starts_with("info depth") && line.contains(" pv ")),
            "expected at least one info line with a pv field, got: {text:?}"
        );
    }

    #[test]
    fn go_movetime_takes_priority_over_depth_when_both_given() {
        // depth 1 alone would report only one info line; movetime
        // should win and drive iterative deepening across several
        // depths within its budget instead.
        // No `quit`: see `SlowEofInput`'s docs.
        let input = std::io::BufReader::new(SlowEofInput::new(
            "position startpos\ngo depth 1 movetime 200\n",
            std::time::Duration::from_millis(300),
        ));
        let mut output = Vec::new();
        let mut engine = Engine::default();
        run(input, &mut output, &mut engine).expect("run should succeed");
        let text = String::from_utf8(output).expect("output should be valid utf8");

        let info_lines = text.lines().filter(|l| l.starts_with("info depth")).count();
        assert!(
            info_lines >= 2,
            "expected movetime to drive iterative deepening past depth 1, got {info_lines} info lines"
        );
    }

    // -- Cancellation/race tests, driven directly against
    // `run_event_loop` by sending `Event`s through a channel, per the
    // module docs on why this is the deterministic way to test races
    // that would otherwise depend on real thread timing (a `stop`
    // arriving mid-search, `quit` while searching, natural completion
    // racing a `stop`, ...). Each test owns the `Sender` half so
    // `events.recv()` blocks exactly as long as intended rather than
    // ever seeing a spuriously disconnected channel.

    fn count_bestmoves(output: &[u8]) -> usize {
        String::from_utf8_lossy(output)
            .lines()
            .filter(|line| line.starts_with("bestmove"))
            .count()
    }

    #[test]
    fn event_loop_emits_exactly_one_bestmove_on_natural_completion() {
        let (sender, events) = std::sync::mpsc::channel();
        let mut output = Vec::new();
        let mut engine = Engine::default();

        sender
            .send(Event::Line("position startpos".to_string()))
            .unwrap();
        sender.send(Event::Line("go depth 3".to_string())).unwrap();
        sender.send(Event::Line("quit".to_string())).unwrap();

        run_event_loop(&sender, &events, &mut output, &mut engine).expect("should succeed");

        assert_eq!(count_bestmoves(&output), 1);
    }

    #[test]
    fn event_loop_stop_mid_search_produces_exactly_one_bestmove() {
        let (sender, events) = std::sync::mpsc::channel();
        let mut output = Vec::new();
        let mut engine = Engine::default();

        sender
            .send(Event::Line("position startpos".to_string()))
            .unwrap();
        // A very long movetime -- this search would not finish on its
        // own within the test's lifetime without `stop`.
        sender
            .send(Event::Line("go movetime 600000".to_string()))
            .unwrap();
        sender.send(Event::Line("stop".to_string())).unwrap();
        sender.send(Event::Line("quit".to_string())).unwrap();

        run_event_loop(&sender, &events, &mut output, &mut engine).expect("should succeed");

        assert_eq!(
            count_bestmoves(&output),
            1,
            "stop should cancel the search and report exactly one bestmove, not hang for 600s"
        );
    }

    #[test]
    fn event_loop_quit_mid_search_cancels_without_hanging() {
        let (sender, events) = std::sync::mpsc::channel();
        let mut output = Vec::new();
        let mut engine = Engine::default();

        sender
            .send(Event::Line("position startpos".to_string()))
            .unwrap();
        sender
            .send(Event::Line("go movetime 600000".to_string()))
            .unwrap();
        sender.send(Event::Line("quit".to_string())).unwrap();

        // The real assertion here is that this returns at all (within
        // the test harness's own timeout) rather than blocking for
        // 600 seconds -- `quit` must cancel and join, not wait out the
        // search.
        run_event_loop(&sender, &events, &mut output, &mut engine).expect("should succeed");
    }

    #[test]
    fn event_loop_stop_racing_natural_completion_still_reports_one_bestmove() {
        // A `stop` sent (and, crucially, still processed) after the
        // search has *already* completed naturally: exercise the
        // `SearchEvent::Done` early-clear path (see `Event::Search`'s
        // handling) plus `Stop`'s own "nothing running" no-op path,
        // one after the other, rather than assuming they can't
        // coexist in the same run.
        let (sender, events) = std::sync::mpsc::channel();
        let mut output = Vec::new();
        let mut engine = Engine::default();

        sender
            .send(Event::Line("position startpos".to_string()))
            .unwrap();
        sender.send(Event::Line("go depth 2".to_string())).unwrap();
        // Give the (fast, depth-2) search a moment to genuinely finish
        // before `stop` is even read, so this exercises "stop with
        // nothing running" rather than "stop mid-search" (already
        // covered above).
        std::thread::sleep(std::time::Duration::from_millis(50));
        sender.send(Event::Line("stop".to_string())).unwrap();
        sender.send(Event::Line("quit".to_string())).unwrap();

        run_event_loop(&sender, &events, &mut output, &mut engine).expect("should succeed");

        assert_eq!(count_bestmoves(&output), 1);
    }

    #[test]
    fn event_loop_a_stopped_search_does_not_leak_cancellation_into_the_next_go() {
        // If `go`'s `StopSignal` weren't fresh per search (see
        // `SearchWorker::spawn`'s docs), a second `go` after a `stop`
        // could see a signal that's already been requested and abort
        // instantly instead of actually searching.
        let (sender, events) = std::sync::mpsc::channel();
        let mut output = Vec::new();
        let mut engine = Engine::default();

        sender
            .send(Event::Line("position startpos".to_string()))
            .unwrap();
        sender
            .send(Event::Line("go movetime 600000".to_string()))
            .unwrap();
        sender.send(Event::Line("stop".to_string())).unwrap();
        sender.send(Event::Line("go depth 3".to_string())).unwrap();
        sender.send(Event::Line("quit".to_string())).unwrap();

        run_event_loop(&sender, &events, &mut output, &mut engine).expect("should succeed");

        let text = String::from_utf8(output).unwrap();
        assert_eq!(
            text.lines().filter(|l| l.starts_with("bestmove")).count(),
            2,
            "both the stopped first search and the second search should each report a bestmove"
        );
        // The second `go depth 3` must have actually searched (not
        // instantly aborted by a leaked stop signal) -- a real depth-3
        // search reports real info lines.
        assert!(
            text.lines().any(|l| l.starts_with("info depth 3")),
            "the second go should have run a real, uncancelled search; got: {text:?}"
        );
    }

    #[test]
    fn event_loop_position_is_restored_after_a_cancelled_search() {
        let (sender, events) = std::sync::mpsc::channel();
        let mut output = Vec::new();
        let mut engine = Engine::default();
        let before = engine.position().clone();

        sender
            .send(Event::Line("go movetime 600000".to_string()))
            .unwrap();
        sender.send(Event::Line("stop".to_string())).unwrap();
        sender.send(Event::Line("quit".to_string())).unwrap();

        run_event_loop(&sender, &events, &mut output, &mut engine).expect("should succeed");

        assert_eq!(
            engine.position(),
            &before,
            "a stopped search must not leave the position mutated -- search always \
             restores it via make/unmake, including on early cancellation"
        );
    }

    #[test]
    fn event_loop_isready_while_a_search_is_active_finishes_the_search_first() {
        // `isready` (like any other command) needs exclusive access to
        // `Engine`, so it must join the outstanding search -- exactly
        // like `position`/`setoption`/another `go` would -- rather
        // than replying `readyok` while a worker still owns `Engine`.
        let (sender, events) = std::sync::mpsc::channel();
        let mut output = Vec::new();
        let mut engine = Engine::default();

        sender
            .send(Event::Line("position startpos".to_string()))
            .unwrap();
        sender
            .send(Event::Line("go movetime 600000".to_string()))
            .unwrap();
        sender.send(Event::Line("isready".to_string())).unwrap();
        sender.send(Event::Line("quit".to_string())).unwrap();

        run_event_loop(&sender, &events, &mut output, &mut engine).expect("should succeed");

        let text = String::from_utf8(output).unwrap();
        let bestmove_index = text.find("bestmove").expect("should have a bestmove");
        let readyok_index = text.find("readyok").expect("should have a readyok");
        assert!(
            bestmove_index < readyok_index,
            "the outstanding search's bestmove must be reported before readyok, \
             since isready had to join it first to get exclusive access to Engine; got: {text:?}"
        );
    }

    #[test]
    fn event_loop_a_new_position_command_joins_the_outstanding_search_first() {
        let (sender, events) = std::sync::mpsc::channel();
        let mut output = Vec::new();
        let mut engine = Engine::default();

        sender
            .send(Event::Line("position startpos".to_string()))
            .unwrap();
        sender
            .send(Event::Line("go movetime 600000".to_string()))
            .unwrap();
        sender
            .send(Event::Line("position startpos moves e2e4".to_string()))
            .unwrap();
        sender.send(Event::Line("quit".to_string())).unwrap();

        run_event_loop(&sender, &events, &mut output, &mut engine).expect("should succeed");

        assert_eq!(
            count_bestmoves(&output),
            1,
            "the first go's search must be joined and report its bestmove"
        );
        assert_eq!(
            engine.position().to_fen(),
            {
                let mut position = crate::chess::Position::startpos();
                let mv = position
                    .generate_legal_moves()
                    .into_iter()
                    .find(|mv| {
                        mv.from() == "e2".parse().unwrap() && mv.to() == "e4".parse().unwrap()
                    })
                    .unwrap();
                position.make_move(mv);
                position.to_fen()
            },
            "the new position command must still take effect after joining the old search"
        );
    }

    #[test]
    fn stop_with_nothing_running_is_a_silent_no_op() {
        let input = b"stop\nisready\nquit\n".as_slice();
        let mut output = Vec::new();
        let mut engine = Engine::default();
        run(input, &mut output, &mut engine).expect("run should succeed");
        let text = String::from_utf8(output).expect("output should be valid utf8");

        assert!(
            text.contains("readyok"),
            "stop must not disrupt the rest of the session"
        );
        assert!(
            !text.lines().any(|l| l.starts_with("bestmove")),
            "a stop with no search running must not conjure a bestmove out of nowhere"
        );
    }
}
