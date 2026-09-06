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
/// erroring, per the UCI convention of tolerating unknown commands. The
/// full asynchronous state machine (`setoption`, `stop`, `ponderhit`,
/// concurrent input handling while searching, and actual search) lands
/// in a follow-up milestone.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum UciCommand {
    Uci,
    IsReady,
    Debug(bool),
    SetOption { name: String, value: String },
    NewGame,
    Position(PositionCommand),
    Go(GoCommand),
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

/// Runs the UCI loop, reading commands from `input` and writing responses
/// to `output`, until `quit` is received or input ends.
///
/// Diagnostics: engine/search code never writes UCI text directly (see
/// `crate::diagnostics`); this is the one place that turns whatever
/// `Engine::emit_diagnostic` accumulated into `info string ...` lines,
/// and it only does so while `debug on` is in effect. With debug off,
/// diagnostics are still drained (so they never pile up unboundedly)
/// but simply discarded rather than written -- unknown commands and
/// other diagnostics stay silent, per normal UCI behavior.
pub fn run<R: BufRead, W: Write>(
    input: R,
    mut output: W,
    engine: &mut Engine,
) -> std::io::Result<()> {
    for line in input.lines() {
        let line = line?;

        if engine.debug() {
            engine.emit_diagnostic(DiagnosticLevel::Debug, format!("received: {line}"));
        }

        match UciCommand::parse(&line) {
            UciCommand::Uci => {
                writeln!(output, "id name {ENGINE_NAME}")?;
                writeln!(output, "id author {ENGINE_AUTHOR}")?;
                writeln!(output, "option name Evaluator type combo default Positional var Positional var Material")?;
                // Experimental search feature switches -- see
                // `SearchOptions`'s docs. Both default to `true` (the
                // normal, strongest configuration); Bee Lab's A/B
                // experiment runner is the intended way to turn one off,
                // not a permanent engine configuration.
                writeln!(output, "option name UseTT type check default true")?;
                writeln!(output, "option name UseQuiescence type check default true")?;
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
                        Ok(ms) => engine.set_move_overhead(std::time::Duration::from_millis(ms)),
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
                // No real cancellation/threading yet (see GoCommand's
                // docs and #6/#7) -- `go` runs to completion
                // synchronously before this loop reads its next line.
                // Priority when more than one applies: movetime, then
                // the real UCI clock (wtime/btime/...), then depth --
                // see `GoCommand`'s docs for why.
                let side_to_move = engine.position().side_to_move();
                let result = if let Some(movetime_ms) = go_command.movetime_ms {
                    let budget = std::time::Duration::from_millis(movetime_ms);
                    let start = Instant::now();
                    engine.search_for_time(budget, |depth_result| {
                        let _ = write_search_info(&mut output, depth_result, start.elapsed());
                    })
                } else if let Some(control) = go_command.clock_for(side_to_move) {
                    let start = Instant::now();
                    engine.search_with_clock(control, |depth_result| {
                        let _ = write_search_info(&mut output, depth_result, start.elapsed());
                    })
                } else {
                    // Default to a shallow depth when none of
                    // movetime/wtime/btime is given, since there's no
                    // time-based stopping condition to fall back on
                    // instead.
                    const DEFAULT_DEPTH: u32 = 4;
                    let depth = go_command.depth.unwrap_or(DEFAULT_DEPTH);
                    let start = Instant::now();
                    let result = engine.search(depth);
                    // depth == 0 means this was a book hit (see
                    // `Engine::book_move`'s docs), not a real
                    // search -- there's no depth/node count to
                    // report, so writing an "info depth 0 ..."
                    // line for it would misrepresent it as one.
                    if result.depth > 0 {
                        write_search_info(&mut output, &result, start.elapsed())?;
                    }
                    result
                };

                match result.best_move {
                    Some(mv) => writeln!(output, "bestmove {}", format_uci_move(mv))?,
                    // No legal moves (checkmate/stalemate): UCI's
                    // convention for "no move to make" is bestmove
                    // 0000 rather than omitting the response.
                    None => writeln!(output, "bestmove 0000")?,
                }
            }
            UciCommand::Quit => {
                break;
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
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

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
        assert!(text.contains("option name UseTT type check default true"));
        assert!(text.contains("option name UseQuiescence type check default true"));
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
        let input = b"position startpos\ngo movetime 200\nquit\n".as_slice();
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
        // depths, then a bestmove.
        let input = b"position startpos\ngo wtime 5000 btime 5000\nquit\n".as_slice();
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
        let input = format!("position fen {fen}\ngo movetime 500\nquit\n");
        let mut output = Vec::new();
        let mut engine = Engine::default();
        run(input.as_bytes(), &mut output, &mut engine).expect("run should succeed");
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
        let input = b"position startpos\ngo movetime 100\nquit\n".as_slice();
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
        let input = b"position startpos\ngo depth 1 movetime 200\nquit\n".as_slice();
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
}
