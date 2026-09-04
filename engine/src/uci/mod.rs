//! UCI process boundary.
//!
//! This is the only module allowed to know about raw UCI text. It parses
//! stdin lines into typed commands and writes typed responses back to
//! stdout. Per ADR 0001, no UCI strings may leak below this module.
//!
//! The bootstrap PR implements only the minimal handshake (`uci`,
//! `isready`, `quit`) needed for a real, talkable-to engine process. The
//! full asynchronous state machine (`setoption`, `ucinewgame`, `position`,
//! `go`, `stop`, `ponderhit`, concurrent input handling while searching)
//! lands in a follow-up PR (`feat/uci-state-machine`).

use std::io::{BufRead, Write};

use crate::chess::{PieceKind, Position, Square};
use crate::engine::Engine;

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

/// A parsed UCI command. Only a subset of the full protocol is
/// implemented so far; unrecognized input is ignored rather than
/// erroring, per the UCI convention of tolerating unknown commands. The
/// full asynchronous state machine (`setoption`, `go`, `stop`,
/// `ponderhit`, concurrent input handling while searching) lands in a
/// follow-up PR.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum UciCommand {
    Uci,
    IsReady,
    Debug(bool),
    NewGame,
    Position(PositionCommand),
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
                match line.split_whitespace().collect::<Vec<_>>().as_slice() {
                    ["debug", "on"] => UciCommand::Debug(true),
                    ["debug", "off"] => UciCommand::Debug(false),
                    _ => UciCommand::Unknown(line.to_string()),
                }
            }
        }
    }
}

/// Runs the UCI loop, reading commands from `input` and writing responses
/// to `output`, until `quit` is received or input ends.
pub fn run<R: BufRead, W: Write>(
    input: R,
    mut output: W,
    engine: &mut Engine,
) -> std::io::Result<()> {
    for line in input.lines() {
        let line = line?;

        if engine.debug() {
            writeln!(output, "info string received: {line}")?;
        }

        match UciCommand::parse(&line) {
            UciCommand::Uci => {
                writeln!(output, "id name {ENGINE_NAME}")?;
                writeln!(output, "id author {ENGINE_AUTHOR}")?;
                writeln!(output, "uciok")?;
            }
            UciCommand::IsReady => {
                writeln!(output, "readyok")?;
            }
            UciCommand::Debug(on) => {
                engine.set_debug(on);
            }
            UciCommand::NewGame => {
                engine.new_game();
            }
            UciCommand::Position(command) => {
                if let Err(error) = command.resolve(engine) {
                    if engine.debug() {
                        writeln!(output, "info string {error:?}")?;
                    }
                }
            }
            UciCommand::Quit => {
                break;
            }
            UciCommand::Unknown(_) => {
                // Unrecognized commands are ignored, per UCI convention.
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
    fn uci_handshake_produces_expected_output() {
        let input = b"uci\nisready\nquit\n".as_slice();
        let mut output = Vec::new();
        let mut engine = Engine::default();
        run(input, &mut output, &mut engine).expect("run should succeed");
        let text = String::from_utf8(output).expect("output should be valid utf8");
        assert!(text.contains(&format!("id name {ENGINE_NAME}")));
        assert!(text.contains(&format!("id author {ENGINE_AUTHOR}")));
        assert!(text.contains("uciok"));
        assert!(text.contains("readyok"));
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
}
