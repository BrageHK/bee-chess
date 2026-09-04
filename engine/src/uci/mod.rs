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

use crate::engine::Engine;

pub const ENGINE_NAME: &str = "bee-chess";
pub const ENGINE_AUTHOR: &str = "bragehk, johsol and sebasabe";

/// A parsed UCI command. Only the handshake subset is implemented in the
/// bootstrap PR; unrecognized input is ignored rather than erroring, per
/// the UCI convention of tolerating unknown commands.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum UciCommand {
    Uci,
    IsReady,
    Debug(bool),
    Quit,
    Unknown(String),
}

impl UciCommand {
    pub fn parse(line: &str) -> Self {
        let line = line.trim();
        match line {
            "uci" => UciCommand::Uci,
            "isready" => UciCommand::IsReady,
            "quit" => UciCommand::Quit,
            _ => match line.split_whitespace().collect::<Vec<_>>().as_slice() {
                ["debug", "on"] => UciCommand::Debug(true),
                ["debug", "off"] => UciCommand::Debug(false),
                _ => UciCommand::Unknown(line.to_string()),
            },
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
        assert_eq!(UciCommand::parse("quit"), UciCommand::Quit);
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
}
