//! Smoke test: the `bee` binary responds to a minimal UCI handshake.
//!
//! This exercises the actual compiled binary over stdin/stdout, so it
//! catches wiring bugs (binary name, entry point, process exit) that a
//! unit test on `uci::run` alone would not.

use std::io::Write;
use std::process::{Command, Stdio};

#[test]
fn uci_handshake_over_stdio() {
    let mut child = Command::new(env!("CARGO_BIN_EXE_bee"))
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .expect("failed to spawn bee binary");

    child
        .stdin
        .take()
        .expect("stdin should be piped")
        .write_all(b"uci\nisready\nquit\n")
        .expect("failed to write to stdin");

    let output = child
        .wait_with_output()
        .expect("failed to wait on bee process");

    assert!(output.status.success());

    let stdout = String::from_utf8(output.stdout).expect("stdout should be valid utf8");
    assert!(stdout.contains("id name Bee"));
    assert!(stdout.contains("id author"));
    assert!(stdout.contains("uciok"));
    assert!(stdout.contains("readyok"));
}
