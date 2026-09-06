//! Smoke test: the `bee` binary responds to a minimal UCI handshake.
//!
//! This exercises the actual compiled binary over stdin/stdout, so it
//! catches wiring bugs (binary name, entry point, process exit) that a
//! unit test on `uci::run` alone would not.
//!
//! This is also the one place that can actually catch a real bug
//! `uci::run`'s own in-memory-buffer tests structurally cannot: `run`
//! (used by those tests) spawns its reader thread *scoped*, so it's
//! always joined on return, which only works because a `&[u8]`/`String`
//! test buffer always reaches EOF promptly. The real `bee` binary uses
//! `run_stdio` specifically because real process stdin does not close
//! just because Bee decided to `quit` -- a regression back to the
//! scoped `run` in `bin/bee.rs` would hang exactly this test (and every
//! real GUI/Lichess bridge session) without ever showing up in
//! `uci::run`'s own test suite, since those tests all let their input
//! reach EOF one way or another.

use std::io::{BufRead, BufReader, Write};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

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
    assert!(stdout.contains("id name bee-chess"));
    assert!(stdout.contains("id author"));
    assert!(stdout.contains("uciok"));
    assert!(stdout.contains("readyok"));
}

/// `quit` must make the process exit promptly even if the caller never
/// closes its end of the stdin pipe -- exactly how a real GUI/Lichess
/// bridge behaves (it doesn't close stdin just because Bee decided to
/// quit). See this file's module docs for why `run_stdio`'s detached
/// reader thread (not `run`'s scoped one) is what makes this possible.
#[test]
fn quit_exits_promptly_even_with_stdin_left_open() {
    let mut child = Command::new(env!("CARGO_BIN_EXE_bee"))
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .expect("failed to spawn bee binary");

    // Deliberately keep `stdin` alive (not `.take()`'d and dropped)
    // for the rest of this test -- the whole point is that the pipe
    // stays open.
    let mut stdin = child.stdin.take().expect("stdin should be piped");
    stdin
        .write_all(b"uci\nisready\nquit\n")
        .expect("failed to write to stdin");

    let start = Instant::now();
    let status = child
        .wait_timeout_or_kill(Duration::from_secs(5))
        .expect("bee should exit promptly after quit, not hang waiting on stdin");
    assert!(status.success());
    assert!(
        start.elapsed() < Duration::from_secs(5),
        "quit should exit almost immediately, not need the full timeout"
    );
}

/// `stop` must cancel an in-progress, otherwise-very-long search and
/// still report a `bestmove` promptly -- and the process must still be
/// able to `quit` cleanly afterward, all with stdin kept open the
/// entire time (see the module docs on why this is the one place that
/// can actually catch a regression here).
#[test]
fn stop_cancels_a_long_search_and_quit_still_exits_promptly() {
    let mut child = Command::new(env!("CARGO_BIN_EXE_bee"))
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .expect("failed to spawn bee binary");

    let mut stdin = child.stdin.take().expect("stdin should be piped");
    let stdout = child.stdout.take().expect("stdout should be piped");
    let mut reader = BufReader::new(stdout);

    stdin
        .write_all(b"uci\nisready\nposition startpos\ngo movetime 600000\n")
        .expect("failed to write to stdin");

    // Give the search a moment to actually start before stopping it.
    std::thread::sleep(Duration::from_millis(200));
    let stop_sent_at = Instant::now();
    stdin.write_all(b"stop\n").expect("failed to send stop");

    let mut saw_bestmove = false;
    let deadline = Instant::now() + Duration::from_secs(5);
    let mut line = String::new();
    while Instant::now() < deadline {
        line.clear();
        let bytes_read = reader.read_line(&mut line).expect("failed to read stdout");
        if bytes_read == 0 {
            break; // process closed stdout unexpectedly
        }
        if line.starts_with("bestmove") {
            saw_bestmove = true;
            break;
        }
    }

    assert!(saw_bestmove, "stop should produce a bestmove line");
    assert!(
        stop_sent_at.elapsed() < Duration::from_secs(2),
        "stop should cancel the 600s search almost immediately, took {:?}",
        stop_sent_at.elapsed()
    );

    stdin.write_all(b"quit\n").expect("failed to send quit");
    let status = child
        .wait_timeout_or_kill(Duration::from_secs(5))
        .expect("bee should exit promptly after quit, not hang waiting on stdin");
    assert!(status.success());
}

/// Test-only convenience: `Child::wait` blocks forever if the process
/// never exits, which would hang the whole test suite on a real
/// regression instead of failing it -- this polls with a timeout and
/// kills the child if it's still alive once that elapses, so a hang
/// becomes a normal (if slow) test failure.
trait WaitTimeoutOrKill {
    fn wait_timeout_or_kill(
        &mut self,
        timeout: Duration,
    ) -> std::io::Result<std::process::ExitStatus>;
}

impl WaitTimeoutOrKill for std::process::Child {
    fn wait_timeout_or_kill(
        &mut self,
        timeout: Duration,
    ) -> std::io::Result<std::process::ExitStatus> {
        let deadline = Instant::now() + timeout;
        loop {
            if let Some(status) = self.try_wait()? {
                return Ok(status);
            }
            if Instant::now() >= deadline {
                let _ = self.kill();
                let _ = self.wait();
                return Err(std::io::Error::new(
                    std::io::ErrorKind::TimedOut,
                    "process did not exit before the timeout",
                ));
            }
            std::thread::sleep(Duration::from_millis(20));
        }
    }
}
