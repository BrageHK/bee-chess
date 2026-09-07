//! Bee Chess engine binary entry point.
//!
//! Runs the UCI loop against stdin/stdout. See `bee_engine::uci`.

use std::io::{stdin, stdout, BufReader};

fn main() -> std::io::Result<()> {
    // `BufReader::new(stdin())` rather than `stdin.lock()`:
    // `run_stdio` reads its input on a detached background thread
    // (see its own docs), which needs `R: Send + 'static` -- `Stdin`
    // is `Send + 'static`, but `StdinLock` is neither `'static` nor
    // `Send` (it holds a `MutexGuard`), so locking stdin up front here
    // isn't an option the way it used to be. Wrapping the unlocked
    // `Stdin` in a `BufReader` gives the same buffered-line-reading
    // behavior `StdinLock` already provided, on a type that can
    // actually move to (and outlive) another thread.
    //
    // `run_stdio`, not `run`: real stdin never closes just because Bee
    // decides to `quit`, so the reader thread needs to be detachable
    // rather than joined -- see `run_stdio`'s own docs for why `run`
    // itself would hang here.
    let stdin = BufReader::new(stdin());
    let stdout = stdout();
    let mut engine = bee_engine::engine::Engine::default();
    bee_engine::uci::run_stdio(stdin, stdout.lock(), &mut engine)
}
