//! Bee Chess engine binary entry point.
//!
//! Runs the UCI loop against stdin/stdout. See `bee_engine::uci`.

use std::io::{stdin, stdout};

fn main() -> std::io::Result<()> {
    let stdin = stdin();
    let stdout = stdout();
    bee_engine::uci::run(stdin.lock(), stdout.lock())
}
