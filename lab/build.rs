//! Embeds the current git commit hash into the `bee-lab` binary at
//! compile time (`BEE_LAB_GIT_COMMIT`), so every experiment's
//! recorded metadata can say exactly which build of Lab/Bee produced
//! it -- see `experiment::ExperimentSpec`'s docs on reproducibility.
//! A build-time embed rather than shelling out to `git` at runtime
//! (e.g. on every `POST /api/experiments`) keeps experiment creation
//! itself independent of `git` being installed or this even being a
//! git checkout at all (a packaged/deployed binary, say) -- it just
//! falls back to `"unknown"` in that case, rather than failing the
//! request.

use std::process::Command;

fn main() {
    let commit = Command::new("git")
        .args(["rev-parse", "HEAD"])
        .output()
        .ok()
        .filter(|output| output.status.success())
        .and_then(|output| String::from_utf8(output.stdout).ok())
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| "unknown".to_string());

    println!("cargo:rustc-env=BEE_LAB_GIT_COMMIT={commit}");
    // Re-run only when HEAD actually moves (a new commit/checkout),
    // not on every build -- `.git/HEAD` changes on every commit/
    // checkout/branch switch, which is exactly what should invalidate
    // this.
    println!("cargo:rerun-if-changed=../.git/HEAD");
}
