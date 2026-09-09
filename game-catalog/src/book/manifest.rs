//! The `.book.json` manifest written alongside a `.book` artifact:
//! answers "what exactly did we bake into this Bee?" without needing to
//! parse the binary artifact itself. Not consumed by any runtime
//! `ExperienceBook` -- purely for reproducibility/debugging, per the
//! design this followed.

use sha2::{Digest, Sha256};

use super::builder::{BuildConfig, BuildReport};
use super::format::FORMAT_VERSION;
use super::key::KEY_SCHEME_VERSION;

/// One build run's manifest, serialized as small hand-rolled JSON (no
/// `serde_json` dependency here -- see `to_json`) -- this is a handful
/// of scalar fields, not worth pulling in a JSON library for.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Manifest {
    pub format_version: u16,
    pub key_scheme_version: u16,
    pub player: String,
    pub games_considered: u64,
    pub games_skipped_unresolvable: u64,
    pub positions: u64,
    pub max_ply: u32,
    pub min_games: u32,
    pub prior_games: u32,
    pub prior_score_per_mille: u32,
    pub book_sha256: String,
}

impl Manifest {
    #[must_use]
    pub fn new(
        player: &str,
        config: &BuildConfig,
        report: &BuildReport,
        book_bytes: &[u8],
    ) -> Self {
        Self {
            format_version: FORMAT_VERSION,
            key_scheme_version: KEY_SCHEME_VERSION,
            player: player.to_string(),
            games_considered: report.games_considered,
            games_skipped_unresolvable: report.games_skipped_unresolvable,
            positions: report.positions,
            max_ply: config.max_ply,
            min_games: config.min_games,
            prior_games: config.prior_games,
            prior_score_per_mille: config.prior_score_per_mille,
            book_sha256: sha256_hex(book_bytes),
        }
    }

    /// Renders as pretty-printed JSON. Field order matches struct
    /// declaration order, deterministically, on every call.
    #[must_use]
    pub fn to_json(&self) -> String {
        format!(
            "{{\n  \"format_version\": {},\n  \"key_scheme_version\": {},\n  \"player\": {},\n  \"games_considered\": {},\n  \"games_skipped_unresolvable\": {},\n  \"positions\": {},\n  \"max_ply\": {},\n  \"min_games\": {},\n  \"prior_games\": {},\n  \"prior_score_per_mille\": {},\n  \"book_sha256\": {}\n}}\n",
            self.format_version,
            self.key_scheme_version,
            json_string(&self.player),
            self.games_considered,
            self.games_skipped_unresolvable,
            self.positions,
            self.max_ply,
            self.min_games,
            self.prior_games,
            self.prior_score_per_mille,
            json_string(&self.book_sha256),
        )
    }
}

/// Escapes `s` as a JSON string literal, including the surrounding
/// quotes. Only handles the escapes a player name or hex digest could
/// plausibly need (`"`, `\`, control characters) -- not a general JSON
/// encoder, since this manifest has no nested structures or arrays.
fn json_string(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for ch in s.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            c if c.is_control() => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

fn sha256_hex(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    digest.iter().map(|b| format!("{b:02x}")).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn json_round_trips_visually_sane_output() {
        let config = BuildConfig::default();
        let report = super::super::builder::BuildReport {
            games_considered: 42,
            games_skipped_unresolvable: 1,
            positions: 7,
        };
        let manifest = Manifest::new("Bee\"Account", &config, &report, b"hello");
        let json = manifest.to_json();

        assert!(json.contains("\"format_version\": 1"));
        assert!(json.contains("\"games_considered\": 42"));
        assert!(json.contains("\"player\": \"Bee\\\"Account\""));
        // sha256("hello")
        assert!(json.contains(
            "\"book_sha256\": \"2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824\""
        ));
    }

    #[test]
    fn same_inputs_produce_identical_json() {
        let config = BuildConfig::default();
        let report = super::super::builder::BuildReport {
            games_considered: 1,
            games_skipped_unresolvable: 0,
            positions: 1,
        };
        let a = Manifest::new("Bee", &config, &report, b"x").to_json();
        let b = Manifest::new("Bee", &config, &report, b"x").to_json();
        assert_eq!(a, b);
    }
}
