//! The `.json` manifest written alongside a `.book` artifact (e.g.
//! `books/experience-v1.book` + `books/experience-v1.json`): answers
//! "what exactly did we bake into this Bee?" without needing to parse
//! the binary artifact itself. Not consumed by any runtime
//! `ExperienceBook` -- purely for reproducibility/debugging, per the
//! design this followed.

use sha2::{Digest, Sha256};

use bee_book_format::{FORMAT_VERSION, KEY_SCHEME_VERSION};

use super::builder::{BuildConfig, BuildReport};

/// One build run's manifest, serialized as small hand-rolled JSON (no
/// `serde_json` dependency here -- see `to_json`) -- this is a handful
/// of scalar fields, not worth pulling in a JSON library for.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Manifest {
    pub format_version: u16,
    pub book_key_version: u16,
    /// Every account identity pooled into this book -- see
    /// `book::build`'s docs on treating multiple names (e.g. Bee's
    /// games played under more than one Lichess account) as one
    /// learning identity. A single-account book still stores a
    /// one-element list, so a manifest reader never needs to special-
    /// case "how many accounts".
    pub source_accounts: Vec<String>,
    pub games_considered: u64,
    /// Games actually usable for learning: `games_considered` minus
    /// those skipped because their move list couldn't be fully
    /// SAN-resolved (see `book::builder::record_game`'s docs).
    pub usable_games: u64,
    pub positions: u64,
    pub max_ply: u32,
    pub min_games: u32,
    pub prior_games: u32,
    pub prior_score_per_mille: u32,
    /// The `bee-chess` commit this artifact was built from, if known --
    /// `None` when built from a working tree `git` can't identify (a
    /// shallow/detached clone, or no `git` available at all), so a
    /// manifest never lies about provenance it couldn't determine.
    pub builder_commit: Option<String>,
    pub sha256: String,
}

impl Manifest {
    #[must_use]
    pub fn new(
        source_accounts: &[&str],
        config: &BuildConfig,
        report: &BuildReport,
        book_bytes: &[u8],
        builder_commit: Option<String>,
    ) -> Self {
        Self {
            format_version: FORMAT_VERSION,
            book_key_version: KEY_SCHEME_VERSION,
            source_accounts: source_accounts.iter().map(|p| p.to_string()).collect(),
            // `report.games_considered` (the builder's own field) counts
            // only *successfully replayed* games; the manifest's
            // `games_considered` means "every game fetched as a
            // candidate" (usable or not), so it adds back the ones
            // skipped as unresolvable -- see `usable_games` for the
            // successfully-replayed count alone.
            games_considered: report.games_considered + report.games_skipped_unresolvable,
            usable_games: report.games_considered,
            positions: report.positions,
            max_ply: config.max_ply,
            min_games: config.min_games,
            prior_games: config.prior_games,
            prior_score_per_mille: config.prior_score_per_mille,
            builder_commit,
            sha256: sha256_hex(book_bytes),
        }
    }

    /// Renders as pretty-printed JSON. Field order matches struct
    /// declaration order, deterministically, on every call.
    #[must_use]
    pub fn to_json(&self) -> String {
        let source_accounts = self
            .source_accounts
            .iter()
            .map(|p| json_string(p))
            .collect::<Vec<_>>()
            .join(", ");
        let builder_commit = match &self.builder_commit {
            Some(commit) => json_string(commit),
            None => "null".to_string(),
        };
        format!(
            "{{\n  \"format_version\": {},\n  \"book_key_version\": {},\n  \"source_accounts\": [{}],\n  \"games_considered\": {},\n  \"usable_games\": {},\n  \"positions\": {},\n  \"max_ply\": {},\n  \"min_games\": {},\n  \"prior_games\": {},\n  \"prior_score_per_mille\": {},\n  \"builder_commit\": {},\n  \"sha256\": {}\n}}\n",
            self.format_version,
            self.book_key_version,
            source_accounts,
            self.games_considered,
            self.usable_games,
            self.positions,
            self.max_ply,
            self.min_games,
            self.prior_games,
            self.prior_score_per_mille,
            builder_commit,
            json_string(&self.sha256),
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
            games_considered: 41,
            games_skipped_unresolvable: 1,
            positions: 7,
        };
        let manifest = Manifest::new(
            &["Bee\"Account"],
            &config,
            &report,
            b"hello",
            Some("abc1234".to_string()),
        );
        let json = manifest.to_json();

        assert!(json.contains("\"format_version\": 1"));
        // 41 usable + 1 skipped = 42 total fetched as candidates.
        assert!(json.contains("\"games_considered\": 42"));
        assert!(json.contains("\"usable_games\": 41"));
        assert!(json.contains("\"source_accounts\": [\"Bee\\\"Account\"]"));
        assert!(json.contains("\"builder_commit\": \"abc1234\""));
        // sha256("hello")
        assert!(json.contains(
            "\"sha256\": \"2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824\""
        ));
    }

    #[test]
    fn multiple_accounts_render_as_a_json_array() {
        let config = BuildConfig::default();
        let report = super::super::builder::BuildReport {
            games_considered: 99,
            games_skipped_unresolvable: 0,
            positions: 12,
        };
        let manifest = Manifest::new(
            &["beechessjohan", "beechessmagnus"],
            &config,
            &report,
            b"x",
            None,
        );
        assert!(manifest
            .to_json()
            .contains("\"source_accounts\": [\"beechessjohan\", \"beechessmagnus\"]"));
    }

    #[test]
    fn an_unknown_builder_commit_renders_as_json_null() {
        let config = BuildConfig::default();
        let report = super::super::builder::BuildReport {
            games_considered: 1,
            games_skipped_unresolvable: 0,
            positions: 1,
        };
        let manifest = Manifest::new(&["Bee"], &config, &report, b"x", None);
        assert!(manifest.to_json().contains("\"builder_commit\": null"));
    }

    #[test]
    fn same_inputs_produce_identical_json() {
        let config = BuildConfig::default();
        let report = super::super::builder::BuildReport {
            games_considered: 1,
            games_skipped_unresolvable: 0,
            positions: 1,
        };
        let a = Manifest::new(
            &["Bee"],
            &config,
            &report,
            b"x",
            Some("deadbee".to_string()),
        )
        .to_json();
        let b = Manifest::new(
            &["Bee"],
            &config,
            &report,
            b"x",
            Some("deadbee".to_string()),
        )
        .to_json();
        assert_eq!(a, b);
    }
}
