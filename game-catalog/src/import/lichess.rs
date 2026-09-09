//! Streaming Lichess game import into a [`GameCatalog`].
//!
//! Uses `litchee`'s `games().export_user(..).stream()`, which is NDJSON
//! under the hood (see litchee's `stream` module) -- games are upserted one
//! at a time as they arrive rather than buffering the whole export in
//! memory first, so this scales to a user's entire history. `pgn-reader`
//! (a dedicated streaming PGN parser) turned out not to be needed for this
//! slice: Lichess's export already hands back structured fields (players,
//! ratings, result, opening, moves, and the raw PGN on request) as JSON, so
//! there is no PGN text to parse ourselves.

use futures_util::StreamExt;
use litchee::api::gameplay::games::{
    GameSort, LichessGame, LichessGamePlayer, LichessGameStatusName,
};
use litchee::model::{GameExportOptions, LichessColor};
use litchee::LichessClient;

use crate::catalog::GameCatalog;
use crate::error::Result;
use crate::game::GameRecord;

/// The `source` value stored on every game imported from Lichess.
pub const SOURCE: &str = "lichess";

/// Imports `username`'s Lichess games into `catalog`.
///
/// Incremental: reads [`GameCatalog::latest_imported_at`] for
/// `("lichess", username)` and, if set, only requests games played since
/// then (`sort(GameSort::DateAsc)` so the stream itself is oldest-first,
/// matching `since` and letting the watermark be updated as we go rather
/// than only at the end). A first sync has no watermark and fetches
/// everything.
///
/// Returns the number of games imported (upserted; already-seen games that
/// come back unchanged still count, since a re-run is expected to be
/// idempotent, not a no-op).
pub async fn sync_user(
    catalog: &GameCatalog,
    client: &LichessClient,
    username: &str,
) -> Result<u64> {
    let since = catalog.latest_imported_at(SOURCE, username)?;

    let mut request = client
        .games()
        .export_user(username)
        .sort(GameSort::DateAsc)
        .export(
            GameExportOptions::default()
                .moves(true)
                .pgn_in_json(true)
                .opening(true)
                .tags(true),
        );
    if let Some(since) = since {
        // `since` is inclusive on Lichess's side; the already-imported game
        // at exactly this timestamp is harmless to re-fetch since upsert is
        // idempotent, and it's simpler than reasoning about a +1ms fencepost
        // against a source that reports millisecond timestamps we don't
        // fully control the granularity of.
        request = request.since(since);
    }

    let mut games = request.stream().await?;
    let mut imported = 0u64;
    let mut watermark = since;

    while let Some(game) = games.next().await {
        let game = game?;
        let played_at = game.created_at;
        catalog.upsert_game(&to_record(game))?;
        imported += 1;
        if let Some(played_at) = played_at {
            watermark = Some(watermark.map_or(played_at, |w| w.max(played_at)));
        }
    }

    if let Some(watermark) = watermark {
        catalog.record_synced(SOURCE, username, watermark)?;
    }

    Ok(imported)
}

fn to_record(game: LichessGame) -> GameRecord {
    let players = game.players;
    let white = players.as_ref().and_then(|p| player_name(&p.white));
    let black = players.as_ref().and_then(|p| player_name(&p.black));
    let white_rating = players.as_ref().and_then(|p| p.white.rating).map(i64::from);
    let black_rating = players.as_ref().and_then(|p| p.black.rating).map(i64::from);

    GameRecord {
        id: game.id,
        source: SOURCE.to_string(),
        played_at: game.created_at,
        white,
        black,
        white_rating,
        black_rating,
        result: result_string(game.status, game.winner),
        termination: game.status.map(|s| format!("{s:?}").to_lowercase()),
        time_control: game.clock.map(|c| format!("{}+{}", c.initial, c.increment)),
        rated: game.rated,
        variant: game.variant.map(|v| format!("{v:?}").to_lowercase()),
        moves: game.moves,
        raw_pgn: game.pgn,
        imported_at: now_millis(),
    }
}

fn player_name(player: &LichessGamePlayer) -> Option<String> {
    player
        .user
        .as_ref()
        .map(|u| u.name.clone())
        .or_else(|| player.name.clone())
}

/// A PGN-style result string derived from Lichess's `status`/`winner`
/// fields, since the export doesn't hand back "1-0"/"0-1"/"1/2-1/2"
/// directly. An unfinished or unrecognized status yields `"*"`, matching
/// PGN's convention for "no result yet".
fn result_string(
    status: Option<LichessGameStatusName>,
    winner: Option<LichessColor>,
) -> Option<String> {
    let status = status?;
    let drawn = matches!(
        status,
        LichessGameStatusName::Draw
            | LichessGameStatusName::Stalemate
            | LichessGameStatusName::InsufficientMaterialClaim
    );
    Some(match (winner, drawn) {
        (Some(LichessColor::White), _) => "1-0".to_string(),
        (Some(LichessColor::Black), _) => "0-1".to_string(),
        (None, true) => "1/2-1/2".to_string(),
        (None, false) => "*".to_string(),
    })
}

fn now_millis() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .expect("system clock is after the Unix epoch")
        .as_millis() as i64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn result_string_reads_winner_and_status() {
        assert_eq!(
            result_string(Some(LichessGameStatusName::Mate), Some(LichessColor::White)),
            Some("1-0".to_string())
        );
        assert_eq!(
            result_string(
                Some(LichessGameStatusName::Resign),
                Some(LichessColor::Black)
            ),
            Some("0-1".to_string())
        );
        assert_eq!(
            result_string(Some(LichessGameStatusName::Draw), None),
            Some("1/2-1/2".to_string())
        );
        assert_eq!(
            result_string(Some(LichessGameStatusName::Started), None),
            Some("*".to_string())
        );
        assert_eq!(result_string(None, None), None);
    }
}
