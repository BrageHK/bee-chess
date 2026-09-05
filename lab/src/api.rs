//! HTTP surface for authoritative game state (#69 / 67b): `POST
//! /api/games` creates a game (optionally engine-vs-engine/engine-vs-
//! human -- see `CreateGameRequest`), `GET /api/games/:id` returns its
//! complete current snapshot, `POST /api/games/:id/moves` applies a
//! move to it.
//!
//! `POST /api/games/:id/moves` stays useful even for an engine-vs-
//! engine game's human-free case turned off, and especially for any
//! human-involving game: a human move still needs to reach the server
//! somehow, and this is that path regardless of whether the *other*
//! side is engine-driven (69b's automatic loop, see `game::
//! run_engine_loop`) or another human.
//!
//! `GET /ws/games/:id` (WebSocket) streams `game::GameEvent`s as JSON
//! -- live UCI traffic per side, and the new snapshot whenever a move
//! is applied or the game reaches a terminal status. Transient
//! telemetry only: `GET /api/games/:id` stays the authoritative resync
//! mechanism (see `game`'s module docs), so a client reconnecting after
//! a dropped WebSocket only needs to re-fetch the snapshot, never to
//! replay missed events.

use std::collections::HashMap;

use axum::extract::ws::{Message, WebSocket, WebSocketUpgrade};
use axum::extract::{Json, Path, State};
use axum::http::StatusCode;
use axum::response::IntoResponse;
use axum::routing::{get, post};
use axum::Router;
use serde::{Deserialize, Serialize};
use tokio::sync::broadcast;

use crate::game::{
    ApplyMoveError, EngineConfig, EngineSlots, EngineSpec, GameEvent, GameSnapshot, GameStore,
    ParticipantInfo,
};
use crate::uci_process::UciDirection;

/// The engine binaries this server knows how to spawn for a game,
/// keyed by the name a `CreateGameRequest` names them with (e.g.
/// `"stockfish"`, `"bee"`). This is a stopgap for #70 (67c)'s real
/// engine/model registry -- see that issue for the actual descriptor-
/// based design (ids, options, model references) this should become;
/// for now it's just enough to let 69b's automatic loop pick an engine
/// by name over the API instead of every caller needing to know a
/// binary path.
#[derive(Clone, Default)]
pub struct EngineRegistry(HashMap<String, EngineSpec>);

impl EngineRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn insert(&mut self, name: impl Into<String>, spec: EngineSpec) -> &mut Self {
        self.0.insert(name.into(), spec);
        self
    }

    fn get(&self, name: &str) -> Option<EngineSpec> {
        self.0.get(name).cloned()
    }
}

#[derive(Clone)]
struct ApiState {
    store: GameStore,
    engines: EngineRegistry,
}

pub fn router(store: GameStore, engines: EngineRegistry) -> Router {
    Router::new()
        .route("/api/games", post(create_game))
        .route("/api/games/{id}", get(get_game))
        .route("/api/games/{id}/moves", post(apply_move))
        .route("/ws/games/{id}", get(game_events_ws))
        .with_state(ApiState { store, engines })
}

/// Default per-move time budget for an engine-driven side, when
/// `CreateGameRequest` doesn't specify one. Generous enough to see a
/// real search happen; nowhere near a real game clock (there is no
/// clock at all yet -- see #69's still-open clocks scope).
const DEFAULT_MOVE_TIME_MS: u64 = 200;

/// One side's requested participant: either a bare engine name
/// (`"stockfish"`) for its defaults, or an object naming the engine
/// plus `setoption`s/debug -- e.g. `{"engine": "stockfish", "options":
/// {"UCI_LimitStrength": true, "UCI_Elo": 1600}}`. `#[serde(untagged)]`
/// so both shapes parse from the same field without the client needing
/// a discriminator for the common case.
#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum ParticipantRequest {
    EngineName(String),
    Engine {
        engine: String,
        #[serde(default)]
        options: HashMap<String, serde_json::Value>,
        #[serde(default)]
        debug: bool,
    },
}

impl ParticipantRequest {
    fn engine_name(&self) -> &str {
        match self {
            ParticipantRequest::EngineName(name) => name,
            ParticipantRequest::Engine { engine, .. } => engine,
        }
    }

    /// `(name, value)` pairs for `EngineConfig::options`. UCI option
    /// values are always sent as plain text regardless of their JSON
    /// type (`UCI_Elo: 1600` and `UCI_LimitStrength: true` both become
    /// `"1600"`/`"true"`) -- that's what `setoption name X value Y`
    /// expects on the wire either way; a JSON string in the request is
    /// passed through as-is rather than re-quoted.
    fn options(&self) -> Vec<(String, String)> {
        let ParticipantRequest::Engine { options, .. } = self else {
            return Vec::new();
        };
        options
            .iter()
            .map(|(name, value)| {
                let value = match value {
                    serde_json::Value::String(s) => s.clone(),
                    other => other.to_string(),
                };
                (name.clone(), value)
            })
            .collect()
    }

    fn debug(&self) -> bool {
        matches!(self, ParticipantRequest::Engine { debug: true, .. })
    }
}

#[derive(Debug, Deserialize, Default)]
struct CreateGameRequest {
    /// Participant driving White automatically, or omitted/`null` for
    /// a human-controlled White (moves arrive via
    /// `POST /api/games/:id/moves` instead).
    #[serde(default)]
    white: Option<ParticipantRequest>,
    #[serde(default)]
    black: Option<ParticipantRequest>,
    #[serde(default)]
    move_time_ms: Option<u64>,
}

async fn create_game(
    State(state): State<ApiState>,
    body: Option<Json<CreateGameRequest>>,
) -> impl IntoResponse {
    let request = body.map(|Json(r)| r).unwrap_or_default();

    let resolve =
        |participant: &Option<ParticipantRequest>| -> Result<Option<EngineConfig>, String> {
            let Some(participant) = participant else {
                return Ok(None);
            };
            let spec = state
                .engines
                .get(participant.engine_name())
                .ok_or_else(|| format!("unknown engine {:?}", participant.engine_name()))?;
            Ok(Some(EngineConfig {
                spec,
                options: participant.options(),
                debug: participant.debug(),
            }))
        };
    let white = match resolve(&request.white) {
        Ok(config) => config,
        Err(message) => {
            return (StatusCode::BAD_REQUEST, Json(ErrorBody::new(message))).into_response()
        }
    };
    let black = match resolve(&request.black) {
        Ok(config) => config,
        Err(message) => {
            return (StatusCode::BAD_REQUEST, Json(ErrorBody::new(message))).into_response()
        }
    };

    // Derived from the request directly (not from `white`/`black`
    // above) so a game's recorded participant info reflects what was
    // actually asked for even independent of engine-name resolution --
    // moot in practice since an unknown name already returned 400
    // above, but keeps this derivation simple and total rather than
    // needing to unwrap an `Option<EngineConfig>` back into a name.
    let white_info = participant_info(&request.white);
    let black_info = participant_info(&request.black);

    let snapshot = state.store.create(white_info, black_info);
    let slots = EngineSlots { white, black };
    if slots.any_engine() {
        let move_time_ms = request.move_time_ms.unwrap_or(DEFAULT_MOVE_TIME_MS);
        tokio::spawn(crate::game::run_engine_loop(
            state.store.clone(),
            snapshot.id,
            slots,
            move_time_ms,
        ));
    }

    (StatusCode::CREATED, Json(snapshot)).into_response()
}

fn participant_info(participant: &Option<ParticipantRequest>) -> ParticipantInfo {
    match participant {
        None => ParticipantInfo::Human,
        Some(participant) => ParticipantInfo::Engine {
            name: participant.engine_name().to_string(),
            debug: participant.debug(),
        },
    }
}

async fn get_game(State(state): State<ApiState>, Path(id): Path<String>) -> impl IntoResponse {
    let Ok(id) = id.parse() else {
        return (
            StatusCode::BAD_REQUEST,
            Json(ErrorBody::new("malformed game id")),
        )
            .into_response();
    };
    match state.store.snapshot(id) {
        Some(snapshot) => Json(snapshot).into_response(),
        None => (StatusCode::NOT_FOUND, Json(ErrorBody::new("no such game"))).into_response(),
    }
}

#[derive(Deserialize)]
struct ApplyMoveRequest {
    /// UCI long algebraic notation, e.g. `"e2e4"` or `"e7e8q"`.
    uci: String,
}

async fn apply_move(
    State(state): State<ApiState>,
    Path(id): Path<String>,
    Json(request): Json<ApplyMoveRequest>,
) -> impl IntoResponse {
    let Ok(id) = id.parse() else {
        return (
            StatusCode::BAD_REQUEST,
            Json(ErrorBody::new("malformed game id")),
        )
            .into_response();
    };
    match state.store.apply_move(id, &request.uci) {
        Ok(snapshot) => Json(snapshot).into_response(),
        Err(None) => (StatusCode::NOT_FOUND, Json(ErrorBody::new("no such game"))).into_response(),
        Err(Some(ApplyMoveError::GameNotRunning)) => (
            StatusCode::CONFLICT,
            Json(ErrorBody::new("game is not running")),
        )
            .into_response(),
        Err(Some(ApplyMoveError::NotAWellFormedMove)) => (
            StatusCode::BAD_REQUEST,
            Json(ErrorBody::new("not a well-formed UCI move")),
        )
            .into_response(),
        Err(Some(ApplyMoveError::IllegalMove)) => (
            StatusCode::UNPROCESSABLE_ENTITY,
            Json(ErrorBody::new("illegal move")),
        )
            .into_response(),
    }
}

/// The JSON shape a `GameEvent` is sent over `/ws/games/:id` as.
/// `#[serde(tag = "type")]` gives each variant a `"type"` discriminator
/// field the frontend can switch on (`"uci"` / `"updated"`), rather
/// than an untagged shape that would force it to guess from which
/// fields happen to be present.
#[derive(Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum GameEventWire {
    Uci {
        color: WireColor,
        direction: WireDirection,
        line: String,
    },
    Updated {
        snapshot: GameSnapshot,
    },
}

#[derive(Serialize)]
#[serde(rename_all = "snake_case")]
enum WireColor {
    White,
    Black,
}

#[derive(Serialize)]
#[serde(rename_all = "snake_case")]
enum WireDirection {
    Sent,
    Received,
}

impl From<GameEvent> for GameEventWire {
    fn from(event: GameEvent) -> Self {
        match event {
            GameEvent::Uci {
                color,
                direction,
                line,
            } => GameEventWire::Uci {
                color: match color {
                    bee_chess_core::Color::White => WireColor::White,
                    bee_chess_core::Color::Black => WireColor::Black,
                },
                direction: match direction {
                    UciDirection::Sent => WireDirection::Sent,
                    UciDirection::Received => WireDirection::Received,
                },
                line,
            },
            GameEvent::Updated(snapshot) => GameEventWire::Updated { snapshot },
        }
    }
}

/// Upgrades to a WebSocket and streams `id`'s live `GameEvent`s as
/// JSON (see `GameEventWire`) until either side closes or the game's
/// event channel is gone (the game never existed, or -- there's no
/// persistence yet -- this process restarted since it was created).
/// Closes the socket immediately, with no error frame, if `id` has no
/// event channel at all: this is a "nothing to stream," not a
/// malformed-request condition worth a different close code for.
async fn game_events_ws(
    ws: WebSocketUpgrade,
    State(state): State<ApiState>,
    Path(id): Path<String>,
) -> impl IntoResponse {
    let Ok(id) = id.parse() else {
        return (
            StatusCode::BAD_REQUEST,
            Json(ErrorBody::new("malformed game id")),
        )
            .into_response();
    };
    let Some(mut receiver) = state.store.subscribe(id) else {
        return (StatusCode::NOT_FOUND, Json(ErrorBody::new("no such game"))).into_response();
    };

    ws.on_upgrade(move |mut socket: WebSocket| async move {
        loop {
            let event = tokio::select! {
                event = receiver.recv() => event,
                _ = socket.recv() => return, // browser closed its side
            };
            match event {
                Ok(event) => {
                    let wire = GameEventWire::from(event);
                    let Ok(text) = serde_json::to_string(&wire) else {
                        continue; // should never fail for this shape; skip rather than panic
                    };
                    if socket.send(Message::Text(text.into())).await.is_err() {
                        return; // browser side gone
                    }
                }
                Err(broadcast::error::RecvError::Lagged(_)) => continue, // see EVENT_CHANNEL_CAPACITY's docs
                Err(broadcast::error::RecvError::Closed) => return,      // game's channel is gone
            }
        }
    })
    .into_response()
}

#[derive(serde::Serialize)]
struct ErrorBody {
    error: String,
}

impl ErrorBody {
    fn new(message: impl Into<String>) -> Self {
        ErrorBody {
            error: message.into(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::Body;
    use axum::http::{Request, StatusCode};
    use serde_json::Value;
    use tower::ServiceExt;

    async fn body_json(response: axum::response::Response) -> Value {
        let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .expect("read body");
        serde_json::from_slice(&bytes).expect("valid JSON body")
    }

    #[tokio::test]
    async fn post_games_creates_a_game_at_startpos() {
        let app = router(GameStore::new(), EngineRegistry::new());

        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/games")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::CREATED);
        let body = body_json(response).await;
        assert_eq!(body["status"], "running");
        assert_eq!(body["moves"], serde_json::json!([]));
        assert!(body["fen"].as_str().unwrap().starts_with("rnbqkbnr/"));
        assert_eq!(body["white"], serde_json::json!({"kind": "human"}));
        assert_eq!(body["black"], serde_json::json!({"kind": "human"}));
    }

    #[tokio::test]
    async fn post_games_snapshot_carries_engine_participant_info() {
        // What lets a client resume a game after a refresh (persisting
        // only the game id) reconstruct which side is engine-driven
        // and with what config, without having remembered it itself.
        let mut registry = EngineRegistry::new();
        registry.insert("fake", fake_engine_spec());
        let app = router(GameStore::new(), registry);

        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/games")
                    .header("content-type", "application/json")
                    .body(Body::from(
                        serde_json::json!({"white": {"engine": "fake", "debug": true}}).to_string(),
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::CREATED);
        let body = body_json(response).await;
        assert_eq!(
            body["white"],
            serde_json::json!({"kind": "engine", "name": "fake", "debug": true})
        );
        assert_eq!(body["black"], serde_json::json!({"kind": "human"}));
    }

    #[tokio::test]
    async fn get_unknown_game_is_404() {
        let app = router(GameStore::new(), EngineRegistry::new());

        let response = app
            .oneshot(
                Request::builder()
                    .uri(format!("/api/games/{}", uuid::Uuid::new_v4()))
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn get_malformed_id_is_400() {
        let app = router(GameStore::new(), EngineRegistry::new());

        let response = app
            .oneshot(
                Request::builder()
                    .uri("/api/games/not-a-uuid")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn create_then_get_round_trips_the_same_game() {
        let app = router(GameStore::new(), EngineRegistry::new());

        let create_response = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/games")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        let created = body_json(create_response).await;
        let id = created["id"].as_str().unwrap();

        let get_response = app
            .oneshot(
                Request::builder()
                    .uri(format!("/api/games/{id}"))
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(get_response.status(), StatusCode::OK);
        let fetched = body_json(get_response).await;
        assert_eq!(fetched, created);
    }

    #[tokio::test]
    async fn post_moves_applies_a_legal_move_and_returns_the_new_snapshot() {
        let app = router(GameStore::new(), EngineRegistry::new());

        let created = body_json(
            app.clone()
                .oneshot(
                    Request::builder()
                        .method("POST")
                        .uri("/api/games")
                        .body(Body::empty())
                        .unwrap(),
                )
                .await
                .unwrap(),
        )
        .await;
        let id = created["id"].as_str().unwrap();

        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri(format!("/api/games/{id}/moves"))
                    .header("content-type", "application/json")
                    .body(Body::from(serde_json::json!({"uci": "e2e4"}).to_string()))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
        let body = body_json(response).await;
        assert_eq!(body["moves"], serde_json::json!(["e2e4"]));
    }

    #[tokio::test]
    async fn post_moves_with_an_illegal_move_is_422() {
        let app = router(GameStore::new(), EngineRegistry::new());

        let created = body_json(
            app.clone()
                .oneshot(
                    Request::builder()
                        .method("POST")
                        .uri("/api/games")
                        .body(Body::empty())
                        .unwrap(),
                )
                .await
                .unwrap(),
        )
        .await;
        let id = created["id"].as_str().unwrap();

        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri(format!("/api/games/{id}/moves"))
                    .header("content-type", "application/json")
                    .body(Body::from(serde_json::json!({"uci": "e2e5"}).to_string()))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::UNPROCESSABLE_ENTITY);
    }

    #[tokio::test]
    async fn post_moves_on_unknown_game_is_404() {
        let app = router(GameStore::new(), EngineRegistry::new());

        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri(format!("/api/games/{}/moves", uuid::Uuid::new_v4()))
                    .header("content-type", "application/json")
                    .body(Body::from(serde_json::json!({"uci": "e2e4"}).to_string()))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::NOT_FOUND);
    }

    /// A fake "engine" (a `sh` one-liner speaking just enough UCI) that
    /// always replies `e2e4` regardless of position, so 69b's automatic
    /// loop can be tested without depending on real Stockfish/Bee
    /// binaries being built in this environment.
    fn fake_engine_spec() -> EngineSpec {
        EngineSpec {
            argv: vec![
                "sh".to_string(),
                "-c".to_string(),
                r#"
                while read -r line; do
                    case "$line" in
                        uci) echo "uciok" ;;
                        isready) echo "readyok" ;;
                        go*) echo "bestmove e2e4" ;;
                    esac
                done
                "#
                .to_string(),
            ],
            cwd: std::env::temp_dir(),
        }
    }

    #[tokio::test]
    async fn post_games_with_an_unknown_engine_name_is_400() {
        let app = router(GameStore::new(), EngineRegistry::new());

        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/games")
                    .header("content-type", "application/json")
                    .body(Body::from(
                        serde_json::json!({"white": "no-such-engine"}).to_string(),
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn post_games_with_an_engine_side_plays_a_move_automatically() {
        let mut registry = EngineRegistry::new();
        registry.insert("fake", fake_engine_spec());
        let store = GameStore::new();
        let app = router(store.clone(), registry);

        let created = body_json(
            app.oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/games")
                    .header("content-type", "application/json")
                    .body(Body::from(
                        serde_json::json!({"white": "fake", "move_time_ms": 50}).to_string(),
                    ))
                    .unwrap(),
            )
            .await
            .unwrap(),
        )
        .await;
        let id: crate::game::GameId = created["id"].as_str().unwrap().parse().unwrap();

        // The engine loop runs in a spawned background task -- poll
        // the store directly (bypassing the router entirely) until it
        // shows the automatic move, rather than assuming any fixed
        // delay is enough.
        let deadline = tokio::time::Instant::now() + std::time::Duration::from_secs(5);
        loop {
            let snapshot = store.snapshot(id).expect("game should still exist");
            if !snapshot.moves.is_empty() {
                assert_eq!(snapshot.moves, vec!["e2e4".to_string()]);
                return;
            }
            assert!(
                tokio::time::Instant::now() < deadline,
                "engine loop should have played a move by now"
            );
            tokio::time::sleep(std::time::Duration::from_millis(20)).await;
        }
    }

    #[tokio::test]
    async fn post_games_with_engine_options_and_debug_applies_them_before_any_go() {
        let mut registry = EngineRegistry::new();
        registry.insert("fake", fake_engine_spec());
        let store = GameStore::new();
        let app = router(store.clone(), registry);

        let created = body_json(
            app.oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/games")
                    .header("content-type", "application/json")
                    .body(Body::from(
                        serde_json::json!({
                            "white": {
                                "engine": "fake",
                                "options": {"UCI_LimitStrength": true, "UCI_Elo": 1600},
                                "debug": true
                            },
                            "move_time_ms": 50
                        })
                        .to_string(),
                    ))
                    .unwrap(),
            )
            .await
            .unwrap(),
        )
        .await;
        let id: crate::game::GameId = created["id"].as_str().unwrap().parse().unwrap();

        let mut events = store.subscribe(id).expect("game should have a channel");
        let sent_lines = tokio::time::timeout(std::time::Duration::from_secs(5), async {
            let mut sent = Vec::new();
            loop {
                if let Ok(GameEvent::Uci {
                    direction: crate::uci_process::UciDirection::Sent,
                    line,
                    ..
                }) = events.recv().await
                {
                    let is_go = line.starts_with("go");
                    sent.push(line);
                    if is_go {
                        return sent;
                    }
                }
            }
        })
        .await
        .expect("should see a 'go' within the timeout");

        assert!(
            sent_lines
                .iter()
                .any(|line| line == "setoption name UCI_LimitStrength value true"),
            "should have sent UCI_LimitStrength before 'go': {sent_lines:?}"
        );
        assert!(
            sent_lines
                .iter()
                .any(|line| line == "setoption name UCI_Elo value 1600"),
            "should have sent UCI_Elo before 'go': {sent_lines:?}"
        );
        assert!(
            sent_lines.iter().any(|line| line == "debug on"),
            "should have sent 'debug on' before 'go': {sent_lines:?}"
        );
    }

    /// Binds `router(store, engines)` on an ephemeral local port and
    /// returns its base `ws://` URL -- WebSocket upgrades don't work
    /// through axum's `oneshot` the way plain HTTP requests do (see the
    /// other tests above), so these need a real bound server instead.
    async fn spawn_real_server(store: GameStore, engines: EngineRegistry) -> String {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind");
        let addr = listener.local_addr().expect("local_addr");
        let app = router(store, engines);
        tokio::spawn(async move {
            axum::serve(listener, app).await.ok();
        });
        format!("ws://{addr}")
    }

    #[tokio::test]
    async fn ws_games_streams_an_updated_event_when_a_move_is_applied() {
        use futures_util::StreamExt;
        use tokio_tungstenite::tungstenite::Message as ClientMessage;

        let store = GameStore::new();
        let created = store.create(ParticipantInfo::Human, ParticipantInfo::Human);
        let base_url = spawn_real_server(store.clone(), EngineRegistry::new()).await;

        let (mut ws, _) =
            tokio_tungstenite::connect_async(format!("{base_url}/ws/games/{}", created.id))
                .await
                .expect("connect");

        // Apply the move directly through the store (equivalent to a
        // POST from another client) after subscribing, so the event
        // is guaranteed to be published after this socket is already
        // listening.
        store
            .apply_move(created.id, "e2e4")
            .expect("e2e4 should be legal");

        let msg = tokio::time::timeout(std::time::Duration::from_secs(5), ws.next())
            .await
            .expect("should receive an event within the timeout")
            .expect("stream should not end")
            .expect("should not be a websocket error");
        let ClientMessage::Text(text) = msg else {
            panic!("expected a text frame, got {msg:?}");
        };
        let event: serde_json::Value = serde_json::from_str(&text).expect("valid JSON event");

        assert_eq!(event["type"], "updated");
        assert_eq!(event["snapshot"]["moves"], serde_json::json!(["e2e4"]));
    }

    #[tokio::test]
    async fn ws_games_on_unknown_id_rejects_the_upgrade_with_404() {
        // No such game -- there's nothing to stream, and the handler
        // checks for a channel *before* upgrading, so the connection
        // attempt itself fails with a plain HTTP 404 rather than
        // upgrading and then immediately closing. tokio_tungstenite
        // surfaces that as a connect error carrying the response.
        let base_url = spawn_real_server(GameStore::new(), EngineRegistry::new()).await;

        let result = tokio_tungstenite::connect_async(format!(
            "{base_url}/ws/games/{}",
            uuid::Uuid::new_v4()
        ))
        .await;

        let Err(tokio_tungstenite::tungstenite::Error::Http(response)) = result else {
            panic!("expected an HTTP error rejecting the upgrade, got {result:?}");
        };
        assert_eq!(response.status(), StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn ws_games_streams_uci_events_from_an_engine_driven_game() {
        use futures_util::StreamExt;
        use tokio_tungstenite::tungstenite::Message as ClientMessage;

        let mut registry = EngineRegistry::new();
        registry.insert("fake", fake_engine_spec());
        let store = GameStore::new();
        let base_url = spawn_real_server(store.clone(), registry.clone()).await;

        let app = router(store, registry);
        let created = body_json(
            app.oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/games")
                    .header("content-type", "application/json")
                    .body(Body::from(
                        serde_json::json!({"white": "fake", "move_time_ms": 50}).to_string(),
                    ))
                    .unwrap(),
            )
            .await
            .unwrap(),
        )
        .await;
        let id = created["id"].as_str().unwrap();

        let (mut ws, _) = tokio_tungstenite::connect_async(format!("{base_url}/ws/games/{id}"))
            .await
            .expect("connect");

        // The fake engine's very first line is "uciok" (received), so
        // waiting for any "uci" event with the expected shape confirms
        // raw engine traffic -- not just the final Updated snapshot --
        // is actually flowing over this socket.
        let saw_uci_event = tokio::time::timeout(std::time::Duration::from_secs(5), async {
            loop {
                let msg = ws
                    .next()
                    .await
                    .expect("stream should not end")
                    .expect("no ws error");
                let ClientMessage::Text(text) = msg else {
                    continue;
                };
                let event: serde_json::Value =
                    serde_json::from_str(&text).expect("valid JSON event");
                if event["type"] == "uci" {
                    return;
                }
            }
        })
        .await;

        assert!(
            saw_uci_event.is_ok(),
            "should have seen at least one raw UCI event from the engine-driven game"
        );
    }
}
