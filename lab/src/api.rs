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
//! No WebSocket event stream yet -- that's 69c's territory (a client
//! can already poll `GET /api/games/:id` to see the effect of every
//! move, engine or human).

use std::collections::HashMap;

use axum::extract::{Json, Path, State};
use axum::http::StatusCode;
use axum::response::IntoResponse;
use axum::routing::{get, post};
use axum::Router;
use serde::Deserialize;

use crate::game::{ApplyMoveError, EngineSlots, GameStore};
use crate::uci_relay::EngineSpec;

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
        .with_state(ApiState { store, engines })
}

/// Default per-move time budget for an engine-driven side, when
/// `CreateGameRequest` doesn't specify one. Generous enough to see a
/// real search happen; nowhere near a real game clock (there is no
/// clock at all yet -- see #69's still-open clocks scope).
const DEFAULT_MOVE_TIME_MS: u64 = 200;

#[derive(Debug, Deserialize, Default)]
struct CreateGameRequest {
    /// Engine name (must be in the server's `EngineRegistry`) to drive
    /// White automatically, or omitted/`null` for a human-controlled
    /// White (moves arrive via `POST /api/games/:id/moves` instead).
    #[serde(default)]
    white: Option<String>,
    #[serde(default)]
    black: Option<String>,
    #[serde(default)]
    move_time_ms: Option<u64>,
}

async fn create_game(
    State(state): State<ApiState>,
    body: Option<Json<CreateGameRequest>>,
) -> impl IntoResponse {
    let request = body.map(|Json(r)| r).unwrap_or_default();

    let resolve = |name: &Option<String>| -> Result<Option<EngineSpec>, String> {
        match name {
            None => Ok(None),
            Some(name) => state
                .engines
                .get(name)
                .map(Some)
                .ok_or_else(|| format!("unknown engine {name:?}")),
        }
    };
    let white = match resolve(&request.white) {
        Ok(spec) => spec,
        Err(message) => {
            return (StatusCode::BAD_REQUEST, Json(ErrorBody::new(message))).into_response()
        }
    };
    let black = match resolve(&request.black) {
        Ok(spec) => spec,
        Err(message) => {
            return (StatusCode::BAD_REQUEST, Json(ErrorBody::new(message))).into_response()
        }
    };

    let snapshot = state.store.create();
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
}
