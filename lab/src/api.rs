//! HTTP surface for authoritative game state (#69 / 67b, slice 69a):
//! `POST /api/games` creates a game, `GET /api/games/:id` returns its
//! complete current snapshot, `POST /api/games/:id/moves` applies a
//! move to it.
//!
//! `POST /api/games/:id/moves` exists in this slice specifically so a
//! game can be driven and observed entirely over HTTP before 69b wires
//! up an automatic server-side engine loop -- once that lands, this
//! endpoint keeps working (a human move still needs to reach the
//! server somehow), it just stops being the *only* way a move happens.
//!
//! No WebSocket event stream yet -- that's also 69b/69c's territory
//! (a client can already poll `GET /api/games/:id` to see the effect of
//! a move applied through this API).

use axum::extract::{Json, Path, State};
use axum::http::StatusCode;
use axum::response::IntoResponse;
use axum::routing::{get, post};
use axum::Router;
use serde::Deserialize;

use crate::game::{ApplyMoveError, GameStore};

pub fn router(store: GameStore) -> Router {
    Router::new()
        .route("/api/games", post(create_game))
        .route("/api/games/{id}", get(get_game))
        .route("/api/games/{id}/moves", post(apply_move))
        .with_state(store)
}

async fn create_game(State(store): State<GameStore>) -> impl IntoResponse {
    let snapshot = store.create();
    (StatusCode::CREATED, Json(snapshot))
}

async fn get_game(State(store): State<GameStore>, Path(id): Path<String>) -> impl IntoResponse {
    let Ok(id) = id.parse() else {
        return (
            StatusCode::BAD_REQUEST,
            Json(ErrorBody::new("malformed game id")),
        )
            .into_response();
    };
    match store.snapshot(id) {
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
    State(store): State<GameStore>,
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
    match store.apply_move(id, &request.uci) {
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
        let app = router(GameStore::new());

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
        let app = router(GameStore::new());

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
        let app = router(GameStore::new());

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
        let app = router(GameStore::new());

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
        let app = router(GameStore::new());

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
        let app = router(GameStore::new());

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
        let app = router(GameStore::new());

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
}
