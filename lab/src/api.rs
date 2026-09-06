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

use crate::experiment::{
    EngineVariant, ExperimentId, ExperimentSnapshot, ExperimentSpec, ExperimentStore,
};
use crate::game::{
    ApplyMoveError, EngineConfig, EngineSlots, EngineSpec, GameEvent, GameSnapshot, GameStore,
    ParticipantInfo, TimeControl,
};
use crate::uci_process::{UciDirection, UciOption, UciProcess, UciProcessError};

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
    /// Not a constructor parameter like `store`/`engines`: nothing
    /// outside this module needs to share or preconfigure it (an
    /// experiment's whole spec arrives with the `POST` that creates
    /// it), so `router` owns creating one rather than forcing every
    /// caller (including ~15 existing tests) to pass a third argument
    /// for something they don't use.
    experiments: ExperimentStore,
}

pub fn router(store: GameStore, engines: EngineRegistry) -> Router {
    Router::new()
        .route("/api/games", get(list_games).post(create_game))
        .route("/api/games/{id}", get(get_game))
        .route("/api/games/{id}/moves", post(apply_move))
        .route("/ws/games/{id}", get(game_events_ws))
        .route("/api/engines/{name}/options", get(get_engine_options))
        .route(
            "/api/experiments",
            get(list_experiments).post(create_experiment),
        )
        .route("/api/experiments/{id}", get(get_experiment))
        .with_state(ApiState {
            store,
            engines,
            experiments: ExperimentStore::new(),
        })
}

/// `GET /api/engines/:name/options`: the UCI options `name` (e.g.
/// `"bee"`) advertises during its own handshake, in the same generic
/// `check`/`spin`/`combo`/`string` vocabulary UCI itself uses -- see
/// `UciOption`. This is the discovery contract the frontend's
/// experiment-configuration UI is meant to render generically: adding
/// a new `setoption` to an engine (e.g. #104's `UseTT`/
/// `UseQuiescence`) makes it appear here automatically, with no Lab or
/// frontend code needing to know its name ahead of time.
///
/// Spawns `name` fresh for this request alone (and kills it once the
/// handshake completes -- `UciProcess::drop` handles that) rather than
/// keeping a running instance around just to answer this: an engine's
/// advertised options don't change between runs of the same binary,
/// so there's nothing to gain from a long-lived process here, and this
/// keeps the endpoint from needing any of `run_engine_loop`'s
/// game-lifecycle machinery. A real registry (#70) may want to cache
/// this instead of spawning per request; not worth it yet at this
/// endpoint's expected call volume (opening the experiment-setup
/// screen, not something polled).
async fn get_engine_options(
    State(state): State<ApiState>,
    Path(name): Path<String>,
) -> Result<Json<Vec<UciOption>>, (StatusCode, String)> {
    let spec = state
        .engines
        .get(&name)
        .ok_or((StatusCode::NOT_FOUND, format!("unknown engine: {name}")))?;

    let process = UciProcess::spawn(&spec.argv, &spec.cwd, None)
        .await
        .map_err(|err| (StatusCode::BAD_GATEWAY, engine_spawn_error_message(&err)))?;

    Ok(Json(process.options().to_vec()))
}

fn engine_spawn_error_message(err: &UciProcessError) -> String {
    format!("failed to query engine options: {err}")
}

/// One side of an experiment request: a human-readable label plus the
/// `setoption`s that define it -- e.g. `{"label": "Candidate",
/// "options": {"UseTT": false}}`. Deliberately no `engine` field here
/// (unlike `ParticipantRequest`): v1 experiments are Bee-vs-Bee only
/// (see `experiment`'s module docs), so `CreateExperimentRequest`
/// names the engine once for the whole experiment, not per variant.
#[derive(Debug, Deserialize)]
struct ExperimentVariantRequest {
    label: String,
    #[serde(default)]
    options: HashMap<String, serde_json::Value>,
}

impl ExperimentVariantRequest {
    /// Same value-stringification rule as `ParticipantRequest::
    /// options` -- see its docs.
    fn options(&self) -> Vec<(String, String)> {
        self.options
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
}

#[derive(Debug, Deserialize)]
struct CreateExperimentRequest {
    /// Which engine both variants are (see this module's docs on why
    /// v1 doesn't support two different engines). Defaults to `"bee"`
    /// since that's the only realistic value right now, but still
    /// resolved through `EngineRegistry` like everything else rather
    /// than hardcoded, so a differently-registered name still works.
    #[serde(default = "default_experiment_engine")]
    engine: String,
    variant_a: ExperimentVariantRequest,
    variant_b: ExperimentVariantRequest,
    games: u32,
    #[serde(default = "default_experiment_concurrency")]
    concurrency: u32,
    /// Both variants' shared clock policy -- see `TimeControl`'s docs
    /// on why time control belongs to the experiment, not either
    /// variant. `None` falls back to `move_time_ms` (old clients/
    /// requests) or, failing that, `DEFAULT_MOVE_TIME_MS` -- see
    /// `resolve`.
    #[serde(default)]
    time_control: Option<TimeControlRequest>,
    /// Deprecated alias for `time_control: {"type": "move_time",
    /// "move_time_ms": ...}` -- kept working so existing callers/tests
    /// that only ever knew about a flat movetime don't break.
    #[serde(default)]
    move_time_ms: Option<u64>,
    #[serde(default)]
    debug: bool,
}

fn default_experiment_engine() -> String {
    "bee".to_string()
}

fn default_experiment_concurrency() -> u32 {
    2
}

/// Wire shape of `TimeControl`, deserialized separately from it so a
/// malformed/missing `time_control` can fall back to `move_time_ms`
/// (see `CreateExperimentRequest`/`CreateGameRequest`) without
/// `TimeControl` itself needing a `Default` impl that would silently
/// paper over a real client bug in other contexts.
#[derive(Debug, Clone, Copy, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum TimeControlRequest {
    MoveTime { move_time_ms: u64 },
    Fischer { initial_ms: u64, increment_ms: u64 },
}

impl From<TimeControlRequest> for TimeControl {
    fn from(request: TimeControlRequest) -> Self {
        match request {
            TimeControlRequest::MoveTime { move_time_ms } => TimeControl::MoveTime { move_time_ms },
            TimeControlRequest::Fischer {
                initial_ms,
                increment_ms,
            } => TimeControl::Fischer {
                initial_ms,
                increment_ms,
            },
        }
    }
}

/// Resolves a request's `time_control`/`move_time_ms` fields to a
/// concrete `TimeControl`: `time_control` wins if present, else
/// `move_time_ms` as a fixed movetime, else `DEFAULT_MOVE_TIME_MS`.
fn resolve_time_control(
    time_control: Option<TimeControlRequest>,
    move_time_ms: Option<u64>,
) -> TimeControl {
    match time_control {
        Some(request) => request.into(),
        None => TimeControl::fixed_move_time(move_time_ms.unwrap_or(DEFAULT_MOVE_TIME_MS)),
    }
}

/// `POST /api/experiments`: starts a new A/B experiment (see
/// `experiment`'s module docs) and immediately begins running it in
/// the background -- the response carries the experiment's id right
/// away (status `Running`, no games yet), the same "create returns
/// immediately, poll/subscribe for progress" shape `POST /api/games`
/// already uses for an engine-driven game.
async fn create_experiment(
    State(state): State<ApiState>,
    Json(request): Json<CreateExperimentRequest>,
) -> impl IntoResponse {
    if request.games == 0 || request.games % 2 != 0 {
        return (
            StatusCode::BAD_REQUEST,
            Json(ErrorBody::new(
                "games must be a positive even number so every game has a color-swapped partner",
            )),
        )
            .into_response();
    }
    if request.concurrency == 0 || request.concurrency > request.games {
        return (
            StatusCode::BAD_REQUEST,
            Json(ErrorBody::new(
                "concurrency must be between 1 and the number of games",
            )),
        )
            .into_response();
    }

    let Some(spec) = state.engines.get(&request.engine) else {
        return (
            StatusCode::BAD_REQUEST,
            Json(ErrorBody::new(format!(
                "unknown engine {:?}",
                request.engine
            ))),
        )
            .into_response();
    };

    let variant_a = EngineVariant {
        label: request.variant_a.label.clone(),
        config: EngineConfig {
            spec: spec.clone(),
            options: request.variant_a.options(),
            debug: request.debug,
        },
    };
    let variant_b = EngineVariant {
        label: request.variant_b.label.clone(),
        config: EngineConfig {
            spec,
            options: request.variant_b.options(),
            debug: request.debug,
        },
    };
    let experiment_spec = ExperimentSpec {
        variant_a,
        variant_b,
        requested_games: request.games,
        concurrency: request.concurrency,
        time_control: resolve_time_control(request.time_control, request.move_time_ms),
    };

    let snapshot = state.experiments.create(experiment_spec.clone());
    tokio::spawn(crate::experiment::run_experiment(
        state.store.clone(),
        state.experiments.clone(),
        snapshot.id,
        experiment_spec,
    ));

    (StatusCode::CREATED, Json(snapshot)).into_response()
}

/// `GET /api/experiments`: every experiment the server currently knows
/// about, newest first -- same "one list, filter client-side by
/// status" reasoning as `list_games`.
async fn list_experiments(State(state): State<ApiState>) -> impl IntoResponse {
    Json(state.experiments.list())
}

async fn get_experiment(
    State(state): State<ApiState>,
    Path(id): Path<String>,
) -> Result<Json<ExperimentSnapshot>, (StatusCode, Json<ErrorBody>)> {
    let id: ExperimentId = id.parse().map_err(|_| {
        (
            StatusCode::BAD_REQUEST,
            Json(ErrorBody::new("malformed experiment id")),
        )
    })?;
    state.experiments.snapshot(id).map(Json).ok_or((
        StatusCode::NOT_FOUND,
        Json(ErrorBody::new("no such experiment")),
    ))
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
    /// See `CreateExperimentRequest::time_control`'s docs -- same
    /// resolution rules (`time_control` wins, else `move_time_ms`,
    /// else the default).
    #[serde(default)]
    time_control: Option<TimeControlRequest>,
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

    let time_control = resolve_time_control(request.time_control, request.move_time_ms);
    let snapshot = state.store.create(white_info, black_info, time_control);
    let slots = EngineSlots { white, black };
    if slots.any_engine() {
        tokio::spawn(crate::game::run_engine_loop(
            state.store.clone(),
            snapshot.id,
            slots,
            time_control,
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

/// `GET /api/games`: every game the server currently knows about,
/// newest first -- the dashboard's running/past game lists filter this
/// client-side by `status` rather than the server offering separate
/// endpoints for each, since it's the same underlying list either way.
async fn list_games(State(state): State<ApiState>) -> impl IntoResponse {
    Json(state.store.list())
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
        snapshot: Box<GameSnapshot>,
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
    use crate::game::GameStatus;
    use axum::body::Body;
    use axum::http::{Request, StatusCode};
    use serde_json::Value;
    use tower::ServiceExt;
    use uuid::Uuid;

    async fn body_json(response: axum::response::Response) -> Value {
        let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .expect("read body");
        serde_json::from_slice(&bytes).expect("valid JSON body")
    }

    #[tokio::test]
    async fn get_games_lists_every_created_game_newest_first() {
        let store = GameStore::new();
        let app = router(store.clone(), EngineRegistry::new());

        let first = store.create(
            ParticipantInfo::Human,
            ParticipantInfo::Human,
            TimeControl::fixed_move_time(200),
        );
        let second = store.create(
            ParticipantInfo::Human,
            ParticipantInfo::Human,
            TimeControl::fixed_move_time(200),
        );

        let response = app
            .oneshot(
                Request::builder()
                    .uri("/api/games")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
        let body = body_json(response).await;
        let ids: Vec<String> = body
            .as_array()
            .unwrap()
            .iter()
            .map(|g| g["id"].as_str().unwrap().to_string())
            .collect();
        assert_eq!(ids, vec![second.id.to_string(), first.id.to_string()]);
    }

    #[tokio::test]
    async fn get_games_is_empty_for_a_fresh_server() {
        let app = router(GameStore::new(), EngineRegistry::new());

        let response = app
            .oneshot(
                Request::builder()
                    .uri("/api/games")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
        let body = body_json(response).await;
        assert_eq!(body.as_array().unwrap().len(), 0);
    }

    #[tokio::test]
    async fn get_experiments_lists_every_created_experiment_newest_first() {
        let mut registry = EngineRegistry::new();
        registry.insert("fake", fake_engine_spec());
        let app = router(GameStore::new(), registry);

        let first = body_json(
            app.clone()
                .oneshot(
                    Request::builder()
                        .method("POST")
                        .uri("/api/experiments")
                        .header("content-type", "application/json")
                        .body(Body::from(
                            serde_json::json!({
                                "engine": "fake",
                                "variant_a": {"label": "A1"},
                                "variant_b": {"label": "B1"},
                                "games": 2,
                            })
                            .to_string(),
                        ))
                        .unwrap(),
                )
                .await
                .unwrap(),
        )
        .await;
        let second = body_json(
            app.clone()
                .oneshot(
                    Request::builder()
                        .method("POST")
                        .uri("/api/experiments")
                        .header("content-type", "application/json")
                        .body(Body::from(
                            serde_json::json!({
                                "engine": "fake",
                                "variant_a": {"label": "A2"},
                                "variant_b": {"label": "B2"},
                                "games": 2,
                            })
                            .to_string(),
                        ))
                        .unwrap(),
                )
                .await
                .unwrap(),
        )
        .await;

        let response = app
            .oneshot(
                Request::builder()
                    .uri("/api/experiments")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
        let body = body_json(response).await;
        let ids: Vec<String> = body
            .as_array()
            .unwrap()
            .iter()
            .map(|e| e["id"].as_str().unwrap().to_string())
            .collect();
        assert_eq!(
            ids,
            vec![
                second["id"].as_str().unwrap(),
                first["id"].as_str().unwrap()
            ]
        );
    }

    #[tokio::test]
    async fn a_game_created_by_an_experiment_carries_its_experiment_id() {
        let mut registry = EngineRegistry::new();
        registry.insert("fake", fake_engine_spec());
        let store = GameStore::new();
        let app = router(store.clone(), registry);

        let created = body_json(
            app.oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/experiments")
                    .header("content-type", "application/json")
                    .body(Body::from(
                        serde_json::json!({
                            "engine": "fake",
                            "variant_a": {"label": "A"},
                            "variant_b": {"label": "B"},
                            "games": 2,
                            "move_time_ms": 20,
                        })
                        .to_string(),
                    ))
                    .unwrap(),
            )
            .await
            .unwrap(),
        )
        .await;
        let experiment_id = created["id"].as_str().unwrap().to_string();

        let deadline = tokio::time::Instant::now() + std::time::Duration::from_secs(5);
        loop {
            let games = store.list();
            if let Some(game) = games
                .iter()
                .find(|g| g.experiment_id.map(|id| id.to_string()) == Some(experiment_id.clone()))
            {
                if !game.moves.is_empty() || game.status != GameStatus::Running {
                    return;
                }
            }
            assert!(
                tokio::time::Instant::now() < deadline,
                "no game tagged with this experiment appeared in time"
            );
            tokio::time::sleep(std::time::Duration::from_millis(20)).await;
        }
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
    async fn get_engine_options_for_an_unknown_engine_is_404() {
        let app = router(GameStore::new(), EngineRegistry::new());

        let response = app
            .oneshot(
                Request::builder()
                    .uri("/api/engines/no-such-engine/options")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn get_engine_options_returns_the_options_the_engine_advertised() {
        let spec = EngineSpec {
            argv: vec![
                "sh".to_string(),
                "-c".to_string(),
                r#"
                read _
                echo "option name UseTT type check default true"
                echo "option name Evaluator type combo default Positional var Positional var Material"
                echo "uciok"
                read _; echo "readyok"
                "#
                .to_string(),
            ],
            cwd: std::env::temp_dir(),
        };
        let mut registry = EngineRegistry::new();
        registry.insert("fake", spec);
        let app = router(GameStore::new(), registry);

        let response = app
            .oneshot(
                Request::builder()
                    .uri("/api/engines/fake/options")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
        let body = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        let options: Vec<UciOption> = serde_json::from_slice(&body).unwrap();
        assert_eq!(
            options,
            vec![
                UciOption::Check {
                    name: "UseTT".to_string(),
                    default: true,
                },
                UciOption::Combo {
                    name: "Evaluator".to_string(),
                    default: "Positional".to_string(),
                    values: vec!["Positional".to_string(), "Material".to_string()],
                },
            ]
        );
    }

    #[tokio::test]
    async fn post_experiments_requires_a_positive_even_game_count() {
        let mut registry = EngineRegistry::new();
        registry.insert("fake", fake_engine_spec());
        let app = router(GameStore::new(), registry);

        for games in [0, 3] {
            let response = app
                .clone()
                .oneshot(
                    Request::builder()
                        .method("POST")
                        .uri("/api/experiments")
                        .header("content-type", "application/json")
                        .body(Body::from(
                            serde_json::json!({
                                "engine": "fake",
                                "variant_a": {"label": "A"},
                                "variant_b": {"label": "B"},
                                "games": games,
                            })
                            .to_string(),
                        ))
                        .unwrap(),
                )
                .await
                .unwrap();

            assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        }
    }

    #[tokio::test]
    async fn post_experiments_requires_concurrency_within_the_game_count() {
        let mut registry = EngineRegistry::new();
        registry.insert("fake", fake_engine_spec());
        let app = router(GameStore::new(), registry);

        for concurrency in [0, 5] {
            let response = app
                .clone()
                .oneshot(
                    Request::builder()
                        .method("POST")
                        .uri("/api/experiments")
                        .header("content-type", "application/json")
                        .body(Body::from(
                            serde_json::json!({
                                "engine": "fake",
                                "variant_a": {"label": "A"},
                                "variant_b": {"label": "B"},
                                "games": 4,
                                "concurrency": concurrency,
                            })
                            .to_string(),
                        ))
                        .unwrap(),
                )
                .await
                .unwrap();

            assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        }
    }

    #[tokio::test]
    async fn post_experiments_with_an_unknown_engine_is_400() {
        let app = router(GameStore::new(), EngineRegistry::new());

        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/experiments")
                    .header("content-type", "application/json")
                    .body(Body::from(
                        serde_json::json!({
                            "engine": "no-such-engine",
                            "variant_a": {"label": "A"},
                            "variant_b": {"label": "B"},
                            "games": 2,
                        })
                        .to_string(),
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn get_experiment_with_a_malformed_id_is_400() {
        let app = router(GameStore::new(), EngineRegistry::new());

        let response = app
            .oneshot(
                Request::builder()
                    .uri("/api/experiments/not-a-uuid")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn get_unknown_experiment_is_404() {
        let app = router(GameStore::new(), EngineRegistry::new());

        let response = app
            .oneshot(
                Request::builder()
                    .uri(format!("/api/experiments/{}", Uuid::new_v4()))
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn post_experiments_starts_running_games_that_complete_through_the_http_api() {
        // The fake engine always replies "bestmove e2e4" -- legal only
        // on a game's first move, so every game this experiment runs
        // aborts on its second move. Same reasoning as `experiment`'s
        // own `run_experiment_...` test: this only needs to prove the
        // HTTP layer actually starts and reports on a real experiment
        // end to end, not exercise real chess results -- result
        // tallying itself is already covered at the `experiment`
        // module level.
        let mut registry = EngineRegistry::new();
        registry.insert("fake", fake_engine_spec());
        let app = router(GameStore::new(), registry);

        let created = body_json(
            app.clone()
                .oneshot(
                    Request::builder()
                        .method("POST")
                        .uri("/api/experiments")
                        .header("content-type", "application/json")
                        .body(Body::from(
                            serde_json::json!({
                                "engine": "fake",
                                "variant_a": {"label": "Baseline"},
                                "variant_b": {"label": "Candidate", "options": {"UseTT": false}},
                                "games": 2,
                                "move_time_ms": 20,
                            })
                            .to_string(),
                        ))
                        .unwrap(),
                )
                .await
                .unwrap(),
        )
        .await;
        assert_eq!(created["status"], "running");
        assert_eq!(created["requested_games"], 2);
        assert_eq!(created["completed_games"], 0);
        let id = created["id"].as_str().unwrap().to_string();

        let deadline = tokio::time::Instant::now() + std::time::Duration::from_secs(5);
        loop {
            let snapshot = body_json(
                app.clone()
                    .oneshot(
                        Request::builder()
                            .uri(format!("/api/experiments/{id}"))
                            .body(Body::empty())
                            .unwrap(),
                    )
                    .await
                    .unwrap(),
            )
            .await;
            if snapshot["status"] == "completed" {
                assert_eq!(snapshot["games"].as_array().unwrap().len(), 2);
                assert_eq!(snapshot["label_a"], "Baseline");
                assert_eq!(snapshot["label_b"], "Candidate");
                return;
            }
            assert!(
                tokio::time::Instant::now() < deadline,
                "experiment did not complete in time: {snapshot:?}"
            );
            tokio::time::sleep(std::time::Duration::from_millis(20)).await;
        }
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
        let created = store.create(
            ParticipantInfo::Human,
            ParticipantInfo::Human,
            TimeControl::fixed_move_time(200),
        );
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
