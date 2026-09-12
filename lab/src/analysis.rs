//! Read-only catalog analysis endpoints. Blocking SQLite work runs off the
//! async runtime; these routes never migrate, import, or spawn an analyzer.

use std::path::PathBuf;

use axum::extract::{Path, Query, State};
use axum::http::StatusCode;
use axum::{routing::get, Json, Router};
use bee_game_catalog::analysis::{AnalysisRun, GamePhase};
use bee_game_catalog::reporting::{self, ReportFilter, ReviewMove, StoredAnalysisReport};
use bee_game_catalog::GameCatalog;
use serde::Deserialize;

type ApiError = (StatusCode, String);

pub fn router(catalog_path: PathBuf) -> Router {
    Router::new()
        .route("/api/analysis/runs", get(runs))
        .route("/api/analysis/runs/{run}", get(report))
        .route("/api/analysis/runs/{run}/moves/{id}", get(review_move))
        .with_state(catalog_path)
}

async fn read<T: Send + 'static>(
    path: PathBuf,
    action: impl FnOnce(&GameCatalog) -> Result<T, ApiError> + Send + 'static,
) -> Result<Json<T>, ApiError> {
    tokio::task::spawn_blocking(move || {
        let catalog = GameCatalog::open_read_only(path).map_err(|e| {
            tracing::warn!("opening analysis catalog: {e}");
            (StatusCode::SERVICE_UNAVAILABLE, "Analysis catalog is unavailable. Check BEE_GAMES_DB and run the current bee-games CLI to prepare it.".into())
        })?;
        action(&catalog).map(Json)
    }).await.map_err(|e| {
        tracing::error!("analysis worker: {e}");
        (StatusCode::INTERNAL_SERVER_ERROR, "Could not read analysis.".into())
    })?
}

fn database_error(error: bee_game_catalog::Error) -> ApiError {
    tracing::error!("reading analysis: {error}");
    (
        StatusCode::INTERNAL_SERVER_ERROR,
        "Could not read analysis.".into(),
    )
}

async fn runs(State(path): State<PathBuf>) -> Result<Json<Vec<AnalysisRun>>, ApiError> {
    if !path.exists() {
        return Ok(Json(vec![]));
    }
    read(path, |c| c.analysis_runs().map_err(database_error)).await
}

#[derive(Debug, Default, Deserialize)]
struct ReportQuery {
    phase: Option<GamePhase>,
    #[serde(default)]
    losses_only: bool,
    over_cp: Option<u32>,
    #[serde(default)]
    offset: usize,
    limit: Option<usize>,
}

async fn report(
    State(path): State<PathBuf>,
    Path(run): Path<i64>,
    Query(query): Query<ReportQuery>,
) -> Result<Json<StoredAnalysisReport>, ApiError> {
    let limit = query.limit.unwrap_or(50);
    if !(1..=200).contains(&limit) {
        return Err((
            StatusCode::BAD_REQUEST,
            "limit must be between 1 and 200".into(),
        ));
    }
    read(path, move |c| {
        reporting::report(
            c,
            run,
            &ReportFilter {
                phase: query.phase,
                losses_only: query.losses_only,
                over_cp: query.over_cp,
                offset: query.offset,
                limit,
            },
        )
        .map_err(database_error)?
        .ok_or((StatusCode::NOT_FOUND, "Analysis run not found.".into()))
    })
    .await
}

async fn review_move(
    State(path): State<PathBuf>,
    Path((run, id)): Path<(i64, i64)>,
) -> Result<Json<ReviewMove>, ApiError> {
    read(path, move |c| {
        reporting::review_move(c, run, id)
            .map_err(database_error)?
            .ok_or((
                StatusCode::NOT_FOUND,
                "Analyzed Bee move not found in this run.".into(),
            ))
    })
    .await
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::{
        body::{to_bytes, Body},
        http::Request,
    };
    use bee_game_catalog::analysis::NewAnalysisRun;
    use tower::ServiceExt;

    async fn request(app: Router, url: &str) -> (StatusCode, serde_json::Value) {
        let response = app
            .oneshot(Request::builder().uri(url).body(Body::empty()).unwrap())
            .await
            .unwrap();
        let status = response.status();
        let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
        (
            status,
            serde_json::from_slice(&body).unwrap_or(serde_json::Value::Null),
        )
    }

    #[tokio::test]
    async fn missing_catalog_is_an_empty_state_and_is_not_created() {
        let path = std::env::temp_dir().join(format!("bee-analysis-api-{}", uuid::Uuid::new_v4()));
        let (status, body) = request(router(path.clone()), "/api/analysis/runs").await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(body, serde_json::json!([]));
        assert!(!path.exists());
    }

    #[tokio::test]
    async fn reads_runs_and_reports_with_validated_filters_and_404s() {
        let path = std::env::temp_dir().join(format!("bee-analysis-api-{}", uuid::Uuid::new_v4()));
        let catalog = GameCatalog::open(&path).unwrap();
        catalog
            .record_analysis_run(&NewAnalysisRun {
                engine: "Stockfish test".into(),
                nodes_per_position: Some(100_000),
                multipv: Some(1),
                schema_version: 2,
                created_at: 0,
                configuration: Some("{}".into()),
            })
            .unwrap();
        drop(catalog);
        let app = router(path.clone());
        let (status, body) = request(app.clone(), "/api/analysis/runs").await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(body[0]["nodes_per_position"], 100_000);
        let (status, body) = request(
            app.clone(),
            "/api/analysis/runs/1?phase=middlegame&over_cp=200&losses_only=true",
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(body["summary"]["games_analyzed"], 0);
        assert_eq!(body["summary"]["avg_cpl"], serde_json::Value::Null);
        for url in [
            "/api/analysis/runs/1?limit=0",
            "/api/analysis/runs/1?limit=201",
            "/api/analysis/runs/1?phase=invalid",
            "/api/analysis/runs/1?over_cp=-1",
        ] {
            assert_eq!(
                request(app.clone(), url).await.0,
                StatusCode::BAD_REQUEST,
                "{url}"
            );
        }
        for url in ["/api/analysis/runs/999", "/api/analysis/runs/1/moves/999"] {
            assert_eq!(request(app.clone(), url).await.0, StatusCode::NOT_FOUND);
        }
        std::fs::remove_file(path).unwrap();
    }
}
