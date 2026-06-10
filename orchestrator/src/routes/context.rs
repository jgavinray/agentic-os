use axum::extract::{Query, State};
use axum::http::{HeaderMap, StatusCode};
use axum::response::{IntoResponse, Response};
use std::sync::Arc;

use crate::auth::check_auth;
use crate::db;
use crate::state::AppState;

#[derive(Debug, serde::Deserialize)]
pub struct ContextArtifactsQuery {
    pub repo: String,
    pub limit: Option<i64>,
    pub ledger_limit: Option<i64>,
}

#[derive(Debug, serde::Deserialize)]
pub struct RegisterContextArtifactRequest {
    pub repo: String,
    /// Artifact content (e.g. an architecture/spec document section).
    pub content: String,
    /// Defaults to "architecture_spec".
    pub artifact_type: Option<String>,
    /// Defaults to the repo scope.
    pub scope: Option<String>,
    /// Stable key so re-registering the same document updates in place.
    pub invalidation_key: Option<String>,
    /// Source file path for provenance (e.g. "docs/architecture.md").
    pub source_path: Option<String>,
}

/// Register an operator-supplied context artifact (spec grounding).
///
/// Active artifacts for a repo are compiled into the stable cacheable system
/// prefix on every request, so registering an architecture document here puts
/// the relevant design truth in front of the model deterministically instead
/// of relying on memory recall.
#[tracing::instrument(name = "handler.register_context_artifact", skip(state, headers, body))]
pub async fn register_context_artifact(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    axum::Json(body): axum::Json<RegisterContextArtifactRequest>,
) -> Response {
    if !check_auth(&state, &headers) {
        return (
            StatusCode::UNAUTHORIZED,
            axum::Json(serde_json::json!({"error": "unauthorized"})),
        )
            .into_response();
    }
    if body.repo.trim().is_empty() || body.content.trim().is_empty() {
        return (
            StatusCode::BAD_REQUEST,
            axum::Json(serde_json::json!({"error": "repo_and_content_required"})),
        )
            .into_response();
    }

    let artifact_type = body.artifact_type.as_deref().unwrap_or("architecture_spec");
    let scope = body.scope.as_deref().unwrap_or("repo");
    let invalidation_key = body
        .invalidation_key
        .clone()
        .or_else(|| body.source_path.clone())
        .unwrap_or_else(|| artifact_type.to_string());
    let source_file_paths = body
        .source_path
        .as_deref()
        .map(|path| serde_json::json!([path]))
        .unwrap_or_else(|| serde_json::json!([]));

    let content = body.content.trim().to_string();
    let artifact = crate::context_artifacts::ContextArtifact::new(
        body.repo.clone(),
        scope,
        artifact_type,
        Some(content.clone()),
        content.clone(),
        content,
        invalidation_key,
        source_file_paths,
    );

    if let Err(e) = db::upsert_context_artifact(&state.pool, &artifact).await {
        return (
            StatusCode::INTERNAL_SERVER_ERROR,
            axum::Json(serde_json::json!({
                "error": "failed_to_register_context_artifact",
                "detail": e.to_string()
            })),
        )
            .into_response();
    }

    axum::Json(serde_json::json!({
        "registered": true,
        "id": artifact.id,
        "repo": artifact.repo,
        "artifact_type": artifact.artifact_type,
        "invalidation_key": artifact.invalidation_key,
        "token_estimate": artifact.token_estimate,
    }))
    .into_response()
}

#[tracing::instrument(name = "handler.context_artifacts", skip(state, headers))]
pub async fn context_artifacts(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    Query(query): Query<ContextArtifactsQuery>,
) -> Response {
    if !check_auth(&state, &headers) {
        return (
            StatusCode::UNAUTHORIZED,
            axum::Json(serde_json::json!({"error": "unauthorized"})),
        )
            .into_response();
    }

    let limit = query.limit.unwrap_or(25).clamp(1, 100);
    let ledger_limit = query.ledger_limit.unwrap_or(50).clamp(1, 200);
    let artifacts = match db::get_context_artifacts_for_repo(&state.pool, &query.repo, limit).await
    {
        Ok(artifacts) => artifacts,
        Err(e) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                axum::Json(serde_json::json!({
                    "error": "failed_to_fetch_context_artifacts",
                    "detail": e.to_string()
                })),
            )
                .into_response();
        }
    };
    let ledger = match db::get_context_compiler_ledger(&state.pool, &query.repo, ledger_limit).await
    {
        Ok(ledger) => ledger,
        Err(e) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                axum::Json(serde_json::json!({
                    "error": "failed_to_fetch_context_compiler_ledger",
                    "detail": e.to_string()
                })),
            )
                .into_response();
        }
    };

    axum::Json(serde_json::json!({
        "repo": query.repo,
        "artifacts": artifacts,
        "ledger": ledger,
    }))
    .into_response()
}
