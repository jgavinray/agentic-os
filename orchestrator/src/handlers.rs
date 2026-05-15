use axum::extract::State;
use axum::http::HeaderMap;
use axum::http::StatusCode;
use axum::response::Sse;
use axum::response::IntoResponse;
use axum::response::sse::Event;
use axum::response::Response;
use futures::Stream;
use futures::StreamExt;
use serde_json::Value;
use std::sync::Arc;

use crate::state::*;
use crate::db;
use crate::qdrant;

// ── Auth helper ────────────────────────────────────────────────

fn check_auth(state: &AppState, headers: &HeaderMap) -> bool {
    headers
        .get("Authorization")
        .and_then(|v| v.to_str().ok())
        .and_then(|s| s.strip_prefix("Bearer "))
        .map(|token| token == state.api_key)
        .unwrap_or(false)
}

// ── Health checks (no auth) ─────────────────────────────────────

pub async fn health() -> axum::Json<Value> {
    axum::Json(serde_json::json!({"status": "ok"}))
}

pub async fn health_live() -> axum::Json<Value> {
    axum::Json(serde_json::json!({"status": "ok"}))
}

pub async fn health_ready(
    State(state): State<Arc<AppState>>,
) -> Result<axum::Json<Value>, (StatusCode, axum::Json<Value>)> {
    let http = state.http.clone();
    let mut healthy = Vec::new();
    let mut unhealthy = Vec::new();

    // Check Postgres
    if http.get("http://localhost:5432").send().await.is_ok() {
        healthy.push("postgres");
    } else {
        unhealthy.push("postgres");
    }

    // Check Qdrant
    if http
        .get(&format!("{}/api/v1/collections", state.qdrant_url))
        .send()
        .await
        .map(|r| r.status().is_success())
        .unwrap_or(false)
    {
        healthy.push("qdrant");
    } else {
        unhealthy.push("qdrant");
    }

    // Check LiteLLM
    if http
        .get(&format!("{}/models", state.litellm_url))
        .send()
        .await
        .map(|r| r.status().is_success())
        .unwrap_or(false)
    {
        healthy.push("litellm");
    } else {
        unhealthy.push("litellm");
    }

    if !unhealthy.is_empty() {
        return Err((
            StatusCode::SERVICE_UNAVAILABLE,
            axum::Json(serde_json::json!({
                "status": "degraded",
                "healthy": healthy,
                "unhealthy": unhealthy,
            })),
        ));
    }

    Ok(axum::Json(serde_json::json!({
        "status": "ready",
        "services": healthy,
    })))
}

// ── Model listing (auth required) ───────────────────────────────

pub async fn list_models(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
) -> Response {
    if !check_auth(&state, &headers) {
        return (
            StatusCode::UNAUTHORIZED,
            axum::Json(serde_json::json!({"error": "unauthorized"})),
        )
            .into_response();
    }

    axum::Json(serde_json::json!({
        "data": [{
            "id": state.default_model,
            "object": "model",
            "owned_by": "orchestrator",
        }],
        "object": "list",
    }))
    .into_response()
}

// ── Session management (auth required) ──────────────────────────

pub async fn start_session(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    axum::Json(req): axum::Json<StartSessionRequest>,
) -> Response {
    if !check_auth(&state, &headers) {
        return (
            StatusCode::UNAUTHORIZED,
            axum::Json(serde_json::json!({"error": "unauthorized"})),
        )
            .into_response();
    }

    let id = match db::start_session_from_request(&state.pool, &req).await {
        Ok(id) => id,
        Err(e) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                axum::Json(serde_json::json!({"error": "failed_to_create_session", "detail": e.to_string()})),
            )
                .into_response();
        }
    };

    axum::Json(StartSessionResponse { session_id: id }).into_response()
}

// ── Event management (auth required) ────────────────────────────

pub async fn append_event(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    axum::Json(req): axum::Json<AppendEventRequest>,
) -> Response {
    if !check_auth(&state, &headers) {
        return (
            StatusCode::UNAUTHORIZED,
            axum::Json(serde_json::json!({"error": "unauthorized"})),
        )
            .into_response();
    }

    let event_id = match db::append_event_from_request(&state.pool, &state.qdrant_url, &req).await {
        Ok(id) => id,
        Err(e) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                axum::Json(serde_json::json!({"error": "failed_to_append_event", "detail": e.to_string()})),
            )
                .into_response();
        }
    };

    axum::Json(serde_json::json!({"event_id": event_id})).into_response()
}

// ── Context pack (auth required) ────────────────────────────────

pub async fn context_pack(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    axum::Json(req): axum::Json<ContextPackRequest>,
) -> Response {
    if !check_auth(&state, &headers) {
        return (
            StatusCode::UNAUTHORIZED,
            axum::Json(serde_json::json!({"error": "unauthorized"})),
        )
            .into_response();
    }

    let limit = req.limit.unwrap_or(8);

    let events = match db::get_events_for_repo(&state.pool, &req.repo, limit).await {
        Ok(events) => events,
        Err(e) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                axum::Json(serde_json::json!({"error": "failed_to_fetch_context", "detail": e.to_string()})),
            )
                .into_response();
        }
    };

    let memories: Vec<EventMemory> = events.iter().map(|e| e.to_memory()).collect();

    // Semantic retrieval from Qdrant
    let search_query = format!("{} {}", req.repo, req.task);
    let semantic_results = qdrant::search(&state.qdrant_url, &search_query, 5)
        .await
        .ok()
        .and_then(|v| v.get("results").and_then(|r| r.as_array()).map(|arr| arr.to_vec()))
        .unwrap_or_default();

    // Inject semantic memories into context pack
    let semantic_context = if !semantic_results.is_empty() {
        let semantic_entries: Vec<String> = semantic_results.iter()
            .filter_map(|item| item.get("payload").and_then(|p| p.get("summary").and_then(|s| s.as_str())))
            .map(|s| format!("- [semantic] {}", s))
            .collect();
        semantic_entries.join("\n")
    } else {
        String::new()
    };

    let context = db::build_context(&req.repo, &req.task, &memories);

    // Append semantic context if available
    let full_context = if !semantic_context.is_empty() {
        format!("{}\n\nSemantic recall:\n{}", context, semantic_context)
    } else {
        context
    };

    axum::Json(ContextPackResponse {
        repo: req.repo,
        task: req.task,
        context: full_context,
        memories,
    })
    .into_response()
}

// ── Checkpoint (auth required) ──────────────────────────────────

pub async fn checkpoint(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    axum::Json(req): axum::Json<CheckpointRequest>,
) -> Response {
    if !check_auth(&state, &headers) {
        return (
            StatusCode::UNAUTHORIZED,
            axum::Json(serde_json::json!({"error": "unauthorized"})),
        )
            .into_response();
    }

    let event = AppendEventRequest {
        session_id: req.session_id,
        repo: req.repo,
        actor: Some("checkpoint".to_string()),
        event_type: "checkpoint".to_string(),
        summary: req.summary,
        evidence: None,
        metadata: Some(serde_json::json!({
            "next_actions": req.next_actions.unwrap_or_default(),
            "open_questions": req.open_questions.unwrap_or_default(),
        })),
    };

    let event_id = match db::append_event_from_request(&state.pool, &state.qdrant_url, &event).await {
        Ok(id) => id,
        Err(e) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                axum::Json(serde_json::json!({"error": "failed_to_checkpoint", "detail": e.to_string()})),
            )
                .into_response();
        }
    };

    axum::Json(serde_json::json!({"event_id": event_id})).into_response()
}

// ── Chat completions (auth required, streaming + non-streaming) ─

fn inject_system_context(payload: &mut Value, context: &str) {
    if let Some(messages) = payload.get_mut("messages").and_then(|v| v.as_array_mut()) {
        messages.insert(
            0,
            serde_json::json!({"role": "system", "content": context}),
        );
    }
}

pub async fn chat_completions(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    axum::Json(payload): axum::Json<Value>,
) -> Response {
    // Auth check
    if !check_auth(&state, &headers) {
        return (
            StatusCode::UNAUTHORIZED,
            axum::Json(serde_json::json!({"error": "unauthorized"})),
        )
            .into_response();
    }

    let is_stream = payload.get("stream").and_then(|v| v.as_bool()).unwrap_or(false);

    // Extract routing headers
    let repo = headers.get("x-agent-repo").and_then(|v| v.to_str().ok()).map(str::to_string);
    let task = headers.get("x-agent-task").and_then(|v| v.to_str().ok()).map(str::to_string);

    let mut req = payload.clone();

    // Set default model if missing
    if req.get("model").is_none() {
        req["model"] = Value::String(state.default_model.clone());
    }

    if is_stream {
        // Streaming path: proxy SSE from LiteLLM
        req["stream"] = serde_json::json!(true);

        if let (Some(_repo), Some(_task)) = (&repo, &task) {
            inject_system_context(
                &mut req,
                "Repository context active. Use prior decisions as constraints.",
            );
        }

        let url = format!("{}/chat/completions", state.litellm_url);
        let http = state.http.clone();
        let key = state.litellm_key.clone();

        let stream = async_stream::stream! {
            let resp = http
                .post(&url)
                .bearer_auth(&key)
                .json(&req)
                .send()
                .await;

            match resp {
                Ok(resp) => {
                    let bytes_stream = resp.bytes_stream();
                    let mut stream = bytes_stream;
                    while let Some(chunk) = stream.next().await {
                        match chunk {
                            Ok(bytes) => {
                                let text = String::from_utf8_lossy(&bytes);
                                yield Ok::<Event, axum::Error>(Event::default().data(text));
                            }
                            Err(e) => {
                                yield Ok::<Event, axum::Error>(Event::default().data(format!("Error: {}", e)));
                            }
                        }
                    }
                }
                Err(e) => {
                    yield Ok::<Event, axum::Error>(Event::default().data(format!("Connection error: {}", e)));
                }
            }
        };

        return (
            {
                let mut headers = axum::http::HeaderMap::new();
                headers.insert(
                    axum::http::header::CONTENT_TYPE,
                    "text/event-stream".parse().unwrap(),
                );
                headers
            },
            Sse::new(stream),
        ).into_response();
    }

    // Non-streaming path: inject context from semantic search + Postgres
    if let (Some(ref repo), Some(ref task)) = (&repo, &task) {
        let limit = 8i64;
        let events = db::get_events_for_repo(&state.pool, &repo, limit).await.ok();

        let search_query = format!("{} {}", &repo, &task);
        let semantic_results = qdrant::search(&state.qdrant_url, &search_query, 5)
            .await
            .ok()
            .and_then(|v| v.get("results").and_then(|r| r.as_array()).map(|arr| arr.to_vec()))
            .unwrap_or_default();

        let memories: Vec<EventMemory> = events
            .iter()
            .flat_map(|e| e.iter().map(|m| m.to_memory()).collect::<Vec<_>>())
            .collect();

        let context = db::build_context(&repo, &task, &memories);

        let semantic_context: String = semantic_results.iter()
            .filter_map(|item| item.get("payload").and_then(|p| p.get("summary").and_then(|s| s.as_str())))
            .map(|s| format!("- [semantic] {}", s))
            .collect::<Vec<_>>()
            .join("\n");

        let full_context = if !semantic_context.is_empty() {
            format!("{}\n\nSemantic recall:\n{}", context, semantic_context)
        } else {
            context
        };

        inject_system_context(&mut req, &full_context);
    }

    // Forward to LiteLLM
    let url = format!("{}/chat/completions", state.litellm_url);
    let http = state.http.clone();
    let key = state.litellm_key.clone();

    let resp = http
        .post(&url)
        .bearer_auth(&key)
        .json(&req)
        .send()
        .await;

    match resp {
        Ok(r) => {
            let text = r.text().await;
            let val: Value = match text {
                Ok(t) => match serde_json::from_str(&t) {
                    Ok(v) => v,
                    Err(_) => return (StatusCode::BAD_GATEWAY, axum::Json(serde_json::json!({"error": "litellm_parse_error", "detail": "failed_to_parse_litellm_response"}))).into_response(),
                },
                Err(_) => return (StatusCode::BAD_GATEWAY, axum::Json(serde_json::json!({"error": "litellm_text_error", "detail": "failed_to_read_litellm_response_body"}))).into_response(),
            };
            axum::Json(val).into_response()
        }
        Err(e) => (
            StatusCode::BAD_GATEWAY,
            axum::Json(serde_json::json!({"error": "litellm_timeout", "detail": e.to_string()})),
        )
            .into_response(),
    }
}

// ── Semantic search (auth required) ─────────────────────────────

pub async fn search(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    axum::Json(req): axum::Json<Value>,
) -> Response {
    if !check_auth(&state, &headers) {
        return (
            StatusCode::UNAUTHORIZED,
            axum::Json(serde_json::json!({"error": "unauthorized"})),
        )
            .into_response();
    }

    let query = req.get("q").and_then(|v| v.as_str()).unwrap_or("");
    let limit = req.get("limit").and_then(|v| v.as_u64()).unwrap_or(10) as usize;

    let results = match qdrant::search(&state.qdrant_url, query, limit).await {
        Ok(r) => r,
        Err(e) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                axum::Json(serde_json::json!({"error": "search_failed", "detail": e.to_string()})),
            )
                .into_response();
        }
    };

    axum::Json(serde_json::json!({"results": results})).into_response()
}
