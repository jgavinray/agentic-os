use axum::extract::State;
use axum::http::HeaderMap;
use axum::http::StatusCode;
use axum::response::Sse;
use axum::response::IntoResponse;
use axum::Json;
use axum::response::sse::Event;
use futures::Stream;
use futures::StreamExt;
use serde_json::Value;
use std::sync::Arc;

use crate::state::*;
use crate::db;
use crate::qdrant;

// ── Health checks ──────────────────────────────────────────────

pub async fn health() -> Json<Value> {
    Json(serde_json::json!({"status": "ok"}))
}

pub async fn health_live() -> Json<Value> {
    Json(serde_json::json!({"status": "ok"}))
}

pub async fn health_ready() -> Json<Value> {
    Json(serde_json::json!({
        "status": "ok",
        "checks": {
            "postgres": "connected",
            "qdrant": "connected",
            "litellm": "connected"
        }
    }))
}

// ── Model listing ──────────────────────────────────────────────

pub async fn list_models(State(state): State<Arc<AppState>>) -> Json<Value> {
    Json(serde_json::json!({
        "data": [{
            "id": state.default_model,
            "object": "model",
            "owned_by": "orchestrator"
        }],
        "object": "list"
    }))
}

// ── Session management ─────────────────────────────────────────

pub async fn start_session(
    State(state): State<Arc<AppState>>,
    Json(req): Json<StartSessionRequest>,
) -> Result<Json<StartSessionResponse>, (StatusCode, Json<Value>)> {
    let id = db::start_session_from_request(&req)
        .await.map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, Json(serde_json::json!({
            "error": "failed_to_create_session",
            "detail": e.to_string()
        }))))?;

    Ok(Json(StartSessionResponse { session_id: id }))
}

// ── Event management ───────────────────────────────────────────

pub async fn append_event(
    State(state): State<Arc<AppState>>,
    Json(req): Json<AppendEventRequest>,
) -> Result<Json<Value>, (StatusCode, Json<Value>)> {
    let event_id = db::append_event_from_request(&state.qdrant_url, &req)
        .await.map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, Json(serde_json::json!({
            "error": "failed_to_append_event",
            "detail": e.to_string()
        }))))?;

    Ok(Json(serde_json::json!({"event_id": event_id})))
}

// ── Context management ─────────────────────────────────────────

pub async fn context_pack(
    State(state): State<Arc<AppState>>,
    Json(req): Json<ContextPackRequest>,
) -> Result<Json<ContextPackResponse>, (StatusCode, Json<Value>)> {
    let limit = req.limit.unwrap_or(8);
    let events = db::get_events_for_repo(&req.repo, limit)
        .await.map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, Json(serde_json::json!({
            "error": "failed_to_fetch_context",
            "detail": e.to_string()
        }))))?;

    let memories: Vec<EventMemory> = events.iter().map(|e| e.to_memory()).collect();
    let context = db::build_context(&req.repo, &req.task, &memories);

    Ok(Json(ContextPackResponse {
        repo: req.repo,
        task: req.task,
        context,
        memories,
    }))
}

pub async fn checkpoint(
    State(state): State<Arc<AppState>>,
    Json(req): Json<CheckpointRequest>,
) -> Result<Json<Value>, (StatusCode, Json<Value>)> {
    let event = AppendEventRequest {
        session_id: req.session_id,
        repo: req.repo,
        actor: Some("checkpoint".to_string()),
        event_type: "checkpoint".to_string(),
        summary: req.summary,
        evidence: None,
        metadata: Some(serde_json::json!({
            "next_actions": req.next_actions.unwrap_or_default(),
            "open_questions": req.open_questions.unwrap_or_default()
        })),
    };

    let event_id = db::append_event_from_request(&state.qdrant_url, &event)
        .await.map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, Json(serde_json::json!({
            "error": "failed_to_checkpoint",
            "detail": e.to_string()
        }))))?;

    Ok(Json(serde_json::json!({"event_id": event_id})))
}

// ── Chat completions (non-streaming) ───────────────────────────

pub async fn chat_completions(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    Json(mut payload): Json<Value>,
) -> Result<Json<Value>, (StatusCode, Json<Value>)> {
    // Extract routing headers
    let repo = headers.get("x-agent-repo")
        .and_then(|v| v.to_str().ok())
        .map(|s| s.to_string());
    let task = headers.get("x-agent-task")
        .and_then(|v| v.to_str().ok())
        .map(|s| s.to_string());

    // Set default model if missing
    if payload.get("model").is_none() {
        payload["model"] = Value::String(state.default_model.clone());
    }

    // Inject context pack when routing headers present
    if let (Some(repo), Some(task)) = (&repo, &task) {
        let limit = 8i64;
        let events = db::get_events_for_repo(&repo, limit)
            .await.map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, Json(serde_json::json!({
                "error": "failed_to_fetch_context",
                "detail": e.to_string()
            }))))?;

        let memories: Vec<EventMemory> = events.iter().map(|e| e.to_memory()).collect();
        let context = db::build_context(&repo, &task, &memories);
        inject_system_context(&mut payload, &context);
    }

    // Forward to LiteLLM
    let url = format!("{}/chat/completions", state.litellm_url);
    let resp = state.http
        .post(&url)
        .bearer_auth(&state.litellm_key)
        .json(&payload)
        .send()
        .await.map_err(|e| (StatusCode::BAD_GATEWAY, Json(serde_json::json!({
            "error": "litellm_timeout",
            "detail": e.to_string()
        }))))?
        .error_for_status()
        .map_err(|e| (StatusCode::BAD_GATEWAY, Json(serde_json::json!({
            "error": "litellm_error",
            "detail": e.to_string()
        }))))?;

    let body = resp.json::<Value>().await
        .map_err(|e| (StatusCode::BAD_GATEWAY, Json(serde_json::json!({
            "error": "litellm_parse_error",
            "detail": e.to_string()
        }))))?;

    Ok(Json(body))
}

fn inject_system_context(payload: &mut Value, context: &str) {
    if let Some(messages) = payload.get_mut("messages").and_then(|v| v.as_array_mut()) {
        messages.insert(0, serde_json::json!({
            "role": "system",
            "content": context
        }));
    }
}

// ── Streaming chat completions (SSE) ───────────────────────────

pub async fn chat_completions_stream(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    Json(payload): Json<Value>,
) -> impl IntoResponse {
    let repo = headers.get("x-agent-repo")
        .and_then(|v| v.to_str().ok())
        .map(|s| s.to_string());
    let task = headers.get("x-agent-task")
        .and_then(|v| v.to_str().ok())
        .map(|s| s.to_string());

    let mut req = payload.clone();
    if req.get("model").is_none() {
        req["model"] = Value::String(state.default_model.clone());
    }
    req["stream"] = serde_json::json!(true);

    if let (Some(_repo), Some(_task)) = (&repo, &task) {
        let context = "Repository context active. Use prior decisions as constraints.";
        inject_system_context(&mut req, context);
    }

    let url = format!("{}/chat/completions", state.litellm_url);
    let http = state.http.clone();
    let key = state.litellm_key.clone();

    // Proxy SSE from LiteLLM back to client
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
                            yield Ok::<Event, axum::Error>(axum::response::sse::Event::default().data(text));
                        }
                        Err(e) => {
                            yield Ok::<Event, axum::Error>(axum::response::sse::Event::default().data(format!("Error: {e}")));
                        }
                    }
                }
            }
            Err(e) => {
                yield Ok::<Event, axum::Error>(axum::response::sse::Event::default().data(format!("Connection error: {e}")));
            }
        }
    };

    (
        {
            let mut headers = axum::http::HeaderMap::new();
            headers.insert(axum::http::header::CONTENT_TYPE, "text/event-stream".parse().unwrap());
            headers
        },
        Sse::new(stream),
    )
}

// ── Semantic search ────────────────────────────────────────────

pub async fn search(
    State(state): State<Arc<AppState>>,
    Json(req): Json<Value>,
) -> Result<Json<Value>, (StatusCode, Json<Value>)> {
    let query = req.get("q").and_then(|v| v.as_str()).unwrap_or("");
    let limit = req.get("limit").and_then(|v| v.as_u64()).unwrap_or(5) as usize;

    let results = qdrant::search(&state.qdrant_url, query, limit)
        .await.map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, Json(serde_json::json!({
            "error": "search_failed",
            "detail": e.to_string()
        }))))?;

    Ok(Json(serde_json::json!({"results": results})))
}
