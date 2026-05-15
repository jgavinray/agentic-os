use axum::extract::State;
use axum::http::HeaderMap;
use axum::http::StatusCode;
use axum::response::IntoResponse;
use axum::response::Response;
use bytes::Bytes;
use futures::StreamExt;
use serde_json::Value;
use std::sync::Arc;
use subtle::ConstantTimeEq;

use crate::state::*;
use crate::db;
use crate::qdrant;

// ── Auth helper ────────────────────────────────────────────────

// BUG-8: Constant-time comparison prevents timing-based key recovery.
fn check_auth(state: &AppState, headers: &HeaderMap) -> bool {
    let provided = headers
        .get("Authorization")
        .and_then(|v| v.to_str().ok())
        .and_then(|s| s.strip_prefix("Bearer "))
        .unwrap_or("");

    let expected = state.api_key.as_bytes();
    let provided = provided.as_bytes();

    if expected.len() != provided.len() {
        return false;
    }
    expected.ct_eq(provided).into()
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

    if db::check_ready(&state.pool).await.is_ok() {
        healthy.push("postgres");
    } else {
        unhealthy.push("postgres");
    }

    if http
        .get(&format!("{}/collections", state.qdrant_url))
        .send()
        .await
        .map(|r| r.status().is_success())
        .unwrap_or(false)
    {
        healthy.push("qdrant");
    } else {
        unhealthy.push("qdrant");
    }

    // LITELLM_URL already includes /v1 — no extra path segment needed here.
    if http
        .get(&format!("{}/models", state.litellm_url))
        .send()
        .await
        .map(|r| r.status().is_success() || r.status() == 401)
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

// ── Model listing — BUG-10: proxy to LiteLLM ───────────────────

pub async fn list_models(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
) -> Response {
    if !check_auth(&state, &headers) {
        return (StatusCode::UNAUTHORIZED, axum::Json(serde_json::json!({"error": "unauthorized"}))).into_response();
    }

    let url = format!("{}/models", state.litellm_url);
    match state.http.get(&url).bearer_auth(&state.litellm_key).send().await {
        Ok(r) => {
            let status = r.status();
            match r.json::<Value>().await {
                Ok(v) => (status, axum::Json(v)).into_response(),
                Err(_) => fallback_model_list(&state).into_response(),
            }
        }
        Err(_) => fallback_model_list(&state).into_response(),
    }
}

fn fallback_model_list(state: &AppState) -> axum::Json<Value> {
    axum::Json(serde_json::json!({
        "data": [{"id": state.default_model, "object": "model", "owned_by": "orchestrator"}],
        "object": "list"
    }))
}

// ── Session management ──────────────────────────────────────────

pub async fn start_session(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    axum::Json(req): axum::Json<StartSessionRequest>,
) -> Response {
    if !check_auth(&state, &headers) {
        return (StatusCode::UNAUTHORIZED, axum::Json(serde_json::json!({"error": "unauthorized"}))).into_response();
    }

    let id = match db::start_session_from_request(&state.pool, &req).await {
        Ok(id) => id,
        Err(e) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                axum::Json(serde_json::json!({"error": "failed_to_create_session", "detail": e.to_string()})),
            ).into_response();
        }
    };

    axum::Json(StartSessionResponse { session_id: id }).into_response()
}

// ── Event management ────────────────────────────────────────────

pub async fn append_event(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    axum::Json(req): axum::Json<AppendEventRequest>,
) -> Response {
    if !check_auth(&state, &headers) {
        return (StatusCode::UNAUTHORIZED, axum::Json(serde_json::json!({"error": "unauthorized"}))).into_response();
    }

    let event_id = match db::append_event_from_request(
        &state.pool, &state.http, &state.litellm_url, &state.litellm_key,
        &state.embedding_model, &state.qdrant_url, &req,
    ).await {
        Ok(id) => id,
        Err(e) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                axum::Json(serde_json::json!({"error": "failed_to_append_event", "detail": e.to_string()})),
            ).into_response();
        }
    };

    axum::Json(serde_json::json!({"event_id": event_id})).into_response()
}

// ── Context pack ────────────────────────────────────────────────

pub async fn context_pack(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    axum::Json(req): axum::Json<ContextPackRequest>,
) -> Response {
    if !check_auth(&state, &headers) {
        return (StatusCode::UNAUTHORIZED, axum::Json(serde_json::json!({"error": "unauthorized"}))).into_response();
    }

    let limit = req.limit.unwrap_or(8);
    let events = match db::get_events_for_repo(&state.pool, &req.repo, limit).await {
        Ok(e) => e,
        Err(e) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                axum::Json(serde_json::json!({"error": "failed_to_fetch_context", "detail": e.to_string()})),
            ).into_response();
        }
    };

    let memories: Vec<EventMemory> = events.iter().map(|e| e.to_memory()).collect();
    let search_query = format!("{} {}", req.repo, req.task);
    let semantic_summaries = semantic_search_summaries(&state, &search_query, 5).await;
    let context = db::build_context(&req.repo, &req.task, &memories, &semantic_summaries, 8000);

    axum::Json(ContextPackResponse { repo: req.repo, task: req.task, context, memories }).into_response()
}

// ── Checkpoint ──────────────────────────────────────────────────

pub async fn checkpoint(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    axum::Json(req): axum::Json<CheckpointRequest>,
) -> Response {
    if !check_auth(&state, &headers) {
        return (StatusCode::UNAUTHORIZED, axum::Json(serde_json::json!({"error": "unauthorized"}))).into_response();
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

    let event_id = match db::append_event_from_request(
        &state.pool, &state.http, &state.litellm_url, &state.litellm_key,
        &state.embedding_model, &state.qdrant_url, &event,
    ).await {
        Ok(id) => id,
        Err(e) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                axum::Json(serde_json::json!({"error": "failed_to_checkpoint", "detail": e.to_string()})),
            ).into_response();
        }
    };

    axum::Json(serde_json::json!({"event_id": event_id})).into_response()
}

// ── Chat completions ────────────────────────────────────────────

/// BUG-4: Append context to an existing client system message rather than inserting
/// a new one at position 0, which would demote the harness's carefully-tuned prompt.
fn inject_system_context(payload: &mut Value, context: &str) {
    if let Some(messages) = payload.get_mut("messages").and_then(|v| v.as_array_mut()) {
        if let Some(first) = messages.first_mut() {
            if first.get("role").and_then(|r| r.as_str()) == Some("system") {
                let existing = first["content"].as_str().unwrap_or("").to_string();
                first["content"] = Value::String(format!("{existing}\n\n---\n{context}"));
                return;
            }
        }
        messages.insert(0, serde_json::json!({"role": "system", "content": context}));
    }
}

fn extract_assistant_from_sse(raw: &str) -> String {
    let mut content = String::new();
    for line in raw.lines() {
        if let Some(data) = line.strip_prefix("data: ") {
            if data.trim() == "[DONE]" {
                break;
            }
            if let Ok(v) = serde_json::from_str::<Value>(data) {
                if let Some(delta) = v["choices"][0]["delta"]["content"].as_str() {
                    content.push_str(delta);
                }
            }
        }
    }
    content
}

async fn persist_exchange(state: &AppState, session_id: &str, repo: &str, user_content: &str, assistant_content: &str) {
    let make_req = |event_type: &str, actor: &str, content: &str| AppendEventRequest {
        session_id: session_id.to_string(),
        repo: repo.to_string(),
        actor: Some(actor.to_string()),
        event_type: event_type.to_string(),
        summary: content.chars().take(500).collect(),
        evidence: None,
        metadata: None,
    };

    if let Err(e) = db::append_event_from_request(
        &state.pool, &state.http, &state.litellm_url, &state.litellm_key,
        &state.embedding_model, &state.qdrant_url,
        &make_req("user_message", "user", user_content),
    ).await {
        tracing::warn!("failed to persist user_message: {e}");
    }

    if let Err(e) = db::append_event_from_request(
        &state.pool, &state.http, &state.litellm_url, &state.litellm_key,
        &state.embedding_model, &state.qdrant_url,
        &make_req("assistant_message", "assistant", assistant_content),
    ).await {
        tracing::warn!("failed to persist assistant_message: {e}");
    }
}

async fn semantic_search_summaries(state: &AppState, query: &str, limit: usize) -> Vec<String> {
    qdrant::search(
        &state.http, &state.litellm_url, &state.litellm_key,
        &state.embedding_model, &state.qdrant_url, query, limit,
    )
    .await
    .unwrap_or_default()
    .into_iter()
    .filter_map(|item| {
        item.get("payload")
            .and_then(|p| p.get("summary"))
            .and_then(|s| s.as_str())
            .map(str::to_string)
    })
    .collect()
}

pub async fn chat_completions(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    axum::Json(payload): axum::Json<Value>,
) -> Response {
    if !check_auth(&state, &headers) {
        return (StatusCode::UNAUTHORIZED, axum::Json(serde_json::json!({"error": "unauthorized"}))).into_response();
    }

    let is_stream = payload.get("stream").and_then(|v| v.as_bool()).unwrap_or(false);
    let repo = headers.get("x-agent-repo").and_then(|v| v.to_str().ok()).map(str::to_string);
    let task = headers.get("x-agent-task").and_then(|v| v.to_str().ok()).map(str::to_string);

    // BUG-5: Warn (or reject) when routing headers are absent.
    if repo.is_none() || task.is_none() {
        if state.require_routing_headers {
            return (
                StatusCode::BAD_REQUEST,
                axum::Json(serde_json::json!({
                    "error": "missing_routing_headers",
                    "detail": "x-agent-repo and x-agent-task are required (REQUIRE_ROUTING_HEADERS=true)"
                })),
            ).into_response();
        }
        tracing::warn!(
            missing_repo = repo.is_none(),
            missing_task = task.is_none(),
            "routing headers absent — memory context and persistence skipped"
        );
    }

    let mut req = payload.clone();
    if req.get("model").is_none() {
        req["model"] = Value::String(state.default_model.clone());
    }

    if let (Some(ref r), Some(ref t)) = (&repo, &task) {
        let events = db::get_events_for_repo(&state.pool, r, 8).await.ok();
        let memories: Vec<EventMemory> = events
            .iter()
            .flat_map(|evts| evts.iter().map(|e| e.to_memory()))
            .collect();
        let semantic_summaries = semantic_search_summaries(&state, &format!("{r} {t}"), 5).await;
        let context = db::build_context(r, t, &memories, &semantic_summaries, 8000);
        inject_system_context(&mut req, &context);
    }

    if is_stream {
        return handle_streaming(&state, req, repo, task).await;
    }

    // ── Non-streaming ───────────────────────────────────────────
    let url = format!("{}/chat/completions", state.litellm_url);
    match state.http.post(&url).bearer_auth(&state.litellm_key).json(&req).send().await {
        Ok(r) => match r.json::<Value>().await {
            Ok(val) => {
                // BUG-2: Persist after successful completion.
                if let (Some(ref r), Some(ref t)) = (&repo, &task) {
                    let user_content: String = req["messages"]
                        .as_array()
                        .and_then(|msgs| msgs.iter().rfind(|m| m["role"].as_str() == Some("user")))
                        .and_then(|m| m["content"].as_str())
                        .unwrap_or("")
                        .chars().take(500).collect();
                    let assistant_content: String = val["choices"][0]["message"]["content"]
                        .as_str().unwrap_or("").chars().take(500).collect();

                    match db::find_or_create_session(&state.pool, r, t, "agent").await {
                        Ok(session_id) => persist_exchange(&state, &session_id, r, &user_content, &assistant_content).await,
                        Err(e) => tracing::warn!("find_or_create_session failed: {e}"),
                    }
                }
                axum::Json(val).into_response()
            }
            Err(_) => (StatusCode::BAD_GATEWAY, axum::Json(serde_json::json!({"error": "litellm_parse_error"}))).into_response(),
        },
        Err(e) => (StatusCode::BAD_GATEWAY, axum::Json(serde_json::json!({"error": "litellm_unreachable", "detail": e.to_string()}))).into_response(),
    }
}

/// BUG-3: Pass raw upstream SSE bytes through without re-parsing or re-wrapping.
/// ADD-1: Tap the byte stream via a oneshot to trigger persistence after the stream ends.
async fn handle_streaming(state: &AppState, req: Value, repo: Option<String>, task: Option<String>) -> Response {
    let url = format!("{}/chat/completions", state.litellm_url);

    let upstream = match state.http_stream.post(&url).bearer_auth(&state.litellm_key).json(&req).send().await {
        Ok(r) => r,
        Err(e) => return (
            StatusCode::BAD_GATEWAY,
            axum::Json(serde_json::json!({"error": "litellm_unreachable", "detail": e.to_string()})),
        ).into_response(),
    };

    let bytes_stream = upstream.bytes_stream();

    if let (Some(repo_s), Some(task_s)) = (repo, task) {
        let (done_tx, done_rx) = tokio::sync::oneshot::channel::<Vec<u8>>();
        let accumulated = std::sync::Arc::new(std::sync::Mutex::new(Vec::<u8>::new()));
        let acc_clone = accumulated.clone();

        // Wrap the upstream stream to accumulate bytes on the side.
        let tapped = async_stream::stream! {
            tokio::pin!(bytes_stream);
            let mut tx_opt = Some(done_tx);
            while let Some(chunk) = bytes_stream.next().await {
                match chunk {
                    Ok(b) => {
                        if let Ok(mut g) = acc_clone.lock() { g.extend_from_slice(&b); }
                        yield Ok::<Bytes, std::io::Error>(b);
                    }
                    Err(e) => {
                        yield Err(std::io::Error::new(std::io::ErrorKind::Other, e));
                    }
                }
            }
            // ADD-1: signal stream end so the persistence task can run immediately.
            if let Some(tx) = tx_opt.take() {
                let data = acc_clone.lock().map(|g| g.clone()).unwrap_or_default();
                let _ = tx.send(data);
            }
        };

        let user_content: String = req["messages"]
            .as_array()
            .and_then(|msgs| msgs.iter().rfind(|m| m["role"].as_str() == Some("user")))
            .and_then(|m| m["content"].as_str())
            .unwrap_or("")
            .chars().take(500).collect();

        let state_bg = state.clone();
        tokio::spawn(async move {
            if let Ok(raw_bytes) = done_rx.await {
                let raw = String::from_utf8_lossy(&raw_bytes);
                let assistant_content = extract_assistant_from_sse(&raw);
                match db::find_or_create_session(&state_bg.pool, &repo_s, &task_s, "agent").await {
                    Ok(sid) => persist_exchange(&state_bg, &sid, &repo_s, &user_content, &assistant_content).await,
                    Err(e) => tracing::warn!("stream: find_or_create_session failed: {e}"),
                }
            }
        });

        return (
            [
                (axum::http::header::CONTENT_TYPE, "text/event-stream"),
                (axum::http::header::CACHE_CONTROL, "no-cache"),
                (axum::http::header::CONNECTION, "keep-alive"),
            ],
            axum::body::Body::from_stream(tapped),
        ).into_response();
    }

    // No routing headers — simple pass-through.
    (
        [
            (axum::http::header::CONTENT_TYPE, "text/event-stream"),
            (axum::http::header::CACHE_CONTROL, "no-cache"),
            (axum::http::header::CONNECTION, "keep-alive"),
        ],
        axum::body::Body::from_stream(bytes_stream),
    ).into_response()
}

// ── Semantic search ─────────────────────────────────────────────

pub async fn search(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    axum::Json(req): axum::Json<Value>,
) -> Response {
    if !check_auth(&state, &headers) {
        return (StatusCode::UNAUTHORIZED, axum::Json(serde_json::json!({"error": "unauthorized"}))).into_response();
    }

    let query = req.get("q").and_then(|v| v.as_str()).unwrap_or("");
    let limit = req.get("limit").and_then(|v| v.as_u64()).unwrap_or(10) as usize;

    let results = match qdrant::search(
        &state.http, &state.litellm_url, &state.litellm_key,
        &state.embedding_model, &state.qdrant_url, query, limit,
    ).await {
        Ok(r) => r,
        Err(e) => return (
            StatusCode::INTERNAL_SERVER_ERROR,
            axum::Json(serde_json::json!({"error": "search_failed", "detail": e.to_string()})),
        ).into_response(),
    };

    axum::Json(serde_json::json!({"results": results})).into_response()
}
