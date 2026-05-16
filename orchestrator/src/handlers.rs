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

use crate::anthropic;
use crate::db;
use crate::qdrant;
use crate::state::*;

// ── Auth helpers ───────────────────────────────────────────────

fn bearer_token(headers: &HeaderMap) -> &str {
    headers
        .get("Authorization")
        .and_then(|v| v.to_str().ok())
        .and_then(|s| s.strip_prefix("Bearer "))
        .unwrap_or("")
}

// Returns the (token, namespace) pair if auth passes; None otherwise.
// Constant-time comparison prevents timing-based key recovery.
fn authenticate(state: &AppState, headers: &HeaderMap) -> Option<(String, String)> {
    let provided = bearer_token(headers).as_bytes();
    for (token, namespace) in &state.api_keys {
        let expected = token.as_bytes();
        if expected.len() == provided.len() && expected.ct_eq(provided).into() {
            return Some((token.clone(), namespace.clone()));
        }
    }
    None
}

fn check_auth(state: &AppState, headers: &HeaderMap) -> bool {
    authenticate(state, headers).is_some()
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
        .get(format!("{}/collections", state.qdrant_url))
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
        .get(format!("{}/models", state.litellm_url))
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

pub async fn list_models(State(state): State<Arc<AppState>>, headers: HeaderMap) -> Response {
    if !check_auth(&state, &headers) {
        return (
            StatusCode::UNAUTHORIZED,
            axum::Json(serde_json::json!({"error": "unauthorized"})),
        )
            .into_response();
    }

    let url = format!("{}/models", state.litellm_url);
    match state
        .http
        .get(&url)
        .bearer_auth(&state.litellm_key)
        .send()
        .await
    {
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
        return (
            StatusCode::UNAUTHORIZED,
            axum::Json(serde_json::json!({"error": "unauthorized"})),
        )
            .into_response();
    }

    let (event_id, qdrant_indexed) = match db::append_event_from_request(
        &state.pool,
        &state.http,
        &state.embedding_url,
        &state.qdrant_url,
        &req,
    )
    .await
    {
        Ok(v) => v,
        Err(e) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                axum::Json(
                    serde_json::json!({"error": "failed_to_append_event", "detail": e.to_string()}),
                ),
            )
                .into_response();
        }
    };

    // Detect and record errors from the event.
    if let Some(error_type) = &req.error_type {
        let error_desc = req.error_description.as_deref().unwrap_or("");
        let task = req.task.as_deref().unwrap_or("");
        let severity = "medium";
        if let Err(e) = db::insert_error_record(
            &state.pool,
            &req.repo,
            task,
            error_type,
            error_desc,
            severity,
        )
        .await
        {
            tracing::warn!(%error_type, "failed to record error: {e}");
        }
    }

    axum::Json(serde_json::json!({"event_id": event_id, "qdrant_indexed": qdrant_indexed}))
        .into_response()
}

// ── Semantic / hybrid search helpers ───────────────────────────

async fn semantic_search(
    state: &AppState,
    query: &str,
    limit: usize,
) -> Vec<crate::state::SearchHit> {
    qdrant::search(
        &state.http,
        &state.embedding_url,
        &state.qdrant_url,
        query,
        limit,
    )
    .await
    .unwrap_or_default()
    .into_iter()
    .filter_map(|item| {
        let payload = item.get("payload")?;
        Some(crate::state::SearchHit {
            event_id: payload.get("event_id")?.as_str()?.to_string(),
            event_type: payload.get("event_type")?.as_str()?.to_string(),
            summary: payload.get("summary")?.as_str()?.to_string(),
        })
    })
    .collect()
}

async fn hybrid_search(
    state: &AppState,
    repo: &str,
    task: &str,
    semantic_limit: usize,
) -> Vec<crate::state::SearchHit> {
    let query = format!("{repo} {task}");
    let fts_limit = semantic_limit as i64;

    let (semantic, fts_result) = tokio::join!(
        semantic_search(state, &query, semantic_limit),
        db::search_events_fts(&state.pool, repo, task, fts_limit),
    );

    let fts = fts_result.unwrap_or_else(|e| {
        tracing::warn!("FTS search failed, falling back to semantic-only: {e}");
        vec![]
    });
    crate::hybrid::rrf_merge(&semantic, &fts, 60.0, semantic_limit)
}

// ── Context pack ────────────────────────────────────────────────

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

    let task_category = crate::state::TaskCategory::from_task(&req.task);
    let task_config = crate::state::TaskContextConfig::for_category(task_category);
    let limit = req.limit.unwrap_or(task_config.max_events);

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
    let hybrid_hits = hybrid_search(&state, &req.repo, &req.task, task_config.semantic_limit).await;
    let errors = db::get_active_errors(&state.pool, &req.repo, 5)
        .await
        .unwrap_or_default();
    let context = db::build_context(
        &req.repo,
        &req.task,
        &memories,
        &hybrid_hits,
        &errors,
        task_config.char_budget,
    );

    axum::Json(ContextPackResponse {
        repo: req.repo,
        task: req.task,
        context,
        memories,
    })
    .into_response()
}

// ── Checkpoint ──────────────────────────────────────────────────

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

    let task_string = format!(
        "{}/{}",
        req.repo,
        req.summary.chars().take(50).collect::<String>()
    );
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
        task: Some(task_string),
        error_type: None,
        error_description: None,
    };

    let (event_id, qdrant_indexed) = match db::append_event_from_request(
        &state.pool,
        &state.http,
        &state.embedding_url,
        &state.qdrant_url,
        &event,
    )
    .await
    {
        Ok(v) => v,
        Err(e) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                axum::Json(
                    serde_json::json!({"error": "failed_to_checkpoint", "detail": e.to_string()}),
                ),
            )
                .into_response();
        }
    };

    axum::Json(serde_json::json!({"event_id": event_id, "qdrant_indexed": qdrant_indexed}))
        .into_response()
}

// ── Chat completions ────────────────────────────────────────────

/// Return an Anthropic-shaped error response.
fn anthropic_error(
    status: StatusCode,
    error_type: &'static str,
    message: impl Into<String>,
) -> Response {
    (
        status,
        axum::Json(serde_json::json!({
            "type": "error",
            "error": {"type": error_type, "message": message.into()}
        })),
    )
        .into_response()
}

/// BUG-4: Append context to an existing client system message rather than inserting
/// a new one at position 0, which would demote the harness's carefully-tuned prompt.
/// Ensure the request asks for at least `MIN_MAX_TOKENS` output tokens.
/// Anthropic requires `max_tokens`; OpenAI treats it as optional. We floor
/// whatever the client sent so reasoning models don't get starved mid-response.
fn enforce_min_max_tokens(req: &mut Value) {
    let current = req.get("max_tokens").and_then(|v| v.as_u64()).unwrap_or(0);
    if current < MIN_MAX_TOKENS {
        req["max_tokens"] = Value::from(MIN_MAX_TOKENS);
    }
}

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

async fn persist_exchange(
    state: &AppState,
    session_id: &str,
    repo: &str,
    user_content: &str,
    assistant_content: &str,
) {
    let make_req = |event_type: &str, actor: &str, content: &str| AppendEventRequest {
        session_id: session_id.to_string(),
        repo: repo.to_string(),
        actor: Some(actor.to_string()),
        event_type: event_type.to_string(),
        summary: content.chars().take(500).collect(),
        evidence: None,
        metadata: None,
        task: None,
        error_type: None,
        error_description: None,
    };

    for (event_type, actor, content) in [
        ("user_message", "user", user_content),
        ("assistant_message", "assistant", assistant_content),
    ] {
        let req = make_req(event_type, actor, content);
        for attempt in 0u32..3 {
            match db::append_event_from_request(
                &state.pool,
                &state.http,
                &state.embedding_url,
                &state.qdrant_url,
                &req,
            )
            .await
            {
                Ok((_, false)) => {
                    tracing::warn!("{event_type} stored in postgres but not qdrant-indexed");
                    break;
                }
                Ok(_) => break,
                Err(e) if attempt < 2 => {
                    let delay = tokio::time::Duration::from_millis(200 * 2u64.pow(attempt));
                    tracing::debug!(
                        attempt,
                        "persist {event_type} failed, retrying in {delay:?}: {e}"
                    );
                    tokio::time::sleep(delay).await;
                }
                Err(e) => {
                    tracing::warn!("failed to persist {event_type} after 3 attempts: {e}");
                    break;
                }
            }
        }
    }
}

/// Pack orchestrator context into an OpenAI-shaped request.
/// Sets a default model if absent, fetches memory events, builds context string,
/// and injects it as a system message.
async fn pack_context_into_req(state: &AppState, req: &mut Value, repo: &str, task: &str) {
    if req.get("model").is_none() {
        req["model"] = Value::String(state.default_model.clone());
    }
    let task_category = crate::state::TaskCategory::from_task(task);
    let task_config = crate::state::TaskContextConfig::for_category(task_category);
    let events = db::get_events_for_repo(&state.pool, repo, task_config.max_events)
        .await
        .ok();
    let memories: Vec<EventMemory> = events
        .iter()
        .flat_map(|evts| evts.iter().map(|e| e.to_memory()))
        .collect();
    let hybrid_hits = hybrid_search(state, repo, task, task_config.semantic_limit).await;
    let errors = db::get_active_errors(&state.pool, repo, 5)
        .await
        .unwrap_or_default();
    let context = db::build_context(repo, task, &memories, &hybrid_hits, &errors, task_config.char_budget);
    inject_system_context(req, &context);
}

/// POST the OpenAI request to LiteLLM and return the raw response JSON.
/// Returns Err(Response) on network or parse failure.
async fn dispatch_non_streaming_raw(
    state: &AppState,
    openai_req: &Value,
) -> Result<Value, Response> {
    let url = format!("{}/chat/completions", state.litellm_url);
    match state
        .http
        .post(&url)
        .bearer_auth(&state.litellm_key)
        .json(openai_req)
        .send()
        .await
    {
        Ok(r) => r.json::<Value>().await.map_err(|_| {
            (
                StatusCode::BAD_GATEWAY,
                axum::Json(serde_json::json!({"error": "litellm_parse_error"})),
            )
                .into_response()
        }),
        Err(e) => Err((
            StatusCode::BAD_GATEWAY,
            axum::Json(
                serde_json::json!({"error": "litellm_unreachable", "detail": e.to_string()}),
            ),
        )
            .into_response()),
    }
}

pub async fn chat_completions(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    axum::Json(payload): axum::Json<Value>,
) -> Response {
    let Some((_caller_token, _caller_ns)) = authenticate(&state, &headers) else {
        return (
            StatusCode::UNAUTHORIZED,
            axum::Json(serde_json::json!({"error": "unauthorized"})),
        )
            .into_response();
    };

    let is_stream = payload
        .get("stream")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);

    // Explicit headers take precedence; fall back to token-bound namespace so
    // standard clients (OpenCode, OpenHands, curl) get memory without custom headers.
    let repo = headers
        .get("x-agent-repo")
        .and_then(|v| v.to_str().ok())
        .map(str::to_string)
        .unwrap_or_else(|| _caller_ns.clone());
    let task = headers
        .get("x-agent-task")
        .and_then(|v| v.to_str().ok())
        .map(str::to_string)
        .unwrap_or_else(|| state.default_task.clone());

    tracing::info!(repo = %repo, task = %task, "routing request");

    let mut req = payload.clone();
    // Always route to the configured backend model regardless of what the client sent.
    req["model"] = Value::String(state.default_model.clone());
    enforce_min_max_tokens(&mut req);
    pack_context_into_req(&state, &mut req, &repo, &task).await;

    if is_stream {
        return handle_streaming(&state, req, repo, task).await;
    }

    // ── Non-streaming ───────────────────────────────────────────
    match dispatch_non_streaming_raw(&state, &req).await {
        Ok(val) => {
            let user_content: String = req["messages"]
                .as_array()
                .and_then(|msgs| msgs.iter().rfind(|m| m["role"].as_str() == Some("user")))
                .and_then(|m| m["content"].as_str())
                .unwrap_or("")
                .chars()
                .take(500)
                .collect();
            let assistant_content: String = val["choices"][0]["message"]["content"]
                .as_str()
                .unwrap_or("")
                .chars()
                .take(500)
                .collect();

            match db::find_or_create_session(&state.pool, &repo, &task, "agent").await {
                Ok(session_id) => {
                    persist_exchange(&state, &session_id, &repo, &user_content, &assistant_content)
                        .await
                }
                Err(e) => tracing::warn!("find_or_create_session failed: {e}"),
            }
            axum::Json(val).into_response()
        }
        Err(resp) => resp,
    }
}

async fn handle_streaming(state: &AppState, req: Value, repo: String, task: String) -> Response {
    let url = format!("{}/chat/completions", state.litellm_url);

    let upstream =
        match state
            .http_stream
            .post(&url)
            .bearer_auth(&state.litellm_key)
            .json(&req)
            .send()
            .await
        {
            Ok(r) => r,
            Err(e) => return (
                StatusCode::BAD_GATEWAY,
                axum::Json(
                    serde_json::json!({"error": "litellm_unreachable", "detail": e.to_string()}),
                ),
            )
                .into_response(),
        };

    let bytes_stream = upstream.bytes_stream();
    let (done_tx, done_rx) = tokio::sync::oneshot::channel::<Vec<u8>>();
    let accumulated = std::sync::Arc::new(std::sync::Mutex::new(Vec::<u8>::new()));
    let acc_clone = accumulated.clone();

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
                    yield Err(std::io::Error::other(e));
                }
            }
        }
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
        .chars()
        .take(500)
        .collect();

    let state_bg = state.clone();
    tokio::spawn(async move {
        if let Ok(raw_bytes) = done_rx.await {
            let raw = String::from_utf8_lossy(&raw_bytes);
            let assistant_content = extract_assistant_from_sse(&raw);
            match db::find_or_create_session(&state_bg.pool, &repo, &task, "agent").await {
                Ok(sid) => {
                    persist_exchange(&state_bg, &sid, &repo, &user_content, &assistant_content)
                        .await
                }
                Err(e) => tracing::warn!("stream: find_or_create_session failed: {e}"),
            }
        }
    });

    (
        [
            (axum::http::header::CONTENT_TYPE, "text/event-stream"),
            (axum::http::header::CACHE_CONTROL, "no-cache"),
            (axum::http::header::CONNECTION, "keep-alive"),
        ],
        axum::body::Body::from_stream(tapped),
    )
        .into_response()
}

// ── Anthropic /v1/messages ──────────────────────────────────────

pub async fn messages(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    axum::Json(payload): axum::Json<Value>,
) -> Response {
    let Some((_caller_token, _caller_ns)) = authenticate(&state, &headers) else {
        return anthropic_error(
            StatusCode::UNAUTHORIZED,
            "authentication_error",
            "invalid or missing API key",
        );
    };

    let is_stream = payload
        .get("stream")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);

    let repo = headers
        .get("x-agent-repo")
        .and_then(|v| v.to_str().ok())
        .map(str::to_string)
        .unwrap_or_else(|| _caller_ns.clone());
    let task = headers
        .get("x-agent-task")
        .and_then(|v| v.to_str().ok())
        .map(str::to_string)
        .unwrap_or_else(|| state.default_task.clone());

    tracing::info!(repo = %repo, task = %task, endpoint = "messages", "routing request");

    // Extract user content from the original Anthropic request for persistence.
    let user_content = anthropic::extract_user_content_from_anthropic(&payload);

    // Remember model from original request for response translation.
    let model = payload
        .get("model")
        .and_then(|v| v.as_str())
        .unwrap_or(&state.default_model)
        .to_string();

    // Translate Anthropic → OpenAI.
    let mut openai_req = match anthropic::anthropic_to_openai(payload) {
        Ok(r) => r,
        Err(e) => {
            return anthropic_error(
                StatusCode::from_u16(e.http_status).unwrap_or(StatusCode::BAD_REQUEST),
                e.error_type,
                e.message,
            );
        }
    };

    // Always route to the configured backend model regardless of what the client sent.
    openai_req["model"] = Value::String(state.default_model.clone());
    enforce_min_max_tokens(&mut openai_req);

    // Inject orchestrator memory context.
    pack_context_into_req(&state, &mut openai_req, &repo, &task).await;

    if is_stream {
        return handle_streaming_anthropic(&state, openai_req, user_content, repo, task, model)
            .await;
    }

    // ── Non-streaming ───────────────────────────────────────────
    match dispatch_non_streaming_raw(&state, &openai_req).await {
        Ok(val) => {
            let assistant_content: String = val["choices"][0]["message"]["content"]
                .as_str()
                .unwrap_or("")
                .chars()
                .take(500)
                .collect();

            match db::find_or_create_session(&state.pool, &repo, &task, "agent").await {
                Ok(sid) => {
                    persist_exchange(&state, &sid, &repo, &user_content, &assistant_content).await
                }
                Err(e) => tracing::warn!("messages: find_or_create_session failed: {e}"),
            }

            let anthropic_resp = anthropic::openai_to_anthropic_response(val, &model);
            axum::Json(anthropic_resp).into_response()
        }
        Err(_) => anthropic_error(
            StatusCode::BAD_GATEWAY,
            "api_error",
            "upstream LiteLLM request failed",
        ),
    }
}

async fn handle_streaming_anthropic(
    state: &AppState,
    openai_req: Value,
    user_content: String,
    repo: String,
    task: String,
    model: String,
) -> Response {
    let url = format!("{}/chat/completions", state.litellm_url);

    let upstream = match state
        .http_stream
        .post(&url)
        .bearer_auth(&state.litellm_key)
        .json(&openai_req)
        .send()
        .await
    {
        Ok(r) => r,
        Err(e) => {
            return anthropic_error(
                StatusCode::BAD_GATEWAY,
                "api_error",
                format!("upstream unreachable: {e}"),
            )
        }
    };

    if !upstream.status().is_success() {
        return anthropic_error(
            StatusCode::BAD_GATEWAY,
            "api_error",
            format!("upstream returned {}", upstream.status()),
        );
    }

    let bytes_stream = upstream.bytes_stream();
    let (done_tx, done_rx) = tokio::sync::oneshot::channel::<Vec<u8>>();
    let accumulated = std::sync::Arc::new(std::sync::Mutex::new(Vec::<u8>::new()));
    let acc_clone = accumulated.clone();

    let mut sse_state = anthropic::SseTranslationState::new(model);

    let tapped = async_stream::stream! {
        tokio::pin!(bytes_stream);
        let mut tx_opt = Some(done_tx);
        // Buffer for incomplete lines across byte chunks.
        let mut carry = String::new();

        while let Some(chunk) = bytes_stream.next().await {
            match chunk {
                Ok(b) => {
                    if let Ok(mut g) = acc_clone.lock() { g.extend_from_slice(&b); }
                    carry.push_str(&String::from_utf8_lossy(&b));
                    // Process all complete lines.
                    while let Some(pos) = carry.find('\n') {
                        let line = carry[..pos].trim_end_matches('\r').to_string();
                        carry = carry[pos + 1..].to_string();
                        for event_bytes in anthropic::translate_openai_sse_chunk(&line, &mut sse_state) {
                            yield Ok::<Bytes, std::io::Error>(event_bytes);
                        }
                    }
                }
                Err(e) => {
                    yield Err(std::io::Error::other(e));
                }
            }
        }
        // Flush any remaining carry buffer.
        if !carry.trim().is_empty() {
            for event_bytes in anthropic::translate_openai_sse_chunk(carry.trim(), &mut sse_state) {
                yield Ok::<Bytes, std::io::Error>(event_bytes);
            }
        }
        if let Some(tx) = tx_opt.take() {
            let data = acc_clone.lock().map(|g| g.clone()).unwrap_or_default();
            let _ = tx.send(data);
        }
    };

    let state_bg = state.clone();
    tokio::spawn(async move {
        if let Ok(raw_bytes) = done_rx.await {
            let raw = String::from_utf8_lossy(&raw_bytes);
            let assistant_content = extract_assistant_from_sse(&raw);
            match db::find_or_create_session(&state_bg.pool, &repo, &task, "agent").await {
                Ok(sid) => {
                    persist_exchange(&state_bg, &sid, &repo, &user_content, &assistant_content)
                        .await
                }
                Err(e) => tracing::warn!("messages stream: find_or_create_session failed: {e}"),
            }
        }
    });

    (
        [
            (axum::http::header::CONTENT_TYPE, "text/event-stream"),
            (axum::http::header::CACHE_CONTROL, "no-cache"),
            (axum::http::header::CONNECTION, "keep-alive"),
        ],
        axum::body::Body::from_stream(tapped),
    )
        .into_response()
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    // ── inject_system_context ──────────────────────────────────────────

    #[test]
    fn inject_inserts_system_message_when_none_exists() {
        let mut payload = json!({
            "messages": [{"role": "user", "content": "hello"}]
        });
        inject_system_context(&mut payload, "prior context");
        let msgs = payload["messages"].as_array().unwrap();
        assert_eq!(msgs.len(), 2);
        assert_eq!(msgs[0]["role"], "system");
        assert_eq!(msgs[0]["content"], "prior context");
        assert_eq!(msgs[1]["role"], "user");
    }

    #[test]
    fn inject_appends_to_existing_system_message_not_duplicates_it() {
        let mut payload = json!({
            "messages": [
                {"role": "system", "content": "base prompt"},
                {"role": "user", "content": "hello"}
            ]
        });
        inject_system_context(&mut payload, "prior context");
        let msgs = payload["messages"].as_array().unwrap();
        assert_eq!(msgs.len(), 2, "must not insert a second system message");
        let content = msgs[0]["content"].as_str().unwrap();
        assert!(content.contains("base prompt"));
        assert!(content.contains("prior context"));
    }

    #[test]
    fn inject_keeps_system_message_at_index_zero() {
        let mut payload = json!({
            "messages": [
                {"role": "system", "content": "base prompt"},
                {"role": "user", "content": "hello"}
            ]
        });
        inject_system_context(&mut payload, "context");
        let msgs = payload["messages"].as_array().unwrap();
        assert_eq!(msgs[0]["role"], "system");
    }

    #[test]
    fn inject_is_noop_when_messages_missing() {
        let mut payload = json!({"model": "gpt-4"});
        inject_system_context(&mut payload, "context");
        assert!(payload.get("messages").is_none());
    }

    // ── extract_assistant_from_sse ─────────────────────────────────────

    #[test]
    fn extract_sse_assembles_content_across_chunks() {
        let sse = "data: {\"choices\":[{\"delta\":{\"content\":\"Hello\"}}]}\n\
                   data: {\"choices\":[{\"delta\":{\"content\":\" world\"}}]}\n\
                   data: [DONE]\n";
        assert_eq!(extract_assistant_from_sse(sse), "Hello world");
    }

    #[test]
    fn extract_sse_stops_at_done() {
        let sse = "data: {\"choices\":[{\"delta\":{\"content\":\"before\"}}]}\n\
                   data: [DONE]\n\
                   data: {\"choices\":[{\"delta\":{\"content\":\"after\"}}]}\n";
        assert_eq!(extract_assistant_from_sse(sse), "before");
    }

    #[test]
    fn extract_sse_returns_empty_for_blank_input() {
        assert_eq!(extract_assistant_from_sse(""), "");
    }

    #[test]
    fn extract_sse_ignores_malformed_json_data_lines() {
        let sse = "data: not-valid-json\n\
                   data: {\"choices\":[{\"delta\":{\"content\":\"ok\"}}]}\n\
                   data: [DONE]\n";
        assert_eq!(extract_assistant_from_sse(sse), "ok");
    }

    #[test]
    fn extract_sse_ignores_non_data_lines() {
        let sse = ": keep-alive\n\
                   data: {\"choices\":[{\"delta\":{\"content\":\"hi\"}}]}\n\
                   data: [DONE]\n";
        assert_eq!(extract_assistant_from_sse(sse), "hi");
    }

    #[test]
    fn extract_sse_handles_missing_content_field_gracefully() {
        let sse = "data: {\"choices\":[{\"delta\":{}}]}\n\
                   data: {\"choices\":[{\"delta\":{\"content\":\"real\"}}]}\n\
                   data: [DONE]\n";
        assert_eq!(extract_assistant_from_sse(sse), "real");
    }

    #[test]
    fn retry_backoff_durations_are_correct() {
        let delays: Vec<u64> = (0u32..2).map(|a| 200 * 2u64.pow(a)).collect();
        assert_eq!(delays, vec![200, 400]);
    }

    // ── API_KEYS parsing: semicolon-delimited token,namespace entries ──

    #[test]
    fn parse_simple_single_entry() {
        let input = "agent-os,project-alpha";
        let mut parts = input.splitn(2, ',');
        let token = parts.next().unwrap_or(input).trim().to_string();
        let namespace = parts.next().unwrap_or(&token).trim().to_string();
        assert_eq!(token, "agent-os");
        assert_eq!(namespace, "project-alpha");
    }

    #[test]
    fn parse_multiple_semicolon_entries() {
        let input = "agent-os,project-alpha;agent-os,project-beta;sk-work,work";
        let entries: Vec<(String, String)> = input
            .split(';')
            .map(|s| {
                let s = s.trim();
                let mut parts = s.splitn(2, ',');
                let token = parts.next().unwrap_or(s).trim().to_string();
                let namespace = parts.next().unwrap_or(&token).trim().to_string();
                (token, namespace)
            })
            .filter(|(t, _)| !t.is_empty())
            .collect();

        assert_eq!(entries.len(), 3);
        assert_eq!(entries[0], ("agent-os".to_string(), "project-alpha".to_string()));
        assert_eq!(entries[1], ("agent-os".to_string(), "project-beta".to_string()));
        assert_eq!(entries[2], ("sk-work".to_string(), "work".to_string()));
    }

    #[test]
    fn parse_empty_token_filtered() {
        let input = "token1,ns1;;token2,ns2";
        let entries: Vec<(String, String)> = input
            .split(';')
            .map(|s| {
                let s = s.trim();
                let mut parts = s.splitn(2, ',');
                let token = parts.next().unwrap_or(s).trim().to_string();
                let namespace = parts.next().unwrap_or(&token).trim().to_string();
                (token, namespace)
            })
            .filter(|(t, _)| !t.is_empty())
            .collect();

        assert_eq!(entries.len(), 2);
    }

    #[test]
    fn parse_fallback_to_token_when_no_namespace() {
        let input = "my-token";
        let mut parts = input.splitn(2, ',');
        let token = parts.next().unwrap_or(input).trim().to_string();
        let namespace = parts.next().unwrap_or(&token).trim().to_string();
        assert_eq!(token, "my-token");
        assert_eq!(namespace, "my-token");
    }

    #[test]
    fn parse_default_value() {
        let input = "agent-os,agentic-os";
        let entries: Vec<(String, String)> = input
            .split(';')
            .map(|s| {
                let s = s.trim();
                let mut parts = s.splitn(2, ',');
                let token = parts.next().unwrap_or(s).trim().to_string();
                let namespace = parts.next().unwrap_or(&token).trim().to_string();
                (token, namespace)
            })
            .filter(|(t, _)| !t.is_empty())
            .collect();

        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0], ("agent-os".to_string(), "agentic-os".to_string()));
    }

    // ── Model substitution ────────────────────────────────────────

    #[test]
    fn model_substituted_with_default_in_chat_completions() {
        let mut req = json!({
            "model": "claude-opus-4-7",
            "messages": [{"role": "user", "content": "hi"}]
        });
        let default_model = "qwen36-35b-heretic";
        req["model"] = Value::String(default_model.to_string());
        assert_eq!(req["model"].as_str().unwrap(), default_model);
    }

    #[test]
    fn model_substituted_with_default_for_any_client_model_name() {
        let client_models = ["gpt-4o", "claude-sonnet-4-6", "claude-opus-4-7", "gpt-4-turbo"];
        let default_model = "qwen36-35b-heretic";
        for client_model in client_models {
            let mut req = json!({
                "model": client_model,
                "messages": [{"role": "user", "content": "hi"}]
            });
            req["model"] = Value::String(default_model.to_string());
            assert_eq!(
                req["model"].as_str().unwrap(),
                default_model,
                "client model '{client_model}' was not replaced"
            );
        }
    }
}

// ── Semantic search ─────────────────────────────────────────────

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

    let results = match qdrant::search(
        &state.http,
        &state.embedding_url,
        &state.qdrant_url,
        query,
        limit,
    )
    .await
    {
        Ok(r) => r,
        Err(e) => {
            tracing::warn!("qdrant search failed: {e}");
            return (
                StatusCode::SERVICE_UNAVAILABLE,
                axum::Json(
                    serde_json::json!({"error": "search_unavailable", "detail": e.to_string()}),
                ),
            )
                .into_response();
        }
    };

    axum::Json(serde_json::json!({"results": results})).into_response()
}
