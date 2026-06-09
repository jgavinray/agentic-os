use std::sync::Arc;

use crate::db;
use crate::state::{AppState, TokenUsage};
use crate::telemetry;

pub(crate) fn record_success_usage(
    state: &Arc<AppState>,
    usage: &TokenUsage,
    requested_model: &str,
    namespace: &str,
    repo: &str,
) {
    telemetry::record_tokens(&state.metrics, usage, &state.default_model);
    if usage.is_empty() {
        return;
    }

    let pool = state.pool.clone();
    let actual = state.default_model.clone();
    let requested_model = requested_model.to_string();
    let namespace = namespace.to_string();
    let repo = repo.to_string();
    let usage = usage.clone();
    tokio::spawn(async move {
        if let Err(e) =
            db::record_token_usage(&pool, &requested_model, &actual, &namespace, &repo, &usage)
                .await
        {
            tracing::warn!("failed to record token usage: {e}");
        }
    });
}
