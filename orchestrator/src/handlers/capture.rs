use std::sync::Arc;

use axum::http::StatusCode;
use serde_json::Value;

use crate::client_capture::RawHttpCapture;
use crate::state::AppState;

pub(crate) async fn record_json_success_capture(
    state: &Arc<AppState>,
    mut capture: RawHttpCapture,
    body: &Value,
) {
    capture.response_status = Some(StatusCode::OK.as_u16() as i32);
    capture.response_headers = Some(serde_json::json!({"content-type": ["application/json"]}));
    capture.raw_response_body = Some(crate::client_capture::to_json_bytes(body));
    crate::client_capture::record_best_effort(state.capture_pool.as_ref(), capture).await;
}
