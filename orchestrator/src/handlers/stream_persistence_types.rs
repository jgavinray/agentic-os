use serde_json::Value;
use std::time::Instant;

use crate::state::AppState;

pub(crate) enum StreamResponseFormat {
    ChatCompletions,
    AnthropicMessages,
}

pub(crate) struct StreamCompletionPersistence {
    pub(crate) state: AppState,
    pub(crate) raw_bytes: Vec<u8>,
    pub(crate) capture: crate::client_capture::RawHttpCapture,
    pub(crate) response_format: StreamResponseFormat,
    pub(crate) requested_model: String,
    pub(crate) namespace: String,
    pub(crate) repo: String,
    pub(crate) task: String,
    pub(crate) user_content: String,
    pub(crate) correlation_id: Option<uuid::Uuid>,
    pub(crate) request_metadata: Option<Value>,
    pub(crate) session_id: Option<String>,
    pub(crate) trajectory: Option<crate::trajectory::TrajectoryContext>,
    pub(crate) request_event_id: Option<uuid::Uuid>,
    pub(crate) context_pack_id: Option<uuid::Uuid>,
    pub(crate) started: Instant,
}
