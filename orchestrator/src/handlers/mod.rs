#[cfg(test)]
use axum::http::header;
#[cfg(test)]
use axum::http::{HeaderMap, StatusCode};

#[cfg(test)]
use crate::auth::{provided_api_token, rate_limited_response};
#[cfg(test)]
use crate::context_packing::context_task_category;
#[cfg(test)]
use crate::context_packing_policy::apply_orchestration_context_limits;
#[cfg(test)]
use crate::local_reasoning::local_reasoning_selection;
#[cfg(test)]
use crate::local_reasoning::LocalReasoningPolicy;
#[cfg(test)]
use crate::local_reasoning::LocalReasoningSelection;
#[cfg(test)]
use crate::orchestration_policy;
#[cfg(test)]
use crate::sse::{
    extract_assistant_from_anthropic_sse, extract_assistant_from_sse,
    extract_token_usage_from_anthropic_sse, extract_token_usage_from_sse,
};
#[cfg(test)]
use crate::state::*;
#[cfg(test)]
use crate::system_context::{inject_system_context, inject_system_context_anthropic};
#[cfg(test)]
use crate::telemetry;

pub mod anthropic_completion;
pub mod anthropic_dispatch;
pub mod anthropic_json_upstream;
pub mod anthropic_messages;
pub mod anthropic_persistence;
pub mod anthropic_stream_upstream;
pub mod anthropic_streaming;
pub mod anthropic_upstream;
pub mod capture;
pub mod chat_completions;
pub mod context;
pub mod litellm_attempt;
pub mod openai_dispatch;
pub mod openai_persistence;
pub mod openai_upstream;
pub mod request;
pub mod request_preparation;
pub mod stream_persistence;
pub mod stream_persistence_types;
pub mod streaming;
pub mod trajectory;
pub mod usage;

pub use crate::background::trajectory::run_trajectory_idle_sweep;
pub use crate::context_packing::context_pack;
pub use crate::routes::checkpoints::checkpoint;
pub use crate::routes::context::{context_artifacts, register_context_artifact};
pub use crate::routes::harness::{harness_guardrail, harness_outcome, litellm_callback_payload};
pub use crate::routes::health::{health, health_live, health_ready, list_models};
pub use crate::routes::observability::{cache_stats, metrics, metrics_json};
pub use crate::routes::search::search;
pub use crate::routes::sessions::{append_event, start_session};
pub use crate::routes::tools::authorize_tool;
#[cfg(test)]
pub(crate) use crate::routes::tools::{
    derive_tool_authorization_policy, tool_authorization_classification_text,
};
pub use crate::routes::validations::validations;
pub use anthropic_messages::messages;
pub use chat_completions::chat_completions;

#[cfg(test)]
mod tests {
    pub(crate) use super::*;

    mod auth;
    mod context;
    mod policy;
    mod reasoning;
    mod request_preparation;
    mod sse;
    mod system_context;
}
