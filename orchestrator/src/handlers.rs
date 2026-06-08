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

pub use crate::background::trajectory::run_trajectory_idle_sweep;
pub use crate::context_packing::context_pack;
pub use crate::handlers_anthropic_messages::messages;
pub use crate::handlers_chat_completions::chat_completions;
pub use crate::routes::checkpoints::checkpoint;
pub use crate::routes::context::context_artifacts;
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

#[cfg(test)]
#[path = "handlers_auth_tests.rs"]
mod handlers_auth_tests;

#[cfg(test)]
#[path = "handlers_policy_tests.rs"]
mod handlers_policy_tests;

#[cfg(test)]
#[path = "handlers_reasoning_tests.rs"]
mod handlers_reasoning_tests;

#[cfg(test)]
#[path = "handlers_request_preparation_tests.rs"]
mod handlers_request_preparation_tests;

#[cfg(test)]
#[path = "handlers_sse_tests.rs"]
mod handlers_sse_tests;

#[cfg(test)]
#[path = "handlers_context_tests.rs"]
mod handlers_context_tests;

#[cfg(test)]
#[path = "handlers_system_context_tests.rs"]
mod handlers_system_context_tests;
