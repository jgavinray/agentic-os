use axum::http::HeaderMap;
use serde_json::Value;

use crate::state::MAX_MAX_TOKENS;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum LocalReasoningPolicy {
    Low,
    Medium,
    High,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct LocalReasoningSelection {
    pub(crate) policy: LocalReasoningPolicy,
    pub(crate) source: &'static str,
}

impl LocalReasoningPolicy {
    fn as_str(self) -> &'static str {
        match self {
            Self::Low => "low",
            Self::Medium => "medium",
            Self::High => "high",
        }
    }

    fn default_max_tokens(self) -> u64 {
        match self {
            Self::Low => 2048,
            Self::Medium => 4096,
            Self::High => MAX_MAX_TOKENS,
        }
    }

    fn default_temperature(self) -> f64 {
        match self {
            Self::Low => 0.1,
            Self::Medium => 0.2,
            Self::High => 0.2,
        }
    }

    pub(crate) fn system_contract(self) -> &'static str {
        match self {
            Self::Low => {
                "Local harness reasoning policy: low. Answer directly with concrete facts, cite evidence when available, and avoid expanded plans unless required."
            }
            Self::Medium => {
                "Local harness reasoning policy: medium. Use bounded engineering reasoning: objective, evidence, assessment, plan, and verification when the task benefits from structure. Do not add filler."
            }
            Self::High => {
                "Local harness reasoning policy: high. Use a fuller engineering analysis: objective, evidence, constraints, tradeoffs, implementation steps, risks, and verification. Mark unknowns explicitly."
            }
        }
    }
}

fn parse_local_reasoning_policy(value: &str) -> Option<LocalReasoningPolicy> {
    match value.trim().to_ascii_lowercase().as_str() {
        "low" | "fast" => Some(LocalReasoningPolicy::Low),
        "medium" | "normal" | "default" => Some(LocalReasoningPolicy::Medium),
        "high" | "deep" => Some(LocalReasoningPolicy::High),
        _ => None,
    }
}

pub(crate) fn local_reasoning_selection(
    headers: &HeaderMap,
    payload: &Value,
) -> LocalReasoningSelection {
    if let Some(policy) = headers
        .get("x-agent-reasoning-policy")
        .and_then(|value| value.to_str().ok())
        .and_then(parse_local_reasoning_policy)
    {
        return LocalReasoningSelection {
            policy,
            source: "x-agent-reasoning-policy",
        };
    }

    if let Some(policy) = payload
        .get("reasoning_effort")
        .and_then(Value::as_str)
        .and_then(parse_local_reasoning_policy)
    {
        return LocalReasoningSelection {
            policy,
            source: "client_reasoning_effort",
        };
    }

    if let Some(budget) = payload
        .get("thinking")
        .and_then(|thinking| thinking.get("budget_tokens"))
        .and_then(Value::as_u64)
    {
        let policy = if budget <= 2048 {
            LocalReasoningPolicy::Low
        } else if budget <= 8192 {
            LocalReasoningPolicy::Medium
        } else {
            LocalReasoningPolicy::High
        };
        return LocalReasoningSelection {
            policy,
            source: "client_thinking_budget",
        };
    }

    LocalReasoningSelection {
        policy: LocalReasoningPolicy::Medium,
        source: "local_default",
    }
}

pub(crate) fn apply_local_reasoning_defaults(req: &mut Value, selection: LocalReasoningSelection) {
    if req.get("max_tokens").is_none() {
        req["max_tokens"] = Value::from(selection.policy.default_max_tokens());
    }
    if req.get("temperature").is_none() {
        req["temperature"] = serde_json::json!(selection.policy.default_temperature());
    }
}

pub(crate) fn add_local_reasoning_metadata(
    attempt: &mut crate::litellm::LiteLlmCallAttempt,
    selection: LocalReasoningSelection,
) {
    attempt.reasoning_policy = Some(selection.policy.as_str().to_string());
    attempt.reasoning_policy_source = Some(selection.source.to_string());
}
