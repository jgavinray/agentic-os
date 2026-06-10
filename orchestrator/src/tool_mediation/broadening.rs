//! Trajectory-aware policy broadening.
//!
//! Intent classification reads the operator's request text, but an agent loop
//! re-sends the whole conversation on every turn. Once the trajectory has
//! visibly entered implementation (the history contains edit tool calls), the
//! policy must keep the implement surface available — otherwise a stale or
//! flipped classification mid-loop hides Edit/Bash and wedges the agent.
//!
//! Risk-derived blocks always win: a request under the high-stakes or
//! destructive-command overlay never regains mutation capabilities through
//! this path. Intent-derived read-only blocklists, however, are overridden —
//! observed edits are direct evidence that the read-only classification was
//! wrong for this trajectory.

use serde_json::Value;

use crate::orchestration_policy::{OrchestrationPolicy, RiskPolicy, ToolCapability as PolicyCap};
use crate::tool_mediation_classification::capability_for_tool_name;
use crate::tool_mediation_types::ToolCapability;

/// Capabilities guaranteed once the trajectory has observed edit tool calls.
const IMPLEMENT_SURFACE: &[PolicyCap] = &[
    PolicyCap::RepoRead,
    PolicyCap::FileRead,
    PolicyCap::FileEdit,
    PolicyCap::Validation,
    PolicyCap::ShellRead,
    PolicyCap::GitRead,
];

/// Broaden the policy when the request history shows the assistant already
/// performing file edits. Returns true when any capability was added.
pub fn broaden_policy_for_observed_edits(req: &Value, policy: &mut OrchestrationPolicy) -> bool {
    if !history_contains_edit_tool_use(req) {
        return false;
    }

    let risk_guarded = policy
        .risk_policy
        .contains(&RiskPolicy::HighStakesGuardrail)
        || policy
            .risk_policy
            .contains(&RiskPolicy::DestructiveRequiresConfirmation);

    let mut broadened = false;
    for capability in IMPLEMENT_SURFACE {
        if policy.blocked_tools.contains(capability) {
            if risk_guarded {
                continue;
            }
            policy.blocked_tools.retain(|blocked| blocked != capability);
        }
        if !policy.allowed_tools.contains(capability) {
            policy.allowed_tools.push(*capability);
            broadened = true;
        }
    }
    broadened
}

fn history_contains_edit_tool_use(req: &Value) -> bool {
    let Some(messages) = req.get("messages").and_then(Value::as_array) else {
        return false;
    };
    messages
        .iter()
        .filter(|message| message.get("role").and_then(Value::as_str) == Some("assistant"))
        .any(assistant_used_edit_tool)
}

fn assistant_used_edit_tool(message: &Value) -> bool {
    // Anthropic format: content blocks of type tool_use carry the tool name.
    let anthropic_hit = message
        .get("content")
        .and_then(Value::as_array)
        .map(|blocks| {
            blocks
                .iter()
                .filter(|block| block.get("type").and_then(Value::as_str) == Some("tool_use"))
                .filter_map(|block| block.get("name").and_then(Value::as_str))
                .any(is_edit_tool)
        })
        .unwrap_or(false);
    if anthropic_hit {
        return true;
    }
    // OpenAI format: assistant messages carry tool_calls with function names.
    message
        .get("tool_calls")
        .and_then(Value::as_array)
        .map(|calls| {
            calls
                .iter()
                .filter_map(|call| {
                    call.get("function")
                        .and_then(|function| function.get("name"))
                        .and_then(Value::as_str)
                })
                .any(is_edit_tool)
        })
        .unwrap_or(false)
}

fn is_edit_tool(name: &str) -> bool {
    capability_for_tool_name(name) == ToolCapability::FileEdit
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::request_classification::{
        RequestArtifactType, RequestClassification, RequestIntent, RequestRisk,
    };
    use serde_json::json;

    fn policy_for(intent: RequestIntent, risk: Vec<RequestRisk>) -> OrchestrationPolicy {
        let mut classification = RequestClassification::deterministic(
            format!("event-{}", intent.as_str()),
            "agentic-os".to_string(),
            "session-broadening".to_string(),
            chrono::Utc::now(),
        );
        classification.intent = intent;
        classification.artifact_type = RequestArtifactType::Code;
        classification.risk = risk;
        crate::orchestration_policy::derive_orchestration_policy(
            &classification,
            "representative request",
            false,
        )
    }

    fn anthropic_history_with(tool_name: &str) -> Value {
        json!({
            "messages": [
                {"role": "user", "content": "continue"},
                {"role": "assistant", "content": [
                    {"type": "text", "text": "editing"},
                    {"type": "tool_use", "id": "t1", "name": tool_name, "input": {}}
                ]}
            ]
        })
    }

    #[test]
    fn edit_history_restores_implement_surface_on_read_only_policy() {
        // A trajectory that is already editing was misclassified if the
        // current turn derives a read-only policy; broadening must unwedge it.
        let mut policy = policy_for(RequestIntent::Explain, vec![RequestRisk::None]);
        assert!(policy.blocked_tools.contains(&PolicyCap::FileEdit));

        let broadened =
            broaden_policy_for_observed_edits(&anthropic_history_with("Edit"), &mut policy);

        assert!(broadened);
        assert!(policy.allowed_tools.contains(&PolicyCap::FileEdit));
        assert!(policy.allowed_tools.contains(&PolicyCap::ShellRead));
        assert!(policy.allowed_tools.contains(&PolicyCap::Validation));
        assert!(
            policy.blocked_tools.contains(&PolicyCap::ShellMutation),
            "capabilities outside the implement surface stay blocked"
        );
    }

    #[test]
    fn read_only_history_does_not_broaden() {
        let mut policy = policy_for(RequestIntent::Explain, vec![RequestRisk::None]);

        let broadened =
            broaden_policy_for_observed_edits(&anthropic_history_with("Read"), &mut policy);

        assert!(!broadened);
        assert!(!policy.allowed_tools.contains(&PolicyCap::FileEdit));
    }

    #[test]
    fn high_stakes_risk_blocks_survive_edit_history() {
        let mut policy = policy_for(RequestIntent::Implement, vec![RequestRisk::HighStakes]);
        assert!(policy.blocked_tools.contains(&PolicyCap::FileEdit));

        broaden_policy_for_observed_edits(&anthropic_history_with("Write"), &mut policy);

        assert!(
            !policy.allowed_tools.contains(&PolicyCap::FileEdit),
            "risk-derived blocks must never be overridden by broadening"
        );
        assert!(policy.blocked_tools.contains(&PolicyCap::FileEdit));
    }

    #[test]
    fn openai_tool_calls_history_broadens() {
        let mut policy = policy_for(RequestIntent::Summarize, vec![RequestRisk::None]);
        let req = json!({
            "messages": [
                {"role": "assistant", "content": null, "tool_calls": [
                    {"type": "function", "function": {"name": "Write", "arguments": "{}"}}
                ]}
            ]
        });

        let broadened = broaden_policy_for_observed_edits(&req, &mut policy);

        assert!(broadened);
        assert!(policy.allowed_tools.contains(&PolicyCap::ShellRead));
        assert!(policy.allowed_tools.contains(&PolicyCap::FileEdit));
    }
}
