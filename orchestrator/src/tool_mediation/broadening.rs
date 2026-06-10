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
use crate::tool_mediation_classification::{capability_for_tool_name, command_capability};
use crate::tool_mediation_types::ToolCapability;

/// What the request's assistant history shows the trajectory already doing.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct TrajectoryToolEvidence {
    /// An edit-capability tool call appears in the history.
    pub edits_observed: bool,
    /// A validation tool call (named validation tool, or a shell command that
    /// classifies as validation such as `cargo test`) appears in the history.
    pub validation_observed: bool,
    /// Assistant turns that called tools without making any edit. While
    /// `edits_observed` is false this is the discovery-spiral counter: a small
    /// model that keeps reading instead of starting the work.
    pub exploration_turns: usize,
}

/// Capabilities guaranteed once the trajectory has observed edit tool calls.
const IMPLEMENT_SURFACE: &[PolicyCap] = &[
    PolicyCap::RepoRead,
    PolicyCap::FileRead,
    PolicyCap::FileEdit,
    PolicyCap::Validation,
    PolicyCap::ShellRead,
    PolicyCap::GitRead,
];

/// Scan the request's assistant messages for edit and validation tool use.
pub fn trajectory_tool_evidence(req: &Value) -> TrajectoryToolEvidence {
    let mut evidence = TrajectoryToolEvidence::default();
    let Some(messages) = req.get("messages").and_then(Value::as_array) else {
        return evidence;
    };
    for message in messages
        .iter()
        .filter(|message| message.get("role").and_then(Value::as_str) == Some("assistant"))
    {
        let calls = assistant_tool_calls(message);
        let mut message_edited = false;
        for (name, arguments) in &calls {
            match capability_for_tool_name(name) {
                ToolCapability::FileEdit => {
                    evidence.edits_observed = true;
                    message_edited = true;
                }
                ToolCapability::Validation => evidence.validation_observed = true,
                ToolCapability::Shell
                    if command_capability(arguments) == ToolCapability::Validation =>
                {
                    evidence.validation_observed = true;
                }
                _ => {}
            }
        }
        if !calls.is_empty() && !message_edited {
            evidence.exploration_turns += 1;
        }
    }
    evidence
}

/// Broaden the policy when the request history shows the assistant already
/// performing file edits. Returns true when any capability was added.
pub fn broaden_policy_for_observed_edits(
    evidence: &TrajectoryToolEvidence,
    policy: &mut OrchestrationPolicy,
) -> bool {
    if !evidence.edits_observed {
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

/// Collect (tool name, arguments) pairs from one assistant message in either
/// Anthropic (`tool_use` content blocks) or OpenAI (`tool_calls`) format.
fn assistant_tool_calls(message: &Value) -> Vec<(&str, Value)> {
    let mut calls = Vec::new();
    if let Some(blocks) = message.get("content").and_then(Value::as_array) {
        for block in blocks
            .iter()
            .filter(|block| block.get("type").and_then(Value::as_str) == Some("tool_use"))
        {
            if let Some(name) = block.get("name").and_then(Value::as_str) {
                calls.push((name, block.get("input").cloned().unwrap_or(Value::Null)));
            }
        }
    }
    if let Some(tool_calls) = message.get("tool_calls").and_then(Value::as_array) {
        for call in tool_calls {
            let Some(function) = call.get("function") else {
                continue;
            };
            if let Some(name) = function.get("name").and_then(Value::as_str) {
                // OpenAI serializes arguments as a JSON string.
                let arguments = function
                    .get("arguments")
                    .and_then(Value::as_str)
                    .and_then(|raw| serde_json::from_str(raw).ok())
                    .unwrap_or(Value::Null);
                calls.push((name, arguments));
            }
        }
    }
    calls
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

        let evidence = trajectory_tool_evidence(&anthropic_history_with("Edit"));
        let broadened = broaden_policy_for_observed_edits(&evidence, &mut policy);

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

        let evidence = trajectory_tool_evidence(&anthropic_history_with("Read"));
        let broadened = broaden_policy_for_observed_edits(&evidence, &mut policy);

        assert!(!broadened);
        assert!(!policy.allowed_tools.contains(&PolicyCap::FileEdit));
    }

    #[test]
    fn high_stakes_risk_blocks_survive_edit_history() {
        let mut policy = policy_for(RequestIntent::Implement, vec![RequestRisk::HighStakes]);
        assert!(policy.blocked_tools.contains(&PolicyCap::FileEdit));

        let evidence = trajectory_tool_evidence(&anthropic_history_with("Write"));
        broaden_policy_for_observed_edits(&evidence, &mut policy);

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

        let evidence = trajectory_tool_evidence(&req);
        let broadened = broaden_policy_for_observed_edits(&evidence, &mut policy);

        assert!(broadened);
        assert!(policy.allowed_tools.contains(&PolicyCap::ShellRead));
        assert!(policy.allowed_tools.contains(&PolicyCap::FileEdit));
    }

    #[test]
    fn bash_validation_command_counts_as_validation_evidence() {
        let req = json!({
            "messages": [
                {"role": "assistant", "content": [
                    {"type": "tool_use", "id": "t1", "name": "Edit", "input": {"file_path": "src/lib.rs"}},
                    {"type": "tool_use", "id": "t2", "name": "Bash", "input": {"command": "cargo test -p orchestrator"}}
                ]}
            ]
        });

        let evidence = trajectory_tool_evidence(&req);

        assert!(evidence.edits_observed);
        assert!(evidence.validation_observed);
    }

    #[test]
    fn bash_read_command_is_not_validation_evidence() {
        let req = json!({
            "messages": [
                {"role": "assistant", "content": [
                    {"type": "tool_use", "id": "t1", "name": "Bash", "input": {"command": "cat src/lib.rs"}}
                ]}
            ]
        });

        let evidence = trajectory_tool_evidence(&req);

        assert!(!evidence.edits_observed);
        assert!(!evidence.validation_observed);
    }

    #[test]
    fn openai_string_arguments_are_parsed_for_validation_commands() {
        let req = json!({
            "messages": [
                {"role": "assistant", "content": null, "tool_calls": [
                    {"type": "function", "function": {
                        "name": "Bash",
                        "arguments": "{\"command\": \"cargo check --all-targets\"}"
                    }}
                ]}
            ]
        });

        let evidence = trajectory_tool_evidence(&req);

        assert!(evidence.validation_observed);
    }
}
