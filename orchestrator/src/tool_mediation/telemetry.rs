use metrics::counter;

use crate::telemetry::MetricsRegistry;

pub fn record_tool_menu_outcome(
    registry: &MetricsRegistry,
    outcome: &crate::tool_mediation::ToolMenuOutcome,
) {
    let decision = crate::tool_mediation::bounded_decision(outcome.decision);
    let reason = crate::tool_mediation::bounded_reason(outcome.reason);
    counter!(
        "tool_mediation_decisions_total",
        "decision" => decision,
        "reason" => reason
    )
    .increment(1);
    registry.increment_tool_mediation_decisions();
    for capability in outcome.offered_capabilities() {
        record_tool_mediation_tool(registry, "offered", capability);
    }
    for capability in outcome.allowed_capabilities() {
        record_tool_mediation_tool(registry, "allowed", capability);
    }
    for capability in outcome.hidden_capabilities() {
        record_tool_mediation_tool(registry, "hidden", capability);
    }
}

pub fn record_tool_authorization(
    registry: &MetricsRegistry,
    response: &crate::tool_mediation::ToolAuthorizeResponse,
) {
    let decision = crate::tool_mediation::bounded_decision(response.decision);
    let reason = crate::tool_mediation::bounded_reason(response.reason);
    counter!(
        "tool_mediation_decisions_total",
        "decision" => decision,
        "reason" => reason
    )
    .increment(1);
    registry.increment_tool_mediation_decisions();
    let action = if response.decision == "deny" {
        "denied"
    } else {
        "allowed"
    };
    record_tool_mediation_tool(registry, action, response.capability);
}

fn record_tool_mediation_tool(registry: &MetricsRegistry, action: &str, capability: &str) {
    let action = crate::tool_mediation::bounded_tool_action(action);
    let capability = crate::tool_mediation::bounded_capability(capability);
    counter!(
        "tool_mediation_tools_total",
        "action" => action,
        "capability" => capability
    )
    .increment(1);
    registry.increment_tool_mediation_tools();
}
