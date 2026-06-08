use serde_json::Value;
use std::collections::BTreeSet;

use crate::tool_mediation_canonical::hidden_tool_names;
use crate::tool_mediation_classification::detect_tool_intent;
use crate::tool_mediation_payload::{
    normalize_tool_choice, outcome, tool_name, tool_summaries, WithHidden,
};
use crate::tool_mediation_shaping_policy::policy_shaping_for_tools;
use crate::tool_mediation_types::{ToolMenuOutcome, ToolPayloadFormat, ToolSummary};

pub fn shape_openai_request(req: &mut Value, user_content: &str) -> ToolMenuOutcome {
    shape_openai_request_with_policy(req, user_content, None)
}

pub fn shape_openai_request_with_policy(
    req: &mut Value,
    user_content: &str,
    policy: Option<&crate::orchestration_policy::OrchestrationPolicy>,
) -> ToolMenuOutcome {
    shape_request(req, user_content, ToolPayloadFormat::OpenAi, policy)
}

pub fn shape_anthropic_request(req: &mut Value, user_content: &str) -> ToolMenuOutcome {
    shape_anthropic_request_with_policy(req, user_content, None)
}

pub fn shape_anthropic_request_with_policy(
    req: &mut Value,
    user_content: &str,
    policy: Option<&crate::orchestration_policy::OrchestrationPolicy>,
) -> ToolMenuOutcome {
    shape_request(req, user_content, ToolPayloadFormat::Anthropic, policy)
}

fn shape_request(
    req: &mut Value,
    user_content: &str,
    format: ToolPayloadFormat,
    policy: Option<&crate::orchestration_policy::OrchestrationPolicy>,
) -> ToolMenuOutcome {
    let intent = detect_tool_intent(user_content);
    let Some(tools) = req.get_mut("tools").and_then(Value::as_array_mut) else {
        return outcome(format, intent, "pass", "no_tools", vec![], vec![], false);
    };

    let offered = tool_summaries(tools, format);
    if offered.is_empty() {
        return outcome(format, intent, "pass", "no_tools", vec![], vec![], false);
    }

    let hidden_names = hidden_tool_names(intent, &offered);
    let policy_shaping = policy_shaping_for_tools(&offered, &hidden_names, policy);

    let mut all_hidden: BTreeSet<String> = hidden_names.clone();
    all_hidden.extend(policy_shaping.hidden_names.clone());

    if all_hidden.len() == offered.len() && hidden_names.len() == offered.len() {
        tools.retain(|tool| {
            tool_name(tool, format)
                .map(|name| !all_hidden.contains(&name))
                .unwrap_or(true)
        });
        let tool_choice_changed = normalize_tool_choice(req, format, &all_hidden);
        let allowed = offered
            .iter()
            .filter(|tool| !all_hidden.contains(&tool.name))
            .cloned()
            .collect::<Vec<_>>();
        let hidden = offered
            .iter()
            .filter(|tool| all_hidden.contains(&tool.name))
            .cloned()
            .collect::<Vec<_>>();

        return outcome(
            format,
            intent,
            "shape",
            "prefer_canonical_tool",
            offered,
            allowed,
            tool_choice_changed,
        )
        .with_hidden(hidden);
    }

    if policy.is_some() {
        if policy_shaping.all_blocked {
            tools.clear();
            let tool_choice_changed = normalize_tool_choice(req, format, &all_hidden);
            let allowed: Vec<ToolSummary> = vec![];
            let hidden = offered.clone();
            return outcome(
                format,
                intent,
                "shape",
                "policy_filtered_all_tools",
                offered,
                allowed,
                tool_choice_changed,
            )
            .with_hidden(hidden);
        }
    }

    if hidden_names.is_empty() && policy_shaping.policy_only_hidden_count == 0 {
        return outcome(
            format,
            intent,
            "pass",
            "not_applicable",
            offered.clone(),
            offered,
            false,
        );
    }

    tools.retain(|tool| {
        tool_name(tool, format)
            .map(|name| !all_hidden.contains(&name))
            .unwrap_or(true)
    });
    let tool_choice_changed = normalize_tool_choice(req, format, &all_hidden);
    let reason = if policy_shaping.policy_only_hidden_count > 0 {
        "policy_filtered"
    } else {
        "prefer_canonical_tool"
    };

    let allowed = offered
        .iter()
        .filter(|tool| !all_hidden.contains(&tool.name))
        .cloned()
        .collect::<Vec<_>>();
    let hidden = offered
        .iter()
        .filter(|tool| all_hidden.contains(&tool.name))
        .cloned()
        .collect::<Vec<_>>();

    outcome(
        format,
        intent,
        "shape",
        reason,
        offered,
        allowed,
        tool_choice_changed,
    )
    .with_hidden(hidden)
}
