use serde_json::{json, Value};
use std::collections::BTreeSet;

use crate::tool_mediation::{policy_allows_tool_capability, policy_blocks_tool_capability};
use crate::tool_mediation_classification::{capability_for_tool_name, detect_tool_intent};
use crate::tool_mediation_types::{
    ToolCapability, ToolIntent, ToolMenuOutcome, ToolPayloadFormat, ToolSummary,
    TOOL_MEDIATION_POLICY_VERSION,
};

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
    let policy_hidden_names: BTreeSet<String> = if let Some(p) = policy {
        offered
            .iter()
            .filter(|tool| {
                let cap = capability_for_tool_name(&tool.name);
                policy_blocks_tool_capability(p, cap) || !policy_allows_tool_capability(p, cap)
            })
            .map(|tool| tool.name.clone())
            .collect()
    } else {
        BTreeSet::new()
    };

    let mut all_hidden: BTreeSet<String> = hidden_names.clone();
    all_hidden.extend(policy_hidden_names);

    let policy_hidden_count = if let Some(p) = policy {
        offered
            .iter()
            .filter(|tool| {
                let cap = capability_for_tool_name(&tool.name);
                !hidden_names.contains(&tool.name)
                    && (policy_blocks_tool_capability(p, cap)
                        || !policy_allows_tool_capability(p, cap))
            })
            .count()
    } else {
        0
    };

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

    if let Some(p) = policy {
        let all_blocked_by_policy = offered.iter().all(|tool| {
            let cap = capability_for_tool_name(&tool.name);
            policy_blocks_tool_capability(p, cap) || !policy_allows_tool_capability(p, cap)
        });
        if all_blocked_by_policy {
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

    if hidden_names.is_empty() && policy_hidden_count == 0 {
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
    let reason = if policy_hidden_count > 0 {
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

fn outcome(
    format: ToolPayloadFormat,
    intent: ToolIntent,
    decision: &'static str,
    reason: &'static str,
    offered_tools: Vec<ToolSummary>,
    allowed_tools: Vec<ToolSummary>,
    tool_choice_changed: bool,
) -> ToolMenuOutcome {
    ToolMenuOutcome {
        policy_version: TOOL_MEDIATION_POLICY_VERSION,
        endpoint_format: format.as_str(),
        intent: intent.as_str(),
        decision,
        reason,
        offered_tools,
        allowed_tools,
        hidden_tools: vec![],
        tool_choice_changed,
    }
}

trait WithHidden {
    fn with_hidden(self, hidden_tools: Vec<ToolSummary>) -> Self;
}

impl WithHidden for ToolMenuOutcome {
    fn with_hidden(mut self, hidden_tools: Vec<ToolSummary>) -> Self {
        self.hidden_tools = hidden_tools;
        self
    }
}

fn hidden_tool_names(intent: ToolIntent, offered: &[ToolSummary]) -> BTreeSet<String> {
    let Some(canonical) = canonical_capability_for_intent(intent) else {
        return BTreeSet::new();
    };
    if !offered
        .iter()
        .any(|tool| tool.capability == canonical.as_str())
    {
        return BTreeSet::new();
    }

    match canonical {
        ToolCapability::FileRead
        | ToolCapability::TextSearch
        | ToolCapability::FileList
        | ToolCapability::ShellMutation => offered
            .iter()
            .filter(|tool| tool.capability == ToolCapability::Shell.as_str())
            .map(|tool| tool.name.clone())
            .collect(),
        _ => BTreeSet::new(),
    }
}

fn canonical_capability_for_intent(intent: ToolIntent) -> Option<ToolCapability> {
    match intent {
        ToolIntent::FileRead => Some(ToolCapability::FileRead),
        ToolIntent::TextSearch => Some(ToolCapability::TextSearch),
        ToolIntent::FileList => Some(ToolCapability::FileList),
        ToolIntent::FileEdit => Some(ToolCapability::FileEdit),
        ToolIntent::Validation => Some(ToolCapability::Validation),
        ToolIntent::Publishing => Some(ToolCapability::Publishing),
        ToolIntent::General | ToolIntent::Unknown => None,
    }
}

fn tool_summaries(tools: &[Value], format: ToolPayloadFormat) -> Vec<ToolSummary> {
    tools
        .iter()
        .filter_map(|tool| {
            tool_name(tool, format).map(|name| ToolSummary {
                capability: capability_for_tool_name(&name).as_str(),
                name,
            })
        })
        .collect()
}

fn tool_name(tool: &Value, format: ToolPayloadFormat) -> Option<String> {
    match format {
        ToolPayloadFormat::OpenAi => tool
            .get("function")
            .and_then(|function| function.get("name"))
            .and_then(Value::as_str)
            .or_else(|| tool.get("name").and_then(Value::as_str))
            .map(str::to_string),
        ToolPayloadFormat::Anthropic => {
            tool.get("name").and_then(Value::as_str).map(str::to_string)
        }
    }
}

fn normalize_tool_choice(
    req: &mut Value,
    format: ToolPayloadFormat,
    hidden_names: &BTreeSet<String>,
) -> bool {
    let Some(choice) = req.get("tool_choice") else {
        return false;
    };
    let hidden = match format {
        ToolPayloadFormat::OpenAi => {
            if let Some(name) = choice
                .get("function")
                .and_then(|function| function.get("name"))
                .and_then(Value::as_str)
            {
                hidden_names.contains(name)
            } else {
                choice
                    .as_str()
                    .map(|name| hidden_names.contains(name))
                    .unwrap_or(false)
            }
        }
        ToolPayloadFormat::Anthropic => choice
            .get("name")
            .and_then(Value::as_str)
            .map(|name| hidden_names.contains(name))
            .unwrap_or(false),
    };
    if !hidden {
        return false;
    }

    req["tool_choice"] = match format {
        ToolPayloadFormat::OpenAi => Value::String("auto".to_string()),
        ToolPayloadFormat::Anthropic => json!({"type": "auto"}),
    };
    true
}
