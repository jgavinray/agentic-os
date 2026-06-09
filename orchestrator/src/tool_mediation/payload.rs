use serde_json::{json, Value};
use std::collections::BTreeSet;

use crate::tool_mediation_classification::capability_for_tool_name;
use crate::tool_mediation_types::{
    ToolIntent, ToolMenuOutcome, ToolPayloadFormat, ToolSummary, TOOL_MEDIATION_POLICY_VERSION,
};

pub(crate) fn outcome(
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

pub(crate) trait WithHidden {
    fn with_hidden(self, hidden_tools: Vec<ToolSummary>) -> Self;
}

impl WithHidden for ToolMenuOutcome {
    fn with_hidden(mut self, hidden_tools: Vec<ToolSummary>) -> Self {
        self.hidden_tools = hidden_tools;
        self
    }
}

pub(crate) fn tool_summaries(tools: &[Value], format: ToolPayloadFormat) -> Vec<ToolSummary> {
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

pub(crate) fn tool_name(tool: &Value, format: ToolPayloadFormat) -> Option<String> {
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

pub(crate) fn normalize_tool_choice(
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
