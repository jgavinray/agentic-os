//! Deterministic tool menu shaping and tool-call authorization.
//!
//! This module is deliberately policy-like but not model-driven: it maps
//! client-provided tools into bounded capabilities, detects simple request/tool
//! intent, shapes upstream tool menus, and authorizes attempted tool calls.

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::BTreeSet;
use uuid::Uuid;

pub const TOOL_MEDIATION_POLICY_VERSION: &str = "deterministic-v1";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ToolPayloadFormat {
    OpenAi,
    Anthropic,
}

impl ToolPayloadFormat {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::OpenAi => "openai",
            Self::Anthropic => "anthropic",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ToolCapability {
    FileRead,
    TextSearch,
    FileList,
    FileEdit,
    Validation,
    Publishing,
    Shell,
    Unknown,
}

impl ToolCapability {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::FileRead => "file_read",
            Self::TextSearch => "text_search",
            Self::FileList => "file_list",
            Self::FileEdit => "file_edit",
            Self::Validation => "validation",
            Self::Publishing => "publishing",
            Self::Shell => "shell",
            Self::Unknown => "unknown",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ToolIntent {
    FileRead,
    TextSearch,
    FileList,
    FileEdit,
    Validation,
    Publishing,
    General,
    Unknown,
}

impl ToolIntent {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::FileRead => "file_read",
            Self::TextSearch => "text_search",
            Self::FileList => "file_list",
            Self::FileEdit => "file_edit",
            Self::Validation => "validation",
            Self::Publishing => "publishing",
            Self::General => "general",
            Self::Unknown => "unknown",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ToolSummary {
    pub name: String,
    pub capability: &'static str,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ToolMenuOutcome {
    pub policy_version: &'static str,
    pub endpoint_format: &'static str,
    pub intent: &'static str,
    pub decision: &'static str,
    pub reason: &'static str,
    pub offered_tools: Vec<ToolSummary>,
    pub allowed_tools: Vec<ToolSummary>,
    pub hidden_tools: Vec<ToolSummary>,
    pub tool_choice_changed: bool,
}

impl ToolMenuOutcome {
    pub fn metadata(&self) -> Value {
        json!({
            "tool_mediation": self,
        })
    }

    pub fn hidden_capabilities(&self) -> impl Iterator<Item = &'static str> + '_ {
        self.hidden_tools.iter().map(|tool| tool.capability)
    }

    pub fn allowed_capabilities(&self) -> impl Iterator<Item = &'static str> + '_ {
        self.allowed_tools.iter().map(|tool| tool.capability)
    }

    pub fn offered_capabilities(&self) -> impl Iterator<Item = &'static str> + '_ {
        self.offered_tools.iter().map(|tool| tool.capability)
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct ToolAuthorizeRequest {
    pub session_id: Option<String>,
    pub repo: Option<String>,
    pub task: Option<String>,
    pub trajectory_id: Option<Uuid>,
    pub attempt_index: Option<i32>,
    pub parent_event_id: Option<Uuid>,
    pub user_intent: Option<String>,
    pub tool_name: String,
    #[serde(default)]
    pub arguments: Value,
    #[serde(default)]
    pub available_tools: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ToolAuthorizeResponse {
    pub policy_version: &'static str,
    pub decision: &'static str,
    pub reason: &'static str,
    pub capability: &'static str,
    pub attempted_tool: String,
    pub preferred_tool: Option<String>,
    pub replacement: Option<Value>,
    pub message: String,
}

impl ToolAuthorizeResponse {
    pub fn metadata(&self, request: &ToolAuthorizeRequest) -> Value {
        json!({
            "tool_authorization": {
                "policy_version": self.policy_version,
                "decision": self.decision,
                "reason": self.reason,
                "capability": self.capability,
                "attempted_tool": self.attempted_tool,
                "preferred_tool": self.preferred_tool,
                "replacement": self.replacement,
                "available_tool_count": request.available_tools.len(),
            }
        })
    }
}

pub fn shape_openai_request(req: &mut Value, user_content: &str) -> ToolMenuOutcome {
    shape_request(req, user_content, ToolPayloadFormat::OpenAi)
}

pub fn shape_anthropic_request(req: &mut Value, user_content: &str) -> ToolMenuOutcome {
    shape_request(req, user_content, ToolPayloadFormat::Anthropic)
}

fn shape_request(
    req: &mut Value,
    user_content: &str,
    format: ToolPayloadFormat,
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
    if hidden_names.is_empty() {
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
            .map(|name| !hidden_names.contains(&name))
            .unwrap_or(true)
    });
    let tool_choice_changed = normalize_tool_choice(req, format, &hidden_names);
    let allowed = offered
        .iter()
        .filter(|tool| !hidden_names.contains(&tool.name))
        .cloned()
        .collect::<Vec<_>>();
    let hidden = offered
        .iter()
        .filter(|tool| hidden_names.contains(&tool.name))
        .cloned()
        .collect::<Vec<_>>();

    outcome(
        format,
        intent,
        "shape",
        "prefer_canonical_tool",
        offered,
        allowed,
        tool_choice_changed,
    )
    .with_hidden(hidden)
}

pub fn authorize_tool_call(req: &ToolAuthorizeRequest, enabled: bool) -> ToolAuthorizeResponse {
    if !enabled {
        return ToolAuthorizeResponse {
            policy_version: TOOL_MEDIATION_POLICY_VERSION,
            decision: "allow",
            reason: "disabled",
            capability: ToolCapability::Unknown.as_str(),
            attempted_tool: req.tool_name.clone(),
            preferred_tool: None,
            replacement: None,
            message: "Tool mediation is disabled; tool call allowed.".to_string(),
        };
    }

    let attempted_capability = capability_for_tool_name(&req.tool_name);
    let command_capability = if attempted_capability == ToolCapability::Shell {
        command_capability(&req.arguments)
    } else {
        attempted_capability
    };

    if attempted_capability == ToolCapability::Shell {
        if let Some(preferred) =
            preferred_tool_for_capability(command_capability, &req.available_tools)
        {
            let replacement = replacement_for(command_capability, &preferred, &req.arguments);
            return ToolAuthorizeResponse {
                policy_version: TOOL_MEDIATION_POLICY_VERSION,
                decision: "deny",
                reason: "prefer_canonical_tool",
                capability: command_capability.as_str(),
                attempted_tool: req.tool_name.clone(),
                preferred_tool: Some(preferred.clone()),
                replacement,
                message: format!(
                    "Use the {preferred} tool for {} instead of shell fallback commands.",
                    command_capability.as_str()
                ),
            };
        }
    }

    ToolAuthorizeResponse {
        policy_version: TOOL_MEDIATION_POLICY_VERSION,
        decision: "allow",
        reason: "not_applicable",
        capability: command_capability.as_str(),
        attempted_tool: req.tool_name.clone(),
        preferred_tool: None,
        replacement: None,
        message: "Tool call allowed.".to_string(),
    }
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
        ToolCapability::FileRead | ToolCapability::TextSearch | ToolCapability::FileList => offered
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

fn preferred_tool_for_capability(
    capability: ToolCapability,
    available_tools: &[String],
) -> Option<String> {
    if matches!(capability, ToolCapability::Shell | ToolCapability::Unknown) {
        return None;
    }
    available_tools
        .iter()
        .find(|name| capability_for_tool_name(name) == capability)
        .cloned()
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

pub fn detect_tool_intent(text: &str) -> ToolIntent {
    let lower = text.to_ascii_lowercase();
    if lower.trim().is_empty() {
        return ToolIntent::Unknown;
    }
    if contains_any(
        &lower,
        &["commit", "push", "pull request", "open a pr", "create pr"],
    ) {
        ToolIntent::Publishing
    } else if contains_any(
        &lower,
        &[
            "run test",
            "run the test",
            "cargo test",
            "npm test",
            "lint",
            "typecheck",
            "type check",
        ],
    ) {
        ToolIntent::Validation
    } else if contains_any(
        &lower,
        &[
            "edit ", "modify ", "change ", "update ", "fix ", "patch ", "write to",
        ],
    ) {
        ToolIntent::FileEdit
    } else if contains_any(
        &lower,
        &[
            "grep",
            "search",
            "find usages",
            "find references",
            "look for",
        ],
    ) {
        ToolIntent::TextSearch
    } else if contains_any(
        &lower,
        &["list files", "show files", "tree", "directory listing"],
    ) {
        ToolIntent::FileList
    } else if (contains_any(
        &lower,
        &[
            "read ", "show me", "show the", "open ", "view ", "inspect ", "look at", "cat ",
        ],
    ) && (looks_like_file_reference(&lower) || lower.contains("file")))
        || looks_like_file_reference(&lower)
            && contains_any(&lower, &["what is in", "what's in", "contents of"])
    {
        ToolIntent::FileRead
    } else {
        ToolIntent::General
    }
}

fn capability_for_tool_name(name: &str) -> ToolCapability {
    let lower = name.to_ascii_lowercase();
    let normalized = lower.replace('-', "_");
    if matches!(
        normalized.as_str(),
        "read" | "read_file" | "file_read" | "view" | "open_file"
    ) || normalized.ends_with("__read_file")
    {
        ToolCapability::FileRead
    } else if matches!(
        normalized.as_str(),
        "grep" | "rg" | "ripgrep" | "search" | "text_search" | "find_text"
    ) || normalized.ends_with("__search")
    {
        ToolCapability::TextSearch
    } else if matches!(
        normalized.as_str(),
        "glob" | "ls" | "list" | "list_files" | "file_list" | "find_files"
    ) || normalized.ends_with("__list_files")
    {
        ToolCapability::FileList
    } else if matches!(
        normalized.as_str(),
        "edit" | "write" | "write_file" | "apply_patch" | "patch" | "str_replace_editor"
    ) || normalized.ends_with("__write_file")
    {
        ToolCapability::FileEdit
    } else if matches!(
        normalized.as_str(),
        "test" | "run_tests" | "lint" | "typecheck" | "type_check" | "compile"
    ) {
        ToolCapability::Validation
    } else if matches!(
        normalized.as_str(),
        "git" | "commit" | "push" | "pull_request" | "pr" | "create_pr"
    ) {
        ToolCapability::Publishing
    } else if matches!(
        normalized.as_str(),
        "bash" | "shell" | "terminal" | "run_command" | "execute_command"
    ) {
        ToolCapability::Shell
    } else {
        ToolCapability::Unknown
    }
}

fn command_capability(arguments: &Value) -> ToolCapability {
    let Some(command) = command_from_arguments(arguments) else {
        return ToolCapability::Shell;
    };
    let lower = command.trim().to_ascii_lowercase();
    if is_shell_file_read(&lower) {
        ToolCapability::FileRead
    } else if is_shell_text_search(&lower) {
        ToolCapability::TextSearch
    } else if is_shell_file_list(&lower) {
        ToolCapability::FileList
    } else {
        ToolCapability::Shell
    }
}

fn command_from_arguments(arguments: &Value) -> Option<String> {
    arguments
        .as_str()
        .map(str::to_string)
        .or_else(|| {
            arguments
                .get("command")
                .and_then(Value::as_str)
                .map(str::to_string)
        })
        .or_else(|| {
            arguments
                .get("cmd")
                .and_then(Value::as_str)
                .map(str::to_string)
        })
        .or_else(|| {
            arguments
                .get("input")
                .and_then(|input| input.get("command"))
                .and_then(Value::as_str)
                .map(str::to_string)
        })
}

fn is_shell_file_read(command: &str) -> bool {
    starts_with_command(command, "cat")
        || starts_with_command(command, "head")
        || starts_with_command(command, "tail")
        || command.starts_with("sed -n ")
        || command.starts_with("awk ")
}

fn is_shell_text_search(command: &str) -> bool {
    starts_with_command(command, "grep")
        || starts_with_command(command, "rg")
        || starts_with_command(command, "ripgrep")
}

fn is_shell_file_list(command: &str) -> bool {
    starts_with_command(command, "ls")
        || starts_with_command(command, "find")
        || starts_with_command(command, "tree")
}

fn starts_with_command(command: &str, binary: &str) -> bool {
    command == binary || command.starts_with(&format!("{binary} "))
}

fn replacement_for(
    capability: ToolCapability,
    preferred_tool: &str,
    arguments: &Value,
) -> Option<Value> {
    match capability {
        ToolCapability::FileRead => command_from_arguments(arguments)
            .and_then(|command| path_from_read_command(&command))
            .map(|path| json!({"tool_name": preferred_tool, "arguments": {"path": path}})),
        _ => None,
    }
}

fn path_from_read_command(command: &str) -> Option<String> {
    let parts = command
        .split_whitespace()
        .map(|part| part.trim_matches(|c| c == '\'' || c == '"'))
        .filter(|part| {
            !part.is_empty()
                && !part.starts_with('-')
                && *part != "cat"
                && *part != "head"
                && *part != "tail"
                && *part != "sed"
                && *part != "awk"
                && !is_sed_range(part)
        })
        .collect::<Vec<_>>();
    parts.last().map(|part| (*part).to_string())
}

fn is_sed_range(part: &str) -> bool {
    part.ends_with('p')
        && part
            .trim_end_matches('p')
            .chars()
            .all(|c| c.is_ascii_digit() || c == ',' || c == '$')
}

fn looks_like_file_reference(value: &str) -> bool {
    value.contains('/')
        || value.contains(".rs")
        || value.contains(".ts")
        || value.contains(".js")
        || value.contains(".py")
        || value.contains(".md")
        || value.contains(".json")
        || value.contains(".yaml")
        || value.contains(".yml")
        || value.contains(".toml")
}

fn contains_any(value: &str, needles: &[&str]) -> bool {
    needles.iter().any(|needle| value.contains(needle))
}

pub fn bounded_decision(value: &str) -> &'static str {
    match value {
        "allow" => "allow",
        "deny" => "deny",
        "shape" => "shape",
        "pass" => "pass",
        _ => "unknown",
    }
}

pub fn bounded_reason(value: &str) -> &'static str {
    match value {
        "prefer_canonical_tool" => "prefer_canonical_tool",
        "no_tools" => "no_tools",
        "not_applicable" => "not_applicable",
        "disabled" => "disabled",
        _ => "unknown",
    }
}

pub fn bounded_capability(value: &str) -> &'static str {
    match value {
        "file_read" => "file_read",
        "text_search" => "text_search",
        "file_list" => "file_list",
        "file_edit" => "file_edit",
        "validation" => "validation",
        "publishing" => "publishing",
        "shell" => "shell",
        "unknown" => "unknown",
        _ => "unknown",
    }
}

pub fn bounded_tool_action(value: &str) -> &'static str {
    match value {
        "offered" => "offered",
        "allowed" => "allowed",
        "hidden" => "hidden",
        "denied" => "denied",
        _ => "unknown",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn openai_file_read_hides_shell_when_read_exists() {
        let mut req = json!({
            "messages": [{"role": "user", "content": "Read README.md"}],
            "tools": [
                {"type": "function", "function": {"name": "Read"}},
                {"type": "function", "function": {"name": "Bash"}}
            ],
            "tool_choice": {"type": "function", "function": {"name": "Bash"}}
        });

        let outcome = shape_openai_request(&mut req, "Read README.md");

        assert_eq!(outcome.decision, "shape");
        assert_eq!(outcome.reason, "prefer_canonical_tool");
        assert_eq!(outcome.hidden_tools[0].name, "Bash");
        assert_eq!(req["tools"].as_array().unwrap().len(), 1);
        assert_eq!(req["tools"][0]["function"]["name"], "Read");
        assert_eq!(req["tool_choice"], "auto");
    }

    #[test]
    fn anthropic_file_read_hides_shell_when_read_exists() {
        let mut req = json!({
            "messages": [],
            "tools": [
                {"name": "read", "input_schema": {}},
                {"name": "bash", "input_schema": {}}
            ],
            "tool_choice": {"type": "tool", "name": "bash"}
        });

        let outcome = shape_anthropic_request(&mut req, "show the file docs/TRAJECTORIES.md");

        assert_eq!(outcome.decision, "shape");
        assert_eq!(outcome.hidden_tools[0].name, "bash");
        assert_eq!(req["tools"].as_array().unwrap().len(), 1);
        assert_eq!(req["tools"][0]["name"], "read");
        assert_eq!(req["tool_choice"], json!({"type": "auto"}));
    }

    #[test]
    fn shell_stays_available_when_no_canonical_tool_exists() {
        let mut req = json!({
            "messages": [],
            "tools": [
                {"type": "function", "function": {"name": "Bash"}}
            ]
        });

        let outcome = shape_openai_request(&mut req, "Read README.md");

        assert_eq!(outcome.decision, "pass");
        assert_eq!(outcome.reason, "not_applicable");
        assert_eq!(req["tools"].as_array().unwrap().len(), 1);
    }

    #[test]
    fn authorization_denies_shell_file_read_with_read_available() {
        let req = ToolAuthorizeRequest {
            session_id: None,
            repo: None,
            task: None,
            trajectory_id: None,
            attempt_index: None,
            parent_event_id: None,
            user_intent: Some("read README.md".to_string()),
            tool_name: "Bash".to_string(),
            arguments: json!({"command": "cat README.md"}),
            available_tools: vec!["Read".to_string(), "Bash".to_string()],
        };

        let response = authorize_tool_call(&req, true);

        assert_eq!(response.decision, "deny");
        assert_eq!(response.reason, "prefer_canonical_tool");
        assert_eq!(response.preferred_tool.as_deref(), Some("Read"));
        assert_eq!(
            response.replacement.unwrap(),
            json!({"tool_name": "Read", "arguments": {"path": "README.md"}})
        );
    }

    #[test]
    fn authorization_allows_shell_file_read_without_read_available() {
        let req = ToolAuthorizeRequest {
            session_id: None,
            repo: None,
            task: None,
            trajectory_id: None,
            attempt_index: None,
            parent_event_id: None,
            user_intent: Some("read README.md".to_string()),
            tool_name: "Bash".to_string(),
            arguments: json!({"command": "cat README.md"}),
            available_tools: vec!["Bash".to_string()],
        };

        let response = authorize_tool_call(&req, true);

        assert_eq!(response.decision, "allow");
        assert_eq!(response.reason, "not_applicable");
    }
}
