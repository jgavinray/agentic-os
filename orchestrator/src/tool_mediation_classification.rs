use serde_json::{json, Value};

use crate::tool_mediation_types::{ToolCapability, ToolIntent};

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

pub(crate) fn capability_for_tool_name(name: &str) -> ToolCapability {
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

pub(crate) fn command_capability(arguments: &Value) -> ToolCapability {
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
        // Unknown shell means mutation, not "safe shell". This is intentionally
        // conservative: read-only policy must not authorize arbitrary shell just
        // because the command is unfamiliar.
        ToolCapability::ShellMutation
    }
}

fn command_from_arguments(arguments: &Value) -> Option<String> {
    // Different clients serialize shell commands differently. Keep this list
    // deliberately small and explicit so authorization remains predictable.
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

pub(crate) fn replacement_for(
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
        "policy_filtered" => "policy_filtered",
        "policy_would_hide_all_tools" => "policy_would_hide_all_tools",
        "policy_filtered_all_tools" => "policy_filtered_all_tools",
        "policy_blocked_tool" => "policy_blocked_tool",
        "policy_tool_not_allowed" => "policy_tool_not_allowed",
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
        "shell_mutation" => "shell_mutation",
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
