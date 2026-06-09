use serde_json::{json, Value};

use crate::tool_mediation_types::ToolCapability;

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
