pub(crate) use crate::tool_mediation_shell::{command_capability, replacement_for};
use crate::tool_mediation_types::{ToolCapability, ToolIntent};

pub fn detect_tool_intent(text: &str) -> ToolIntent {
    let lower = text.to_ascii_lowercase();
    if lower.trim().is_empty() {
        return ToolIntent::Unknown;
    }
    if contains_any(
        &lower,
        &[
            // Publishing must be an explicit verb phrase. Bare "commit"/"push"
            // appear incidentally in edit requests ("do not commit") and were
            // flipping the intent.
            "git commit",
            "commit the",
            "commit this",
            "commit my",
            "commit and",
            "git push",
            "push the",
            "push this",
            "push my",
            "pull request",
            "open a pr",
            "create pr",
            "create a pr",
        ],
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
    let normalized = normalize_tool_name(name);
    if matches!(
        normalized.as_str(),
        "read" | "read_file" | "file_read" | "view" | "open_file"
    ) || normalized.ends_with("__read_file")
    {
        ToolCapability::FileRead
    } else if matches!(
        normalized.as_str(),
        "grep"
            | "rg"
            | "ripgrep"
            | "search"
            | "text_search"
            | "find_text"
            | "lsp"
            | "language_server"
            | "workspace_symbol"
            | "find_references"
    ) || normalized.ends_with("__search")
        || normalized.ends_with("__search_code")
        || normalized.ends_with("__search_repositories")
    {
        ToolCapability::TextSearch
    } else if matches!(
        normalized.as_str(),
        "websearch" | "web_search" | "webfetch" | "web_fetch" | "fetch_url" | "search_web"
    ) {
        ToolCapability::WebSearch
    } else if matches!(
        normalized.as_str(),
        "glob" | "ls" | "list" | "list_files" | "file_list" | "find_files"
    ) || normalized.ends_with("__list_files")
        || normalized.ends_with("__get_pull_request_files")
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

/// Normalize a client tool name to lower snake_case so naming styles map to
/// the same capability: `RunCommand`, `run-command`, and `run_command` all
/// normalize to `run_command`. Runs of uppercase stay together (`LSP` → `lsp`).
fn normalize_tool_name(name: &str) -> String {
    let mut normalized = String::with_capacity(name.len() + 4);
    let mut prev_lower_or_digit = false;
    for c in name.chars() {
        if c.is_ascii_uppercase() {
            if prev_lower_or_digit {
                normalized.push('_');
            }
            normalized.push(c.to_ascii_lowercase());
            prev_lower_or_digit = false;
        } else if c == '-' {
            normalized.push('_');
            prev_lower_or_digit = false;
        } else {
            normalized.push(c);
            prev_lower_or_digit = c.is_ascii_lowercase() || c.is_ascii_digit();
        }
    }
    normalized
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
        "web_search" => "web_search",
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
