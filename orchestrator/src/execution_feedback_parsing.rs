use crate::execution_feedback_fingerprints::summarize_text;
use regex::Regex;
use serde_json::Value;
use std::sync::OnceLock;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CapturedToolResult {
    pub tool_name: String,
    pub content: String,
    pub exit_code: i32,
    pub duration_ms: u64,
    pub stdout_summary: String,
    pub stderr_summary: String,
}

fn regex_cell(cell: &'static OnceLock<Regex>, pattern: &str) -> &'static Regex {
    cell.get_or_init(|| Regex::new(pattern).expect("execution feedback regex must compile"))
}

pub(crate) fn exit_code_from_text(text: &str) -> Option<i32> {
    static RE: OnceLock<Regex> = OnceLock::new();
    let re = regex_cell(
        &RE,
        r"(?i)\b(?:exit code|exited with code|status)\s*[:=]?\s*([0-9]+)\b",
    );
    re.captures(text)
        .and_then(|caps| caps.get(1))
        .and_then(|m| m.as_str().parse().ok())
}

pub(crate) fn infer_success(exit_code: i32, content: &str) -> bool {
    if exit_code != 0 {
        return false;
    }
    let lower = content.to_ascii_lowercase();
    !(lower.contains("failed") || lower.contains("error:") || lower.contains("traceback"))
        || lower.contains("0 failed")
}

pub(crate) fn first_number_before(label: &str, text: &str) -> Option<u64> {
    static RE: OnceLock<Regex> = OnceLock::new();
    let re = regex_cell(
        &RE,
        r"(?i)\b([0-9]+)\s+(passed|failed|skipped|ignored|total|errors?|warnings?)\b",
    );
    re.captures_iter(text).find_map(|caps| {
        let found = caps.get(2)?.as_str().to_ascii_lowercase();
        let wanted = label.to_ascii_lowercase();
        (found == wanted || found.trim_end_matches('s') == wanted.trim_end_matches('s'))
            .then(|| caps.get(1)?.as_str().parse().ok())
            .flatten()
    })
}

pub(crate) fn failure_summaries(content: &str) -> Vec<String> {
    content
        .lines()
        .map(str::trim)
        .filter(|line| {
            let lower = line.to_ascii_lowercase();
            lower.contains("fail")
                || lower.contains("error")
                || lower.contains("panic")
                || lower.contains("traceback")
        })
        .take(5)
        .map(|line| line.chars().take(240).collect())
        .collect()
}

pub fn tool_results_from_value(value: &Value) -> Vec<CapturedToolResult> {
    let mut out = Vec::new();
    collect_tool_results(value, &mut out);
    out
}

fn collect_tool_results(value: &Value, out: &mut Vec<CapturedToolResult>) {
    match value {
        Value::Object(map) => {
            // OpenAI-style tool responses are usually `role=tool`; Anthropic
            // content blocks usually use `type=tool_result`. Recursing below
            // lets us handle either shape nested inside provider envelopes.
            let role_tool = map.get("role").and_then(Value::as_str) == Some("tool");
            let typed_tool_result = map.get("type").and_then(Value::as_str) == Some("tool_result");
            if role_tool || typed_tool_result {
                let tool_name = map
                    .get("tool_name")
                    .or_else(|| map.get("name"))
                    .or_else(|| map.get("tool"))
                    .and_then(Value::as_str)
                    .unwrap_or("unknown")
                    .to_string();
                let content = extract_content_text(value);
                let exit_code = map
                    .get("exit_code")
                    .and_then(Value::as_i64)
                    .map(|v| v as i32)
                    .or_else(|| exit_code_from_text(&content))
                    .unwrap_or_else(|| {
                        // Some clients do not report exit_code. Keep a weak,
                        // deterministic fallback so obvious tool errors are not
                        // silently recorded as successful.
                        if content.to_ascii_lowercase().contains("error") {
                            1
                        } else {
                            0
                        }
                    });
                out.push(CapturedToolResult {
                    tool_name,
                    stdout_summary: summarize_text(
                        map.get("stdout")
                            .and_then(Value::as_str)
                            .unwrap_or(&content),
                        500,
                    ),
                    stderr_summary: summarize_text(
                        map.get("stderr").and_then(Value::as_str).unwrap_or(""),
                        500,
                    ),
                    duration_ms: map.get("duration_ms").and_then(Value::as_u64).unwrap_or(0),
                    content,
                    exit_code,
                });
            }

            for value in map.values() {
                collect_tool_results(value, out);
            }
        }
        Value::Array(values) => {
            for value in values {
                collect_tool_results(value, out);
            }
        }
        _ => {}
    }
}

fn extract_content_text(value: &Value) -> String {
    match value {
        Value::String(s) => s.clone(),
        Value::Array(values) => values
            .iter()
            .map(extract_content_text)
            .filter(|s| !s.is_empty())
            .collect::<Vec<_>>()
            .join("\n"),
        Value::Object(map) => {
            if let Some(s) = map
                .get("content")
                .or_else(|| map.get("text"))
                .or_else(|| map.get("result"))
                .or_else(|| map.get("output"))
                .and_then(Value::as_str)
            {
                return s.to_string();
            }
            if map.get("type").and_then(Value::as_str) == Some("text") {
                return map
                    .get("text")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_string();
            }
            String::new()
        }
        _ => String::new(),
    }
}

pub fn tool_results_from_sse(raw: &str) -> Vec<CapturedToolResult> {
    let mut out = Vec::new();
    for line in raw.lines() {
        let Some(data) = line.strip_prefix("data: ") else {
            continue;
        };
        if data.trim() == "[DONE]" {
            break;
        }
        if let Ok(value) = serde_json::from_str::<Value>(data) {
            out.extend(tool_results_from_value(&value));
        }
    }
    out
}
