use regex::Regex;
use serde_json::{json, Value};
use std::sync::OnceLock;

fn regex_cell(cell: &'static OnceLock<Regex>, pattern: &str) -> &'static Regex {
    cell.get_or_init(|| Regex::new(pattern).expect("fingerprint regex must compile"))
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn normalize_validation_payload(
    mut payload: Value,
    validator_name: &str,
    validator_type: &str,
    command: &str,
    success: bool,
    exit_code: Option<i32>,
    duration_ms: Option<u64>,
    failure_excerpt: Option<String>,
) -> Value {
    let Some(obj) = payload.as_object_mut() else {
        return payload;
    };
    obj.entry("validator_name".to_string())
        .or_insert_with(|| json!(validator_name));
    obj.entry("validator_type".to_string())
        .or_insert_with(|| json!(validator_type));
    obj.entry("command".to_string())
        .or_insert_with(|| json!(command));
    obj.insert("success".to_string(), json!(success));
    obj.entry("exit_code".to_string())
        .or_insert_with(|| exit_code.map(Value::from).unwrap_or(Value::Null));
    obj.entry("duration_ms".to_string())
        .or_insert_with(|| duration_ms.map(Value::from).unwrap_or(Value::Null));
    let failure_signature = obj.get("signature").and_then(Value::as_str);
    obj.insert(
        "failure_signature".to_string(),
        failure_signature
            .filter(|signature| *signature != "unknown")
            .map(Value::from)
            .unwrap_or(Value::Null),
    );
    obj.entry("failure_excerpt".to_string()).or_insert_with(|| {
        failure_excerpt
            .filter(|excerpt| !excerpt.is_empty())
            .map(Value::from)
            .unwrap_or(Value::Null)
    });
    payload
}

pub(crate) fn compile_error_count(content: &str) -> usize {
    static RE: OnceLock<Regex> = OnceLock::new();
    regex_cell(&RE, r"(?m)^\s*error(?:\[[A-Za-z0-9]+\])?:")
        .find_iter(content)
        .count()
}

pub(crate) fn warning_count(content: &str) -> usize {
    static RE: OnceLock<Regex> = OnceLock::new();
    regex_cell(&RE, r"(?m)^\s*warning(?:\[.*?\])?:?")
        .find_iter(content)
        .count()
}
