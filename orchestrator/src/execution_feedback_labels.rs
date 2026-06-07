use serde_json::Value;

use crate::execution_feedback_fingerprints::signature_category_from_signature;
use crate::execution_feedback_validators::classify_validator;

pub fn bounded_validator_label(name: &str) -> &'static str {
    classify_validator(name, "")
        .map(|spec| spec.validator)
        .unwrap_or("other")
}

pub fn bounded_failure_signature_label(signature: &str) -> String {
    if signature == "unknown" {
        return "unknown".to_string();
    }
    signature
        .chars()
        .filter(|c| c.is_ascii_alphanumeric() || matches!(c, ':' | '-' | '_'))
        .take(64)
        .collect()
}

pub fn bounded_failure_category_label(category: &str) -> &'static str {
    match category {
        "borrow_checker" => "borrow_checker",
        "import_error" => "import_error",
        "type_error" => "type_error",
        "parse_error" => "parse_error",
        "unknown" => "unknown",
        "none" => "none",
        _ => "other",
    }
}

pub fn retry_trigger_category_from_signature(signature: Option<&str>) -> &'static str {
    signature
        .map(signature_category_from_signature)
        .unwrap_or("none")
}

pub fn retry_trigger_category_from_payload(payload: &Value) -> &'static str {
    match payload.get("signature_category").and_then(Value::as_str) {
        Some("borrow_checker") => "borrow_checker",
        Some("import_error") => "import_error",
        Some("type_error") => "type_error",
        Some("parse_error") => "parse_error",
        Some("unknown") => "unknown",
        Some("none") => "none",
        Some(_) => "unknown",
        None => {
            retry_trigger_category_from_signature(payload.get("signature").and_then(Value::as_str))
        }
    }
}

pub fn task_retry_type(task: &str) -> &'static str {
    let task = task.to_ascii_lowercase();
    const INFRA: &[&str] = &[
        "deploy",
        "docker",
        "kubernetes",
        "kubectl",
        "terraform",
        "infra",
        "infrastructure",
        "migration",
        "postgres",
        "database",
        "ci",
    ];
    const RECALL: &[&str] = &[
        "context",
        "memory",
        "retrieval",
        "recall",
        "summar",
        "search",
        "history",
    ];
    const CODING: &[&str] = &[
        "code",
        "compile",
        "test",
        "lint",
        "rust",
        "python",
        "typescript",
        "bug",
        "fix",
        "patch",
        "refactor",
        "implement",
    ];

    if INFRA.iter().any(|keyword| task.contains(keyword)) {
        "infra"
    } else if RECALL.iter().any(|keyword| task.contains(keyword)) {
        "recall"
    } else if CODING.iter().any(|keyword| task.contains(keyword)) {
        "coding"
    } else {
        "general"
    }
}
