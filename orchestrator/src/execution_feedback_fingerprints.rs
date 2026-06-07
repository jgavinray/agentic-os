use regex::Regex;
use serde_json::{json, Value};
use std::collections::BTreeSet;
use std::sync::OnceLock;

pub const FINGERPRINT_VERSION: u32 = 1;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FailureFingerprint {
    pub signature: String,
    pub category: String,
    pub rule: String,
    pub raw_excerpt: String,
    pub version: u32,
}

struct FingerprintRule {
    name: &'static str,
    category: &'static str,
    apply: fn(&str) -> Option<String>,
}

// Regex compilation is cached because this code runs on request-adjacent paths.
// The rules are deterministic, so the first successful match wins every time.
fn regex_cell(cell: &'static OnceLock<Regex>, pattern: &str) -> &'static Regex {
    cell.get_or_init(|| Regex::new(pattern).expect("fingerprint regex must compile"))
}

fn rust_borrow_checker(text: &str) -> Option<String> {
    static RE: OnceLock<Regex> = OnceLock::new();
    let re = regex_cell(&RE, r"\berror\[E0(382|497|501|506|507)\]");
    re.is_match(text).then(|| "rust:borrow-checker".to_string())
}

fn rust_type_mismatch(text: &str) -> Option<String> {
    static RE: OnceLock<Regex> = OnceLock::new();
    let re = regex_cell(&RE, r"\berror\[E0(308|278|302|305)\]");
    re.is_match(text).then(|| "rust:type-mismatch".to_string())
}

fn python_import_error(text: &str) -> Option<String> {
    static RE: OnceLock<Regex> = OnceLock::new();
    let re = regex_cell(&RE, r"\b(ModuleNotFoundError|ImportError)\b");
    re.is_match(text).then(|| "python:import-error".to_string())
}

fn typescript_type_error(text: &str) -> Option<String> {
    static RE: OnceLock<Regex> = OnceLock::new();
    let re = regex_cell(&RE, r"\bTS(\d{4})\b");
    re.captures(text)
        .and_then(|caps| caps.get(1))
        .map(|code| format!("typescript:TS{}", code.as_str()))
}

fn json_parse_error(text: &str) -> Option<String> {
    static RE: OnceLock<Regex> = OnceLock::new();
    let re = regex_cell(
        &RE,
        r"(?i)\b(JSON.parse|JSONDecodeError|serde_json|unexpected token|expected value|invalid JSON)\b",
    );
    re.is_match(text).then(|| "json:parse-error".to_string())
}

fn generic_non_zero_exit(text: &str) -> Option<String> {
    static RE: OnceLock<Regex> = OnceLock::new();
    let re = regex_cell(
        &RE,
        r"(?i)\b(exit code|exited with code|status)\s*[:=]?\s*([1-9][0-9]*)\b|\bnon-zero\b",
    );
    re.is_match(text)
        .then(|| "process:non-zero-exit".to_string())
}

fn fingerprint_rules() -> Vec<FingerprintRule> {
    // Specific language errors come before the generic non-zero exit fallback.
    // Adding a new failure class should only require appending or reordering a
    // rule here, plus tests for the expected canonical signature.
    vec![
        FingerprintRule {
            name: "rust_borrow_checker",
            category: "rust",
            apply: rust_borrow_checker,
        },
        FingerprintRule {
            name: "rust_type_mismatch",
            category: "rust",
            apply: rust_type_mismatch,
        },
        FingerprintRule {
            name: "python_import_error",
            category: "python",
            apply: python_import_error,
        },
        FingerprintRule {
            name: "typescript_type_error",
            category: "typescript",
            apply: typescript_type_error,
        },
        FingerprintRule {
            name: "json_parse_error",
            category: "json",
            apply: json_parse_error,
        },
        FingerprintRule {
            name: "generic_non_zero_exit",
            category: "process",
            apply: generic_non_zero_exit,
        },
    ]
}

pub fn fingerprint(raw_error_text: &str) -> FailureFingerprint {
    let raw_excerpt = summarize_text(raw_error_text, 1000);
    for rule in fingerprint_rules() {
        if let Some(signature) = (rule.apply)(raw_error_text) {
            return FailureFingerprint {
                signature,
                category: rule.category.to_string(),
                rule: rule.name.to_string(),
                raw_excerpt,
                version: FINGERPRINT_VERSION,
            };
        }
    }

    FailureFingerprint {
        signature: "unknown".to_string(),
        category: "unknown".to_string(),
        rule: "unknown".to_string(),
        raw_excerpt,
        version: FINGERPRINT_VERSION,
    }
}

pub fn signature_category(fp: &FailureFingerprint) -> &'static str {
    match fp.rule.as_str() {
        "rust_borrow_checker" => "borrow_checker",
        "python_import_error" => "import_error",
        "rust_type_mismatch" | "typescript_type_error" => "type_error",
        "json_parse_error" => "parse_error",
        "generic_non_zero_exit" | "unknown" => "unknown",
        _ => "unknown",
    }
}

pub fn signature_category_from_signature(signature: &str) -> &'static str {
    if signature == "unknown" {
        return "unknown";
    }
    if signature == "rust:borrow-checker" {
        return "borrow_checker";
    }
    if signature == "rust:type-mismatch" || signature.starts_with("typescript:TS") {
        return "type_error";
    }
    if signature == "python:import-error" {
        return "import_error";
    }
    if signature == "json:parse-error" {
        return "parse_error";
    }
    "unknown"
}

pub(crate) fn inline_signature_payload_from_fingerprint(
    mut payload: Value,
    success: bool,
    fingerprint: Option<&FailureFingerprint>,
) -> Value {
    let Some(obj) = payload.as_object_mut() else {
        return payload;
    };

    if success {
        obj.insert("signature".to_string(), Value::Null);
        obj.insert("signature_category".to_string(), Value::Null);
        obj.insert("fingerprint_version".to_string(), Value::Null);
        return payload;
    }

    let fp = fingerprint.expect("failed outcome payloads require a fingerprint");
    obj.insert("signature".to_string(), json!(fp.signature));
    obj.insert(
        "signature_category".to_string(),
        json!(signature_category(fp)),
    );
    obj.insert("fingerprint_version".to_string(), json!(fp.version));
    payload
}

pub fn inline_signature_payload(payload: Value, success: bool, raw_error_text: &str) -> Value {
    if success {
        return inline_signature_payload_from_fingerprint(payload, true, None);
    }

    let fp = fingerprint(raw_error_text);
    inline_signature_payload_from_fingerprint(payload, false, Some(&fp))
}

pub fn outcome_raw_text_from_payload(
    payload: &Value,
    fallback_raw_excerpt: Option<&str>,
) -> String {
    let mut parts = Vec::new();
    if let Some(raw) = fallback_raw_excerpt.filter(|raw| !raw.trim().is_empty()) {
        parts.push(raw.to_string());
    }
    if let Some(reason) = payload.get("failure_reason").and_then(Value::as_str) {
        parts.push(reason.to_string());
    }
    if let Some(stderr) = payload.get("stderr_summary").and_then(Value::as_str) {
        parts.push(stderr.to_string());
    }
    if let Some(stdout) = payload.get("stdout_summary").and_then(Value::as_str) {
        parts.push(stdout.to_string());
    }
    for key in ["failure_summaries", "findings"] {
        if let Some(values) = payload.get(key).and_then(Value::as_array) {
            for value in values {
                if let Some(text) = value.as_str() {
                    parts.push(text.to_string());
                }
            }
        }
    }

    if parts.iter().any(|part| !part.trim().is_empty()) {
        parts.join("\n")
    } else {
        payload.to_string()
    }
}

pub fn backfill_inline_signature_metadata(
    metadata: &Value,
    fallback_raw_excerpt: Option<&str>,
) -> Option<Value> {
    let payload = metadata.get("payload")?;
    let raw = outcome_raw_text_from_payload(payload, fallback_raw_excerpt);
    let mut updated = metadata.clone();
    updated["payload"] = inline_signature_payload(payload.clone(), false, &raw);
    Some(updated)
}

pub fn extract_failure_signatures(text: &str) -> Vec<String> {
    let mut signatures = BTreeSet::new();

    // First treat the incoming text as raw error output. This lets a task like
    // "fix error[E0308]" retrieve prior remediations even if the user never
    // mentions the canonical signature string.
    let fp = fingerprint(text);
    if fp.signature != "unknown" {
        signatures.insert(fp.signature);
    }

    static SIG_RE: OnceLock<Regex> = OnceLock::new();
    let sig_re = regex_cell(
        &SIG_RE,
        r"\b(?:rust|python|typescript|json|process|unknown):[A-Za-z0-9:_-]+\b|\bunknown\b",
    );
    // Also accept already-canonical signatures in prompts or context snippets.
    for caps in sig_re.captures_iter(text) {
        if let Some(sig) = caps.get(0) {
            signatures.insert(sig.as_str().trim_matches('.').to_string());
        }
    }

    signatures.into_iter().collect()
}

pub fn summarize_text(text: &str, max_chars: usize) -> String {
    let normalized = text
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .collect::<Vec<_>>()
        .join("\n");
    normalized.chars().take(max_chars).collect()
}
