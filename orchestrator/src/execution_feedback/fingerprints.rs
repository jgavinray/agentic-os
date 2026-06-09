use regex::Regex;
use std::collections::BTreeSet;
use std::sync::OnceLock;

use crate::execution_feedback_fingerprint_rules::{fingerprint_rules, regex_cell};
pub(crate) use crate::execution_feedback_signature_payloads::inline_signature_payload_from_fingerprint;
pub use crate::execution_feedback_signature_payloads::{
    backfill_inline_signature_metadata, inline_signature_payload, outcome_raw_text_from_payload,
};

pub const FINGERPRINT_VERSION: u32 = 1;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FailureFingerprint {
    pub signature: String,
    pub category: String,
    pub rule: String,
    pub raw_excerpt: String,
    pub version: u32,
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
