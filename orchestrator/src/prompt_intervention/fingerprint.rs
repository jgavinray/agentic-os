/// Prompt fingerprinting helpers for telemetry prompt interventions.
///
/// This module is deliberately narrow: it normalizes prompt text, computes
/// stable hashes, and prepares bounded redacted evidence excerpts. It does not
/// parse raw HTTP payloads or classify intervention rules.
use regex::Regex;
use std::sync::OnceLock;

use crate::prompt_intervention_records::{sha256_hex, EVIDENCE_EXCERPT_MAX};

/// Deterministic normalization version for prompt-family fingerprints.
pub const NORMALIZATION_VERSION: &str = "prompt-fingerprint-v1";

/// Numeric fingerprint version stored with prompt intervention records.
pub const PROMPT_FINGERPRINT_VERSION: u32 = 1;

const UUID_PLACEHOLDER: &str = "<uuid>";
const PATH_PLACEHOLDER: &str = "<path>";
const TIMESTAMP_PLACEHOLDER: &str = "<timestamp>";
const TOOL_RESULT_PLACEHOLDER: &str = "<tool_result>";
const SYSTEM_REMINDER_PLACEHOLDER: &str = "<system_reminder>";
const REDACTED_PLACEHOLDER: &str = "<redacted>";
const LARGE_TOOL_RESULT_THRESHOLD: usize = 1_024;

/// Exact and normalized prompt fingerprints for one extracted text source.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PromptFingerprint {
    /// SHA-256 hex digest of the exact extracted text.
    pub exact_prompt_hash: String,
    /// SHA-256 hex digest of the normalized text.
    pub normalized_prompt_hash: String,
    /// Normalized text used for prompt-family analysis.
    pub normalized_text: String,
    /// Numeric fingerprint version.
    pub fingerprint_version: u32,
    /// Human-readable normalization version.
    pub normalization_version: &'static str,
}

/// Compute exact and normalized fingerprints for extracted prompt text.
pub fn fingerprint_prompt(text: &str) -> PromptFingerprint {
    let normalized_text = normalize_prompt(text);
    PromptFingerprint {
        exact_prompt_hash: sha256_hex(text),
        normalized_prompt_hash: sha256_hex(&normalized_text),
        normalized_text,
        fingerprint_version: PROMPT_FINGERPRINT_VERSION,
        normalization_version: NORMALIZATION_VERSION,
    }
}

/// Normalize prompt text for prompt-family grouping.
pub fn normalize_prompt(text: &str) -> String {
    let mut normalized = text.to_lowercase();
    normalized = system_reminder_re()
        .replace_all(&normalized, SYSTEM_REMINDER_PLACEHOLDER)
        .to_string();
    normalized = replace_large_tool_result_blocks(&normalized);
    normalized = timestamp_re()
        .replace_all(&normalized, TIMESTAMP_PLACEHOLDER)
        .to_string();
    normalized = date_re()
        .replace_all(&normalized, TIMESTAMP_PLACEHOLDER)
        .to_string();
    normalized = uuid_re()
        .replace_all(&normalized, UUID_PLACEHOLDER)
        .to_string();
    normalized = absolute_path_re()
        .replace_all(&normalized, format!("$1{}", PATH_PLACEHOLDER).as_str())
        .to_string();
    collapse_whitespace(&normalized)
}

/// Redact obvious secrets from bounded evidence text before storage.
pub fn redact_evidence(text: &str) -> String {
    let mut redacted = text.to_string();
    redacted = authorization_re()
        .replace_all(&redacted, REDACTED_PLACEHOLDER)
        .to_string();
    redacted = quoted_secret_header_re()
        .replace_all(&redacted, format!("$1{}", REDACTED_PLACEHOLDER).as_str())
        .to_string();
    redacted = bearer_re()
        .replace_all(&redacted, REDACTED_PLACEHOLDER)
        .to_string();
    redacted = cookie_re()
        .replace_all(&redacted, REDACTED_PLACEHOLDER)
        .to_string();
    redacted = compound_secret_assignment_re()
        .replace_all(&redacted, format!("$1={}", REDACTED_PLACEHOLDER).as_str())
        .to_string();
    redacted = assignment_secret_re()
        .replace_all(&redacted, format!("$1={}", REDACTED_PLACEHOLDER).as_str())
        .to_string();
    sk_key_re()
        .replace_all(&redacted, REDACTED_PLACEHOLDER)
        .to_string()
}

/// Return a redacted evidence excerpt bounded to `EVIDENCE_EXCERPT_MAX`.
///
/// When a matched phrase is present, the excerpt is centered near the first
/// case-insensitive occurrence. Otherwise it falls back to the beginning.
pub fn evidence_excerpt(text: &str, matched_phrase: &str) -> String {
    let match_start = find_case_insensitive(text, matched_phrase).unwrap_or(0);
    let phrase_len = if matched_phrase.is_empty() {
        0
    } else {
        matched_phrase.len()
    };
    let window_start = match_start.saturating_sub(EVIDENCE_EXCERPT_MAX / 2);
    let mut window_end = window_start
        .saturating_add(EVIDENCE_EXCERPT_MAX)
        .max(match_start.saturating_add(phrase_len))
        .min(text.len());
    let mut window_start = clamp_to_char_boundary(text, window_start);
    window_end = clamp_to_char_boundary(text, window_end);
    if window_start > window_end {
        window_start = 0;
    }
    let excerpt = redact_evidence(&text[window_start..window_end]);
    truncate_to_char_boundary(&excerpt, EVIDENCE_EXCERPT_MAX).to_string()
}

/// Compute the evidence hash for a redacted evidence excerpt.
pub fn evidence_hash_for_excerpt(excerpt: &str) -> String {
    sha256_hex(excerpt)
}

fn uuid_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b")
            .expect("valid uuid regex")
    })
}

fn absolute_path_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(r#"(^|[\s"'(])(/[^ \t\r\n"'<>]+|[a-z]:[\\/][^ \t\r\n"'<>]+)"#)
            .expect("valid absolute path regex")
    })
}

fn timestamp_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(r"\b\d{4}-\d{2}-\d{2}[t ]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:z|[+-]\d{2}:?\d{2})?\b")
            .expect("valid timestamp regex")
    })
}

fn date_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"\b\d{4}-\d{2}-\d{2}\b").expect("valid date regex"))
}

fn system_reminder_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(r"(?s)<system-reminder>.*?</system-reminder>")
            .expect("valid system reminder regex")
    })
}

fn fenced_tool_result_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(r"(?s)```tool_result\b.*?```").expect("valid fenced tool result regex")
    })
}

fn authorization_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(r"(?i)\bauthorization\s*[:=]\s*[^\r\n]+").expect("valid authorization regex")
    })
}

fn bearer_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r#"(?i)\bbearer\s+[^\s"',}\]\r\n]+"#).expect("valid bearer regex"))
}

fn cookie_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(r"(?i)\b(?:cookie|set-cookie)\s*[:=]\s*[^\r\n]+").expect("valid cookie regex")
    })
}

fn assignment_secret_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(r"(?i)\b(api[_-]?key|password|token)\s*[:=]\s*[^\s,;]+")
            .expect("valid assignment secret regex")
    })
}

fn compound_secret_assignment_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(
            r"(?i)\b([a-z0-9_]*(?:api[_-]?key|auth[_-]?token|access[_-]?token|secret|password|token))\s*[:=]\s*[^\s,;]+",
        )
        .expect("valid compound secret assignment regex")
    })
}

fn quoted_secret_header_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(r#"(?i)(["']?(?:authorization|cookie|set-cookie|api[_-]?key|x-api-key|[a-z0-9_]*(?:auth[_-]?token|access[_-]?token|secret|password|token))["']?\s*:\s*["']?)[^"',}\]\r\n]+"#)
            .expect("valid quoted secret header regex")
    })
}

fn sk_key_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"sk-[a-zA-Z0-9_-]{6,}").expect("valid sk key regex"))
}

fn replace_large_tool_result_blocks(text: &str) -> String {
    fenced_tool_result_re()
        .replace_all(text, |captures: &regex::Captures<'_>| {
            let block = captures.get(0).map(|m| m.as_str()).unwrap_or_default();
            if block.len() >= LARGE_TOOL_RESULT_THRESHOLD {
                TOOL_RESULT_PLACEHOLDER.to_string()
            } else {
                block.to_string()
            }
        })
        .to_string()
}

fn collapse_whitespace(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut previous_whitespace = false;
    for ch in text.chars() {
        if ch.is_whitespace() {
            if !previous_whitespace && !out.is_empty() {
                out.push(' ');
            }
            previous_whitespace = true;
        } else {
            out.push(ch);
            previous_whitespace = false;
        }
    }
    out.trim().to_string()
}

fn find_case_insensitive(text: &str, needle: &str) -> Option<usize> {
    if needle.is_empty() {
        return Some(0);
    }
    text.to_lowercase().find(&needle.to_lowercase())
}

fn clamp_to_char_boundary(text: &str, mut idx: usize) -> usize {
    idx = idx.min(text.len());
    while idx > 0 && !text.is_char_boundary(idx) {
        idx -= 1;
    }
    idx
}

fn truncate_to_char_boundary(text: &str, max_len: usize) -> &str {
    if text.len() <= max_len {
        return text;
    }
    &text[..clamp_to_char_boundary(text, max_len)]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fingerprint_prompt_sets_hashes_and_versions() {
        let fingerprint = fingerprint_prompt("Do not implement this yet.");
        assert_eq!(fingerprint.fingerprint_version, PROMPT_FINGERPRINT_VERSION);
        assert_eq!(fingerprint.normalization_version, NORMALIZATION_VERSION);
        assert_eq!(fingerprint.normalized_text, "do not implement this yet.");
        assert_eq!(
            fingerprint.exact_prompt_hash,
            sha256_hex("Do not implement this yet.")
        );
        assert_eq!(
            fingerprint.normalized_prompt_hash,
            sha256_hex("do not implement this yet.")
        );
    }

    #[test]
    fn normalizes_case_and_repeated_whitespace() {
        assert_eq!(
            normalize_prompt("  REVIEW   AND\nREFINE\tthis prompt  "),
            "review and refine this prompt"
        );
    }

    #[test]
    fn replaces_volatile_values_with_stable_placeholders() {
        let input = "Use /archive/agentic-os/src/main.rs at 2026-06-07T12:30:00Z for 550e8400-e29b-41d4-a716-446655440000.";
        let normalized = normalize_prompt(input);
        assert_eq!(normalized, "use <path> at <timestamp> for <uuid>.");
    }

    #[test]
    fn replaces_windows_absolute_paths() {
        assert_eq!(
            normalize_prompt(r#"Open C:\Users\me\repo\file.rs now"#),
            "open <path> now"
        );
    }

    #[test]
    fn replaces_system_reminder_blocks() {
        let normalized = normalize_prompt(
            "Before <system-reminder>private state\nwith lines</system-reminder> after",
        );
        assert_eq!(normalized, "before <system_reminder> after");
    }

    #[test]
    fn replaces_only_large_fenced_tool_result_blocks() {
        let small = normalize_prompt("```tool_result\nshort\n```");
        assert!(small.contains("short"));

        let large = format!("```tool_result\n{}\n```", "x".repeat(1_200));
        assert_eq!(normalize_prompt(&large), "<tool_result>");
    }

    #[test]
    fn preserves_imperative_phrases_after_normalization() {
        let normalized = normalize_prompt(
            "DO NOT IMPLEMENT. EDIT ONLY config. REVIEW AND REFINE. ACCEPTANCE TESTS REQUIRED.",
        );
        assert!(normalized.contains("do not implement"));
        assert!(normalized.contains("edit only"));
        assert!(normalized.contains("review and refine"));
        assert!(normalized.contains("acceptance tests required"));
    }

    #[test]
    fn normalized_hash_groups_equivalent_prompt_families() {
        let a = fingerprint_prompt(
            "Edit /tmp/a.rs for 550e8400-e29b-41d4-a716-446655440000 on 2026-06-07",
        );
        let b = fingerprint_prompt(
            "EDIT /var/b.rs for 6ba7b810-9dad-11d1-80b4-00c04fd430c8 on 2027-01-31",
        );
        assert_eq!(a.normalized_prompt_hash, b.normalized_prompt_hash);
        assert_ne!(a.exact_prompt_hash, b.exact_prompt_hash);
    }

    #[test]
    fn redacts_secret_patterns() {
        let redacted = redact_evidence(
            "Authorization: Bearer abc\nCookie: session=123\napi_key=secret password=hunter2 token=t sk-prod-123456",
        );
        assert!(!redacted.contains("Bearer abc"));
        assert!(!redacted.contains("session=123"));
        assert!(!redacted.contains("secret"));
        assert!(!redacted.contains("hunter2"));
        assert!(!redacted.contains("sk-prod-123456"));
        assert!(redacted.contains(REDACTED_PLACEHOLDER));
    }

    #[test]
    fn redacts_compound_secret_assignments() {
        let redacted = redact_evidence(
            "ANTHROPIC_AUTH_TOKEN=sk-agent-clean-002 OPENAI_API_KEY:sk-openai-123456 SERVICE_SECRET=value DB_PASSWORD:hunter2",
        );

        assert!(!redacted.contains("sk-agent-clean-002"));
        assert!(!redacted.contains("sk-openai-123456"));
        assert!(!redacted.contains("value"));
        assert!(!redacted.contains("hunter2"));
        assert!(!redacted.contains("sk-"));
        assert!(redacted.contains(REDACTED_PLACEHOLDER));
    }

    #[test]
    fn redacts_quoted_json_secret_headers() {
        let redacted = redact_evidence(
            r#"{"Authorization":"Bearer secret-token","x-api-key":"sk-test-123456","Cookie":"session=abc","auth_token":"sk-agent-clean-002","password":"hunter2","safe":"keep"}"#,
        );

        assert!(!redacted.contains("Bearer secret-token"));
        assert!(!redacted.contains("sk-test-123456"));
        assert!(!redacted.contains("sk-agent-clean-002"));
        assert!(!redacted.contains("session=abc"));
        assert!(!redacted.contains("hunter2"));
        assert!(redacted.matches(REDACTED_PLACEHOLDER).count() >= 5);
        assert!(redacted.contains(r#""safe":"keep""#));
    }

    #[test]
    fn evidence_excerpt_is_bounded_centered_and_redacted() {
        let text = format!(
            "{} Bearer secret-token do not implement this change {}",
            "x".repeat(400),
            "y".repeat(400)
        );
        let excerpt = evidence_excerpt(&text, "do not implement");
        assert!(excerpt.len() <= EVIDENCE_EXCERPT_MAX);
        assert!(excerpt.contains("do not implement"));
        assert!(!excerpt.contains("secret-token"));
        assert!(excerpt.contains(REDACTED_PLACEHOLDER));
    }

    #[test]
    fn evidence_excerpt_handles_utf8_boundaries() {
        let text = format!("{} do not implement {}", "é".repeat(400), "z".repeat(400));
        let excerpt = evidence_excerpt(&text, "do not implement");
        assert!(excerpt.len() <= EVIDENCE_EXCERPT_MAX);
        assert!(excerpt.contains("do not implement"));
    }

    #[test]
    fn evidence_hash_uses_redacted_excerpt_text() {
        let excerpt = evidence_excerpt("Bearer secret do not implement", "do not implement");
        assert_eq!(evidence_hash_for_excerpt(&excerpt), sha256_hex(&excerpt));
    }
}
