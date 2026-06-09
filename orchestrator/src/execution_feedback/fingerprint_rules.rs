use regex::Regex;
use std::sync::OnceLock;

pub(crate) struct FingerprintRule {
    pub name: &'static str,
    pub category: &'static str,
    pub apply: fn(&str) -> Option<String>,
}

// Regex compilation is cached because this code runs on request-adjacent paths.
// The rules are deterministic, so the first successful match wins every time.
pub(crate) fn regex_cell(cell: &'static OnceLock<Regex>, pattern: &str) -> &'static Regex {
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

pub(crate) fn fingerprint_rules() -> Vec<FingerprintRule> {
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
