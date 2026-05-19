use crate::db::AgentEvent;
use regex::Regex;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::{BTreeMap, BTreeSet};
use std::sync::OnceLock;
use uuid::Uuid;

pub const FINGERPRINT_VERSION: u32 = 1;

pub const EVENT_TYPE_TOOL_RESULT: &str = "tool_result";
pub const EVENT_TYPE_COMPILE_RESULT: &str = "compile_result";
pub const EVENT_TYPE_TEST_RESULT: &str = "test_result";
pub const EVENT_TYPE_LINT_RESULT: &str = "lint_result";
pub const EVENT_TYPE_VALIDATION_RESULT: &str = "validation_result";
pub const EVENT_TYPE_PATCH_RESULT: &str = "patch_result";
pub const EVENT_TYPE_REMEDIATION: &str = "remediation";
pub const EVENT_TYPE_FAILURE_SIGNATURE: &str = "failure_signature";

pub const EXECUTION_EVENT_TYPES: [&str; 8] = [
    EVENT_TYPE_TOOL_RESULT,
    EVENT_TYPE_COMPILE_RESULT,
    EVENT_TYPE_TEST_RESULT,
    EVENT_TYPE_LINT_RESULT,
    EVENT_TYPE_VALIDATION_RESULT,
    EVENT_TYPE_PATCH_RESULT,
    EVENT_TYPE_REMEDIATION,
    EVENT_TYPE_FAILURE_SIGNATURE,
];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExecutionEventKind {
    ToolResult,
    CompileResult,
    TestResult,
    LintResult,
    ValidationResult,
    PatchResult,
    Remediation,
    FailureSignature,
}

impl ExecutionEventKind {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::ToolResult => EVENT_TYPE_TOOL_RESULT,
            Self::CompileResult => EVENT_TYPE_COMPILE_RESULT,
            Self::TestResult => EVENT_TYPE_TEST_RESULT,
            Self::LintResult => EVENT_TYPE_LINT_RESULT,
            Self::ValidationResult => EVENT_TYPE_VALIDATION_RESULT,
            Self::PatchResult => EVENT_TYPE_PATCH_RESULT,
            Self::Remediation => EVENT_TYPE_REMEDIATION,
            Self::FailureSignature => EVENT_TYPE_FAILURE_SIGNATURE,
        }
    }

    pub fn from_str(value: &str) -> Option<Self> {
        Some(match value {
            EVENT_TYPE_TOOL_RESULT => Self::ToolResult,
            EVENT_TYPE_COMPILE_RESULT => Self::CompileResult,
            EVENT_TYPE_TEST_RESULT => Self::TestResult,
            EVENT_TYPE_LINT_RESULT => Self::LintResult,
            EVENT_TYPE_VALIDATION_RESULT => Self::ValidationResult,
            EVENT_TYPE_PATCH_RESULT => Self::PatchResult,
            EVENT_TYPE_REMEDIATION => Self::Remediation,
            EVENT_TYPE_FAILURE_SIGNATURE => Self::FailureSignature,
            _ => return None,
        })
    }
}

#[derive(Clone, Debug)]
pub struct ExecutionEventContext {
    pub session_id: String,
    pub repo: String,
    pub task: String,
    pub actor: String,
    pub correlation_id: Uuid,
    pub parent_event_id: Option<Uuid>,
}

impl ExecutionEventContext {
    pub fn child_of(&self, parent_event_id: Uuid) -> Self {
        Self {
            parent_event_id: Some(parent_event_id),
            ..self.clone()
        }
    }
}

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

pub fn extract_failure_signatures(text: &str) -> Vec<String> {
    let mut signatures = BTreeSet::new();
    let fp = fingerprint(text);
    if fp.signature != "unknown" {
        signatures.insert(fp.signature);
    }

    static SIG_RE: OnceLock<Regex> = OnceLock::new();
    let sig_re = regex_cell(
        &SIG_RE,
        r"\b(?:rust|python|typescript|json|process|unknown):[A-Za-z0-9:_-]+\b|\bunknown\b",
    );
    for caps in sig_re.captures_iter(text) {
        if let Some(sig) = caps.get(0) {
            signatures.insert(sig.as_str().trim_matches('.').to_string());
        }
    }

    signatures.into_iter().collect()
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CapturedToolResult {
    pub tool_name: String,
    pub content: String,
    pub exit_code: i32,
    pub duration_ms: u64,
    pub stdout_summary: String,
    pub stderr_summary: String,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ValidationKind {
    Compile,
    Test,
    Lint,
    Validation,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ValidatorSpec {
    pub validator: &'static str,
    pub kind: ValidationKind,
}

pub fn classify_validator(tool_name: &str, content: &str) -> Option<ValidatorSpec> {
    let name = tool_name.to_ascii_lowercase();
    let body = content.to_ascii_lowercase();
    if name.contains("pytest") {
        return Some(ValidatorSpec {
            validator: "pytest",
            kind: ValidationKind::Test,
        });
    }
    if name.contains("npm test") || name == "npm-test" || name.contains("jest") {
        return Some(ValidatorSpec {
            validator: "npm test",
            kind: ValidationKind::Test,
        });
    }
    if name.contains("cargo") {
        let kind = if name.contains("test") || body.contains("test result:") {
            ValidationKind::Test
        } else if name.contains("clippy") {
            ValidationKind::Lint
        } else {
            ValidationKind::Compile
        };
        return Some(ValidatorSpec {
            validator: "cargo",
            kind,
        });
    }
    if name.contains("eslint") {
        return Some(ValidatorSpec {
            validator: "eslint",
            kind: ValidationKind::Lint,
        });
    }
    if name.contains("tsc") {
        return Some(ValidatorSpec {
            validator: "tsc",
            kind: ValidationKind::Compile,
        });
    }
    if name.contains("mypy") {
        return Some(ValidatorSpec {
            validator: "mypy",
            kind: ValidationKind::Lint,
        });
    }
    if name.contains("ruff") {
        return Some(ValidatorSpec {
            validator: "ruff",
            kind: ValidationKind::Lint,
        });
    }
    if name.contains("terraform") {
        return Some(ValidatorSpec {
            validator: "terraform",
            kind: ValidationKind::Validation,
        });
    }
    if name.contains("kubectl") {
        return Some(ValidatorSpec {
            validator: "kubectl",
            kind: ValidationKind::Validation,
        });
    }
    None
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

fn exit_code_from_text(text: &str) -> Option<i32> {
    static RE: OnceLock<Regex> = OnceLock::new();
    let re = regex_cell(
        &RE,
        r"(?i)\b(?:exit code|exited with code|status)\s*[:=]?\s*([0-9]+)\b",
    );
    re.captures(text)
        .and_then(|caps| caps.get(1))
        .and_then(|m| m.as_str().parse().ok())
}

fn infer_success(exit_code: i32, content: &str) -> bool {
    if exit_code != 0 {
        return false;
    }
    let lower = content.to_ascii_lowercase();
    !(lower.contains("failed") || lower.contains("error:") || lower.contains("traceback"))
        || lower.contains("0 failed")
}

fn first_number_before(label: &str, text: &str) -> Option<u64> {
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

fn failure_summaries(content: &str) -> Vec<String> {
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

pub fn build_execution_event(
    ctx: &ExecutionEventContext,
    kind: ExecutionEventKind,
    success: bool,
    payload: Value,
) -> AgentEvent {
    let event_id = Uuid::new_v4();
    let event_type = kind.as_str();
    let mut metadata = json!({
        "event_type": event_type,
        "success": success,
        "correlation_id": ctx.correlation_id,
        "parent_event_id": ctx.parent_event_id,
        "repo": ctx.repo,
        "task": ctx.task,
        "payload": payload,
    });
    if kind == ExecutionEventKind::FailureSignature {
        metadata["fingerprint_version"] = json!(FINGERPRINT_VERSION);
    }

    AgentEvent {
        id: event_id.to_string(),
        session_id: ctx.session_id.clone(),
        repo: ctx.repo.clone(),
        actor: ctx.actor.clone(),
        event_type: event_type.to_string(),
        summary: event_summary(event_type, success, &metadata["payload"]),
        evidence: Some(summarize_text(&metadata["payload"].to_string(), 1000)),
        metadata,
        correlation_id: Some(ctx.correlation_id),
        parent_event_id: ctx.parent_event_id,
        created_at: chrono::Utc::now(),
        summary_level: 0,
    }
}

fn event_summary(event_type: &str, success: bool, payload: &Value) -> String {
    match event_type {
        EVENT_TYPE_TOOL_RESULT => format!(
            "{} tool `{}` exit_code={}",
            if success { "successful" } else { "failed" },
            payload["tool_name"].as_str().unwrap_or("unknown"),
            payload["exit_code"].as_i64().unwrap_or_default()
        ),
        EVENT_TYPE_COMPILE_RESULT => format!(
            "{} compile `{}` target `{}` errors={} warnings={}",
            if success { "successful" } else { "failed" },
            payload["language"].as_str().unwrap_or("unknown"),
            payload["target"].as_str().unwrap_or("unknown"),
            payload["error_count"].as_u64().unwrap_or_default(),
            payload["warning_count"].as_u64().unwrap_or_default()
        ),
        EVENT_TYPE_TEST_RESULT => format!(
            "{} tests `{}` passed={} failed={} skipped={}",
            if success { "successful" } else { "failed" },
            payload["framework"].as_str().unwrap_or("unknown"),
            payload["passed"].as_u64().unwrap_or_default(),
            payload["failed"].as_u64().unwrap_or_default(),
            payload["skipped"].as_u64().unwrap_or_default()
        ),
        EVENT_TYPE_LINT_RESULT => format!(
            "{} lint `{}` errors={} warnings={}",
            if success { "successful" } else { "failed" },
            payload["tool_name"].as_str().unwrap_or("unknown"),
            payload["error_count"].as_u64().unwrap_or_default(),
            payload["warning_count"].as_u64().unwrap_or_default()
        ),
        EVENT_TYPE_VALIDATION_RESULT => format!(
            "{} validation `{}`",
            if success { "successful" } else { "failed" },
            payload["validator_name"].as_str().unwrap_or("unknown")
        ),
        EVENT_TYPE_PATCH_RESULT => format!(
            "patch {} files_touched={}",
            payload["outcome"].as_str().unwrap_or("unknown"),
            payload["files_touched"]
                .as_array()
                .map(Vec::len)
                .unwrap_or(0)
        ),
        EVENT_TYPE_REMEDIATION => format!(
            "remediation for {}",
            payload["signature"].as_str().unwrap_or("unknown")
        ),
        EVENT_TYPE_FAILURE_SIGNATURE => format!(
            "failure signature {} ({})",
            payload["signature"].as_str().unwrap_or("unknown"),
            payload["category"].as_str().unwrap_or("unknown")
        ),
        _ => format!("{event_type} success={success}"),
    }
}

pub fn tool_result_payload(result: &CapturedToolResult) -> Value {
    json!({
        "tool_name": result.tool_name,
        "exit_code": result.exit_code,
        "stdout_summary": result.stdout_summary,
        "stderr_summary": result.stderr_summary,
        "duration_ms": result.duration_ms,
    })
}

pub fn compile_result_payload(
    language: &str,
    target: &str,
    exit_code: i32,
    content: &str,
) -> Value {
    json!({
        "language": language,
        "target": target,
        "exit_code": exit_code,
        "error_count": compile_error_count(content),
        "warning_count": warning_count(content),
    })
}

pub fn test_result_payload(framework: &str, content: &str) -> Value {
    let passed = first_number_before("passed", content).unwrap_or(0);
    let failed = first_number_before("failed", content).unwrap_or(0);
    let skipped = first_number_before("skipped", content)
        .or_else(|| first_number_before("ignored", content))
        .unwrap_or(0);
    let total = first_number_before("total", content).unwrap_or(passed + failed + skipped);
    json!({
        "framework": framework,
        "total": total,
        "passed": passed,
        "failed": failed,
        "skipped": skipped,
        "failure_summaries": failure_summaries(content),
    })
}

pub fn lint_result_payload(tool_name: &str, content: &str) -> Value {
    let error_count = first_number_before("error", content)
        .or_else(|| first_number_before("errors", content))
        .unwrap_or_else(|| compile_error_count(content) as u64);
    let warning_count = first_number_before("warning", content)
        .or_else(|| first_number_before("warnings", content))
        .unwrap_or_else(|| warning_count(content) as u64);
    json!({
        "tool_name": tool_name,
        "error_count": error_count,
        "warning_count": warning_count,
        "findings": failure_summaries(content),
    })
}

pub fn validation_result_payload(validator_name: &str, pass: bool, failure_reason: &str) -> Value {
    json!({
        "validator_name": validator_name,
        "pass": pass,
        "failure_reason": failure_reason,
    })
}

#[allow(dead_code)]
pub fn patch_result_payload(
    files_touched: Vec<String>,
    outcome: &str,
    validation_event_ids: Vec<Uuid>,
) -> Value {
    json!({
        "files_touched": files_touched,
        "outcome": outcome,
        "validation_event_ids": validation_event_ids,
    })
}

#[allow(dead_code)]
pub fn remediation_payload(
    failure_event_id: Uuid,
    resolving_event_id: Uuid,
    signature: &str,
) -> Value {
    json!({
        "failure_event_id": failure_event_id,
        "resolving_event_id": resolving_event_id,
        "signature": signature,
    })
}

pub fn failure_signature_payload(fp: &FailureFingerprint) -> Value {
    json!({
        "signature": fp.signature,
        "category": fp.category,
        "raw_excerpt": fp.raw_excerpt,
        "rule": fp.rule,
        "fingerprint_version": fp.version,
    })
}

fn compile_error_count(content: &str) -> usize {
    static RE: OnceLock<Regex> = OnceLock::new();
    regex_cell(&RE, r"(?m)^\s*error(?:\[[A-Za-z0-9]+\])?:")
        .find_iter(content)
        .count()
}

fn warning_count(content: &str) -> usize {
    static RE: OnceLock<Regex> = OnceLock::new();
    regex_cell(&RE, r"(?m)^\s*warning(?:\[.*?\])?:?")
        .find_iter(content)
        .count()
}

pub fn events_for_tool_result(
    ctx: &ExecutionEventContext,
    result: &CapturedToolResult,
) -> Vec<AgentEvent> {
    let tool_success = infer_success(result.exit_code, &result.content);
    let tool_event = build_execution_event(
        ctx,
        ExecutionEventKind::ToolResult,
        tool_success,
        tool_result_payload(result),
    );
    let mut events = vec![tool_event.clone()];

    if let Some(spec) = classify_validator(&result.tool_name, &result.content) {
        let validation_ctx = ctx.child_of(Uuid::parse_str(&tool_event.id).unwrap());
        let (kind, payload, success) = match spec.kind {
            ValidationKind::Compile => {
                let language = if spec.validator == "tsc" {
                    "typescript"
                } else {
                    "rust"
                };
                let payload = compile_result_payload(
                    language,
                    spec.validator,
                    result.exit_code,
                    &result.content,
                );
                let success = result.exit_code == 0
                    && payload["error_count"].as_u64().unwrap_or_default() == 0;
                (ExecutionEventKind::CompileResult, payload, success)
            }
            ValidationKind::Test => {
                let payload = test_result_payload(spec.validator, &result.content);
                let success =
                    result.exit_code == 0 && payload["failed"].as_u64().unwrap_or_default() == 0;
                (ExecutionEventKind::TestResult, payload, success)
            }
            ValidationKind::Lint => {
                let payload = lint_result_payload(spec.validator, &result.content);
                let success = result.exit_code == 0
                    && payload["error_count"].as_u64().unwrap_or_default() == 0;
                (ExecutionEventKind::LintResult, payload, success)
            }
            ValidationKind::Validation => {
                let success = infer_success(result.exit_code, &result.content);
                let failure_reason = if success {
                    String::new()
                } else {
                    summarize_text(&result.content, 500)
                };
                let payload = validation_result_payload(spec.validator, success, &failure_reason);
                (ExecutionEventKind::ValidationResult, payload, success)
            }
        };
        let validation_event = build_execution_event(&validation_ctx, kind, success, payload);
        let validation_id = Uuid::parse_str(&validation_event.id).unwrap();
        events.push(validation_event);
        if !success {
            let fp = fingerprint(&result.content);
            let failure_event = build_execution_event(
                &ctx.child_of(validation_id),
                ExecutionEventKind::FailureSignature,
                false,
                failure_signature_payload(&fp),
            );
            events.push(failure_event);
        }
    }

    events
}

#[derive(Debug, Deserialize)]
pub struct ValidationReportRequest {
    pub session_id: Option<String>,
    pub repo: String,
    pub task: String,
    pub actor: Option<String>,
    pub event_type: Option<String>,
    pub validator_name: String,
    pub validator_type: Option<String>,
    pub success: Option<bool>,
    pub exit_code: Option<i32>,
    pub content: Option<String>,
    pub payload: Option<Value>,
    pub correlation_id: Option<Uuid>,
    pub parent_event_id: Option<Uuid>,
}

#[derive(Debug, Serialize)]
pub struct ValidationReportResponse {
    pub captured: bool,
    pub event_ids: Vec<String>,
    pub qdrant_indexed: bool,
}

pub fn events_for_validation_report(
    ctx: &ExecutionEventContext,
    report: &ValidationReportRequest,
) -> Vec<AgentEvent> {
    let content = report.content.as_deref().unwrap_or("");
    if let Some(event_type) = report.event_type.as_deref() {
        if let Some(kind) = ExecutionEventKind::from_str(event_type) {
            let payload = report
                .payload
                .clone()
                .unwrap_or_else(|| validation_result_payload(&report.validator_name, true, ""));
            let success = report.success.unwrap_or_else(|| {
                payload
                    .get("success")
                    .and_then(Value::as_bool)
                    .unwrap_or(true)
            });
            return vec![build_execution_event(ctx, kind, success, payload)];
        }
    }

    let tool_result = CapturedToolResult {
        tool_name: report.validator_name.clone(),
        content: content.to_string(),
        exit_code: report
            .exit_code
            .unwrap_or_else(|| exit_code_from_text(content).unwrap_or(0)),
        duration_ms: 0,
        stdout_summary: summarize_text(content, 500),
        stderr_summary: String::new(),
    };

    let spec = report
        .validator_type
        .as_deref()
        .and_then(|kind| match kind {
            "compile" => Some(ValidatorSpec {
                validator: "explicit",
                kind: ValidationKind::Compile,
            }),
            "test" => Some(ValidatorSpec {
                validator: "explicit",
                kind: ValidationKind::Test,
            }),
            "lint" => Some(ValidatorSpec {
                validator: "explicit",
                kind: ValidationKind::Lint,
            }),
            "validation" => Some(ValidatorSpec {
                validator: "explicit",
                kind: ValidationKind::Validation,
            }),
            _ => None,
        })
        .or_else(|| classify_validator(&report.validator_name, content));

    let mut events = if spec.is_some() {
        events_for_tool_result(ctx, &tool_result)
    } else {
        vec![build_execution_event(
            ctx,
            ExecutionEventKind::ValidationResult,
            report.success.unwrap_or(true),
            report.payload.clone().unwrap_or_else(|| {
                validation_result_payload(
                    &report.validator_name,
                    report.success.unwrap_or(true),
                    report
                        .content
                        .as_deref()
                        .filter(|_| !report.success.unwrap_or(true))
                        .unwrap_or(""),
                )
            }),
        )]
    };

    if let Some(success) = report.success {
        for event in &mut events {
            if event.event_type != EVENT_TYPE_FAILURE_SIGNATURE {
                event.metadata["success"] = json!(success);
            }
        }
    }
    events
}

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
        "rust" => "rust",
        "python" => "python",
        "typescript" => "typescript",
        "json" => "json",
        "process" => "process",
        "unknown" => "unknown",
        _ => "other",
    }
}

#[allow(dead_code)]
pub fn patch_validation_ids_resolve(chain: &[AgentEvent], patch_event: &AgentEvent) -> bool {
    let ids: BTreeSet<&str> = chain.iter().map(|event| event.id.as_str()).collect();
    patch_event
        .metadata
        .get("payload")
        .and_then(|p| p.get("validation_event_ids"))
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(Value::as_str)
        .all(|id| ids.contains(id))
}

pub fn group_by_parent(events: &[AgentEvent]) -> BTreeMap<Option<Uuid>, Vec<AgentEvent>> {
    let mut grouped: BTreeMap<Option<Uuid>, Vec<AgentEvent>> = BTreeMap::new();
    for event in events {
        grouped
            .entry(event.parent_event_id)
            .or_default()
            .push(event.clone());
    }
    for siblings in grouped.values_mut() {
        siblings.sort_by(|a, b| {
            a.created_at
                .cmp(&b.created_at)
                .then_with(|| a.id.cmp(&b.id))
        });
    }
    grouped
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ctx() -> ExecutionEventContext {
        ExecutionEventContext {
            session_id: "session".to_string(),
            repo: "repo".to_string(),
            task: "task".to_string(),
            actor: "agent".to_string(),
            correlation_id: Uuid::new_v4(),
            parent_event_id: None,
        }
    }

    #[test]
    fn event_types_are_defined_in_one_place() {
        assert_eq!(EXECUTION_EVENT_TYPES.len(), 8);
        assert!(EXECUTION_EVENT_TYPES.contains(&EVENT_TYPE_TOOL_RESULT));
        assert!(EXECUTION_EVENT_TYPES.contains(&EVENT_TYPE_FAILURE_SIGNATURE));
    }

    #[test]
    fn fingerprint_rules_cover_initial_classes() {
        let cases = [
            ("error[E0382]: use of moved value", "rust:borrow-checker"),
            ("error[E0308]: mismatched types", "rust:type-mismatch"),
            (
                "ModuleNotFoundError: No module named 'x'",
                "python:import-error",
            ),
            ("src/a.ts:1:2 - error TS2322", "typescript:TS2322"),
            ("JSONDecodeError: Expecting value", "json:parse-error"),
            ("process exited with code 2", "process:non-zero-exit"),
        ];
        for (input, expected) in cases {
            assert_eq!(fingerprint(input).signature, expected);
        }
    }

    #[test]
    fn fingerprint_is_deterministic() {
        let input = "error[E0382]: borrow of moved value";
        assert_eq!(fingerprint(input), fingerprint(input));
    }

    #[test]
    fn unknown_fingerprint_preserves_excerpt() {
        let fp = fingerprint("very strange failure");
        assert_eq!(fp.signature, "unknown");
        assert!(fp.raw_excerpt.contains("very strange failure"));
    }

    #[test]
    fn tool_result_parser_finds_synthetic_block() {
        let value = json!({
            "content": [{
                "type": "tool_result",
                "tool_name": "pytest",
                "exit_code": 1,
                "duration_ms": 12,
                "content": "1 failed, 2 passed in 0.3s"
            }]
        });
        let results = tool_results_from_value(&value);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].tool_name, "pytest");
        assert_eq!(results[0].exit_code, 1);
    }

    #[test]
    fn sse_tool_result_parser_finds_synthetic_block() {
        let raw = concat!(
            "data: {\"content\":[{\"type\":\"tool_result\",\"tool_name\":\"cargo test\",\"content\":\"test result: FAILED. 2 passed; 1 failed; 0 ignored\",\"exit_code\":101}]}\n",
            "data: [DONE]\n"
        );
        let results = tool_results_from_sse(raw);
        assert_eq!(results.len(), 1);
        let events = events_for_tool_result(&ctx(), &results[0]);
        assert!(events
            .iter()
            .any(|event| event.event_type == EVENT_TYPE_TEST_RESULT));
        assert!(events
            .iter()
            .any(|event| event.event_type == EVENT_TYPE_FAILURE_SIGNATURE));
    }

    #[test]
    fn event_builder_writes_required_envelope_fields() {
        for event_type in EXECUTION_EVENT_TYPES {
            let kind = ExecutionEventKind::from_str(event_type).unwrap();
            let event = build_execution_event(&ctx(), kind, true, json!({"example": true}));
            assert_eq!(event.event_type, event_type);
            assert_eq!(event.metadata["event_type"], event_type);
            assert_eq!(event.metadata["success"], true);
            assert!(event.metadata["correlation_id"].is_string());
            assert_eq!(event.metadata["repo"], "repo");
            assert_eq!(event.metadata["task"], "task");
            assert!(event.metadata["payload"].is_object());
        }
    }

    #[test]
    fn event_type_examples_write_and_retrieve_intact() {
        let ctx = ctx();
        let examples = [
            (
                ExecutionEventKind::ToolResult,
                true,
                tool_result_payload(&CapturedToolResult {
                    tool_name: "cargo".to_string(),
                    content: String::new(),
                    exit_code: 0,
                    duration_ms: 1,
                    stdout_summary: "ok".to_string(),
                    stderr_summary: String::new(),
                }),
            ),
            (
                ExecutionEventKind::CompileResult,
                true,
                compile_result_payload("rust", "cargo", 0, ""),
            ),
            (
                ExecutionEventKind::TestResult,
                true,
                test_result_payload("pytest", "2 passed"),
            ),
            (
                ExecutionEventKind::LintResult,
                true,
                lint_result_payload("ruff", "0 errors, 0 warnings"),
            ),
            (
                ExecutionEventKind::ValidationResult,
                true,
                validation_result_payload("schema", true, ""),
            ),
            (
                ExecutionEventKind::PatchResult,
                true,
                patch_result_payload(vec!["src/lib.rs".to_string()], "applied", vec![]),
            ),
            (
                ExecutionEventKind::Remediation,
                true,
                remediation_payload(Uuid::new_v4(), Uuid::new_v4(), "rust:type-mismatch"),
            ),
            (
                ExecutionEventKind::FailureSignature,
                false,
                failure_signature_payload(&fingerprint("error[E0382]: moved value")),
            ),
        ];

        let mut store = std::collections::BTreeMap::new();
        for (kind, success, payload) in examples {
            let event = build_execution_event(&ctx, kind, success, payload.clone());
            store.insert(event.id.clone(), event.clone());
            let retrieved = store.get(&event.id).unwrap();
            assert_eq!(retrieved.event_type, kind.as_str());
            assert_eq!(retrieved.metadata["payload"], payload);
            assert_eq!(retrieved.metadata["success"], success);
        }
    }

    #[test]
    fn validation_parser_recognizes_initial_validator_list() {
        for tool in [
            "cargo",
            "pytest",
            "npm test",
            "eslint",
            "tsc",
            "mypy",
            "ruff",
            "terraform",
            "kubectl",
        ] {
            assert!(classify_validator(tool, "").is_some(), "{tool}");
        }
        assert!(classify_validator("printf", "").is_none());
    }

    #[test]
    fn unrecognized_tool_only_emits_generic_tool_result() {
        let result = CapturedToolResult {
            tool_name: "printf".to_string(),
            content: "hello".to_string(),
            exit_code: 0,
            duration_ms: 1,
            stdout_summary: "hello".to_string(),
            stderr_summary: String::new(),
        };
        let events = events_for_tool_result(&ctx(), &result);
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].event_type, EVENT_TYPE_TOOL_RESULT);
    }

    #[test]
    fn explicit_validation_report_matches_tool_result_event_shape() {
        let report = ValidationReportRequest {
            session_id: Some("session".to_string()),
            repo: "repo".to_string(),
            task: "task".to_string(),
            actor: None,
            event_type: None,
            validator_name: "pytest".to_string(),
            validator_type: Some("test".to_string()),
            success: Some(false),
            exit_code: Some(1),
            content: Some("1 failed, 2 passed".to_string()),
            payload: None,
            correlation_id: Some(ctx().correlation_id),
            parent_event_id: None,
        };
        let events = events_for_validation_report(&ctx(), &report);
        assert!(events
            .iter()
            .any(|event| event.event_type == EVENT_TYPE_TOOL_RESULT));
        assert!(events
            .iter()
            .any(|event| event.event_type == EVENT_TYPE_TEST_RESULT));
        assert!(events
            .iter()
            .any(|event| event.event_type == EVENT_TYPE_FAILURE_SIGNATURE));
    }

    #[test]
    fn execution_capture_p99_under_one_ms() {
        let result = CapturedToolResult {
            tool_name: "cargo".to_string(),
            content: "error[E0308]: mismatched types\nexit code 101".to_string(),
            exit_code: 101,
            duration_ms: 10,
            stdout_summary: String::new(),
            stderr_summary: "error[E0308]: mismatched types".to_string(),
        };
        let ctx = ctx();
        let _ = events_for_tool_result(&ctx, &result);
        let mut durations = Vec::new();
        for _ in 0..1000 {
            let started = std::time::Instant::now();
            let events = events_for_tool_result(&ctx, &result);
            assert!(!events.is_empty());
            durations.push(started.elapsed());
        }
        durations.sort();
        let p99 = durations[(durations.len() * 99) / 100 - 1];
        let threshold = if cfg!(debug_assertions) {
            std::time::Duration::from_millis(10)
        } else {
            std::time::Duration::from_millis(1)
        };
        assert!(p99 < threshold, "p99 was {p99:?}");
    }

    #[test]
    fn mixed_event_load_test_builds_1000_under_five_seconds() {
        let started = std::time::Instant::now();
        let ctx = ctx();
        let _ = fingerprint("error[E0308]: mismatched types");
        let mut count = 0usize;
        for idx in 0..250 {
            let result = CapturedToolResult {
                tool_name: if idx % 2 == 0 {
                    "pytest".to_string()
                } else {
                    "tsc".to_string()
                },
                content: if idx % 2 == 0 {
                    "2 passed, 0 failed".to_string()
                } else {
                    "src/a.ts(1,1): error TS2322: type mismatch".to_string()
                },
                exit_code: if idx % 2 == 0 { 0 } else { 2 },
                duration_ms: idx,
                stdout_summary: String::new(),
                stderr_summary: String::new(),
            };
            count += events_for_tool_result(&ctx, &result).len();
        }
        while count < 1000 {
            let _ = build_execution_event(
                &ctx,
                ExecutionEventKind::PatchResult,
                true,
                patch_result_payload(vec!["src/lib.rs".to_string()], "applied", vec![]),
            );
            count += 1;
        }
        assert!(started.elapsed() < std::time::Duration::from_secs(5));
        assert!(count >= 1000);
    }

    #[test]
    fn patch_validation_references_resolve_inside_chain() {
        let validation = build_execution_event(
            &ctx(),
            ExecutionEventKind::ValidationResult,
            true,
            validation_result_payload("schema", true, ""),
        );
        let patch = build_execution_event(
            &ctx(),
            ExecutionEventKind::PatchResult,
            true,
            patch_result_payload(
                vec!["src/lib.rs".to_string()],
                "applied",
                vec![Uuid::parse_str(&validation.id).unwrap()],
            ),
        );
        assert!(patch_validation_ids_resolve(&[validation], &patch));
    }
}
