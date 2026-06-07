//! Deterministic execution feedback capture.
//!
//! This module deliberately avoids LLM calls. It turns observed tool output and
//! explicit validation reports into structured `agent_events` payloads using
//! regexes, fixed rule ordering, and small parser helpers. The resulting events
//! can then flow through the existing Postgres, Qdrant, FTS, and metrics paths.

use crate::db::AgentEvent;
use regex::Regex;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::{BTreeMap, BTreeSet};
use std::sync::OnceLock;
use uuid::Uuid;

use crate::execution_feedback_fingerprints::inline_signature_payload_from_fingerprint;
pub use crate::execution_feedback_fingerprints::{
    backfill_inline_signature_metadata, extract_failure_signatures, fingerprint,
    inline_signature_payload, outcome_raw_text_from_payload, signature_category,
    signature_category_from_signature, summarize_text, FailureFingerprint, FINGERPRINT_VERSION,
};
use crate::execution_feedback_parsing::{
    exit_code_from_text, failure_summaries, first_number_before, infer_success,
};
pub use crate::execution_feedback_parsing::{
    tool_results_from_sse, tool_results_from_value, CapturedToolResult,
};
use crate::execution_feedback_validators::validator_type_str;
pub use crate::execution_feedback_validators::{
    classify_validator, validate_validator_type, ValidationKind, ValidatorSpec, VALIDATOR_TYPES,
};

// Keep the public event vocabulary centralized so storage, metrics, docs, and
// tests do not drift into subtly different names for the same artifact.
pub const EVENT_TYPE_TOOL_RESULT: &str = "tool_result";
pub const EVENT_TYPE_COMPILE_RESULT: &str = "compile_result";
pub const EVENT_TYPE_TEST_RESULT: &str = "test_result";
pub const EVENT_TYPE_LINT_RESULT: &str = "lint_result";
pub const EVENT_TYPE_VALIDATION_RESULT: &str = "validation_result";
pub const EVENT_TYPE_PATCH_RESULT: &str = "patch_result";
pub const EVENT_TYPE_REMEDIATION: &str = "remediation";

pub const EXECUTION_EVENT_TYPES: [&str; 7] = [
    EVENT_TYPE_TOOL_RESULT,
    EVENT_TYPE_COMPILE_RESULT,
    EVENT_TYPE_TEST_RESULT,
    EVENT_TYPE_LINT_RESULT,
    EVENT_TYPE_VALIDATION_RESULT,
    EVENT_TYPE_PATCH_RESULT,
    EVENT_TYPE_REMEDIATION,
];

pub const FAILURE_OUTCOME_EVENT_TYPES: [&str; 5] = [
    EVENT_TYPE_TOOL_RESULT,
    EVENT_TYPE_COMPILE_RESULT,
    EVENT_TYPE_TEST_RESULT,
    EVENT_TYPE_LINT_RESULT,
    EVENT_TYPE_VALIDATION_RESULT,
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
        }
    }

    #[allow(clippy::should_implement_trait)]
    pub fn from_str(value: &str) -> Option<Self> {
        Some(match value {
            EVENT_TYPE_TOOL_RESULT => Self::ToolResult,
            EVENT_TYPE_COMPILE_RESULT => Self::CompileResult,
            EVENT_TYPE_TEST_RESULT => Self::TestResult,
            EVENT_TYPE_LINT_RESULT => Self::LintResult,
            EVENT_TYPE_VALIDATION_RESULT => Self::ValidationResult,
            EVENT_TYPE_PATCH_RESULT => Self::PatchResult,
            EVENT_TYPE_REMEDIATION => Self::Remediation,
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
    pub trajectory: Option<crate::trajectory::TrajectoryContext>,
}

impl ExecutionEventContext {
    /// Reuse the same correlation ID while advancing the parent pointer.
    /// This is the chain model: the patch is reconstructed from linked events.
    pub fn child_of(&self, parent_event_id: Uuid) -> Self {
        Self {
            parent_event_id: Some(parent_event_id),
            ..self.clone()
        }
    }
}

fn regex_cell(cell: &'static OnceLock<Regex>, pattern: &str) -> &'static Regex {
    cell.get_or_init(|| Regex::new(pattern).expect("fingerprint regex must compile"))
}

pub fn is_failure_outcome_event_type(event_type: &str) -> bool {
    FAILURE_OUTCOME_EVENT_TYPES.contains(&event_type)
}

#[allow(clippy::too_many_arguments)]
fn normalize_validation_payload(
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

pub fn build_execution_event(
    ctx: &ExecutionEventContext,
    kind: ExecutionEventKind,
    success: bool,
    mut payload: Value,
) -> AgentEvent {
    let event_id = Uuid::new_v4();
    let event_type = kind.as_str();
    if is_failure_outcome_event_type(event_type) {
        let has_signature_shape = ["signature", "signature_category", "fingerprint_version"]
            .iter()
            .all(|key| payload.get(*key).is_some());
        if !has_signature_shape {
            let raw = outcome_raw_text_from_payload(&payload, None);
            payload = inline_signature_payload(payload, success, &raw);
        }
    }
    let event_role = ctx
        .trajectory
        .and_then(|_| crate::trajectory::default_role_for_event_type(event_type));
    let trajectory_id = ctx.trajectory.map(|trajectory| trajectory.trajectory_id);
    let attempt_index = ctx.trajectory.map(|trajectory| trajectory.attempt_index);
    // The structured envelope is duplicated into metadata while the chain IDs
    // are also real columns. That keeps old retrieval/indexing behavior intact
    // and gives SQL exact-match queries fast access to chain fields.
    let mut metadata = json!({
        "event_type": event_type,
        "success": success,
        "correlation_id": ctx.correlation_id,
        "parent_event_id": ctx.parent_event_id,
        "repo": ctx.repo,
        "task": ctx.task,
        "payload": payload,
    });
    if let Some(trajectory) = ctx.trajectory {
        metadata["trajectory_id"] = json!(trajectory.trajectory_id);
        metadata["attempt_index"] = json!(trajectory.attempt_index);
    }
    if let Some(role) = event_role {
        metadata["event_role"] = json!(role.as_str());
    }
    metadata = crate::feature_extraction::annotate_event_metadata(
        event_type,
        &event_summary(event_type, success, &metadata["payload"]),
        Some(&summarize_text(&metadata["payload"].to_string(), 1000)),
        metadata,
    );

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
        trajectory_id,
        attempt_index,
        event_role: event_role.map(|role| role.as_str().to_string()),
        created_at: chrono::Utc::now(),
        summary_level: 0,
    }
}

fn event_summary(event_type: &str, success: bool, payload: &Value) -> String {
    // Summaries stay compact because they are indexed for retrieval and shown
    // in context packs; full structured details live in metadata.payload.
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
        _ => format!("{event_type} success={success}"),
    }
}

pub fn tool_result_payload(result: &CapturedToolResult) -> Value {
    let success = infer_success(result.exit_code, &result.content);
    let fp = (!success).then(|| fingerprint(&result.content));
    tool_result_payload_with_fingerprint(result, success, fp.as_ref())
}

fn tool_result_payload_with_fingerprint(
    result: &CapturedToolResult,
    success: bool,
    fp: Option<&FailureFingerprint>,
) -> Value {
    inline_signature_payload_from_fingerprint(
        json!({
        "tool_name": result.tool_name,
        "exit_code": result.exit_code,
        "stdout_summary": result.stdout_summary,
        "stderr_summary": result.stderr_summary,
        "duration_ms": result.duration_ms,
        }),
        success,
        fp,
    )
}

pub fn compile_result_payload(
    language: &str,
    target: &str,
    exit_code: i32,
    content: &str,
) -> Value {
    let error_count = compile_error_count(content);
    let warning_count = warning_count(content);
    let success = exit_code == 0 && error_count == 0;
    let fp = (!success).then(|| fingerprint(content));
    compile_result_payload_with_fingerprint(
        language,
        target,
        exit_code,
        error_count,
        warning_count,
        success,
        fp.as_ref(),
    )
}

fn compile_result_payload_with_fingerprint(
    language: &str,
    target: &str,
    exit_code: i32,
    error_count: usize,
    warning_count: usize,
    success: bool,
    fp: Option<&FailureFingerprint>,
) -> Value {
    let payload = inline_signature_payload_from_fingerprint(
        json!({
        "language": language,
        "target": target,
        "exit_code": exit_code,
        "error_count": error_count,
        "warning_count": warning_count,
        }),
        success,
        fp,
    );
    normalize_validation_payload(
        payload,
        target,
        "compile",
        target,
        success,
        Some(exit_code),
        None,
        fp.map(|fp| fp.raw_excerpt.clone()),
    )
}

pub fn test_result_payload(framework: &str, content: &str) -> Value {
    // Test runners report counts in many formats. These counters are best-effort
    // summaries, not a replacement for the raw excerpt kept in related events.
    let passed = first_number_before("passed", content).unwrap_or(0);
    let failed = first_number_before("failed", content).unwrap_or(0);
    let skipped = first_number_before("skipped", content)
        .or_else(|| first_number_before("ignored", content))
        .unwrap_or(0);
    let total = first_number_before("total", content).unwrap_or(passed + failed + skipped);
    let success = failed == 0;
    let fp = (!success).then(|| fingerprint(content));
    test_result_payload_with_fingerprint(
        framework,
        total,
        passed,
        failed,
        skipped,
        success,
        fp.as_ref(),
        content,
    )
}

#[allow(clippy::too_many_arguments)]
fn test_result_payload_with_fingerprint(
    framework: &str,
    total: u64,
    passed: u64,
    failed: u64,
    skipped: u64,
    success: bool,
    fp: Option<&FailureFingerprint>,
    content: &str,
) -> Value {
    let payload = inline_signature_payload_from_fingerprint(
        json!({
        "framework": framework,
        "total": total,
        "passed": passed,
        "failed": failed,
        "skipped": skipped,
        "failure_summaries": failure_summaries(content),
        }),
        success,
        fp,
    );
    normalize_validation_payload(
        payload,
        framework,
        "test",
        framework,
        success,
        None,
        None,
        fp.map(|fp| fp.raw_excerpt.clone()),
    )
}

pub fn lint_result_payload(tool_name: &str, content: &str) -> Value {
    let error_count = first_number_before("error", content)
        .or_else(|| first_number_before("errors", content))
        .unwrap_or_else(|| compile_error_count(content) as u64);
    let warning_count = first_number_before("warning", content)
        .or_else(|| first_number_before("warnings", content))
        .unwrap_or_else(|| warning_count(content) as u64);
    let success = error_count == 0;
    let fp = (!success).then(|| fingerprint(content));
    lint_result_payload_with_fingerprint(
        tool_name,
        error_count,
        warning_count,
        success,
        fp.as_ref(),
        content,
    )
}

fn lint_result_payload_with_fingerprint(
    tool_name: &str,
    error_count: u64,
    warning_count: u64,
    success: bool,
    fp: Option<&FailureFingerprint>,
    content: &str,
) -> Value {
    let payload = inline_signature_payload_from_fingerprint(
        json!({
        "tool_name": tool_name,
        "error_count": error_count,
        "warning_count": warning_count,
        "findings": failure_summaries(content),
        }),
        success,
        fp,
    );
    normalize_validation_payload(
        payload,
        tool_name,
        "lint",
        tool_name,
        success,
        None,
        None,
        fp.map(|fp| fp.raw_excerpt.clone()),
    )
}

pub fn validation_result_payload(validator_name: &str, pass: bool, failure_reason: &str) -> Value {
    let payload = inline_signature_payload(
        json!({
        "validator_name": validator_name,
        "pass": pass,
        "failure_reason": failure_reason,
        }),
        pass,
        failure_reason,
    );
    normalize_validation_payload(
        payload,
        validator_name,
        "other",
        validator_name,
        pass,
        None,
        None,
        (!pass).then(|| summarize_text(failure_reason, 500)),
    )
}

#[allow(dead_code)]
pub fn patch_result_payload(
    files_touched: Vec<String>,
    outcome: &str,
    validation_event_ids: Vec<Uuid>,
) -> Value {
    json!({
        "files_touched": files_touched,
        "lines_added": 0,
        "lines_removed": 0,
        "patch_applied": outcome == "applied",
        "patch_reverted": outcome == "reverted",
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
    let tool_fp = (!tool_success).then(|| fingerprint(&result.content));
    let tool_event = build_execution_event(
        ctx,
        ExecutionEventKind::ToolResult,
        tool_success,
        tool_result_payload_with_fingerprint(result, tool_success, tool_fp.as_ref()),
    );
    let mut events = vec![tool_event.clone()];

    if let Some(spec) = classify_validator(&result.tool_name, &result.content) {
        // A validator tool produces two linked memories: the observed tool run
        // and the normalized compile/test/lint/validation result derived from it.
        let validation_ctx = ctx.child_of(Uuid::parse_str(&tool_event.id).unwrap());
        let (kind, payload, success) = match spec.kind {
            ValidationKind::Compile | ValidationKind::TypeCheck => {
                let language = if spec.validator == "tsc" {
                    "typescript"
                } else if spec.validator == "mypy" {
                    "python"
                } else {
                    "rust"
                };
                let error_count = compile_error_count(&result.content);
                let warning_count = warning_count(&result.content);
                let success = result.exit_code == 0 && error_count == 0;
                let mut payload = compile_result_payload_with_fingerprint(
                    language,
                    spec.validator,
                    result.exit_code,
                    error_count,
                    warning_count,
                    success,
                    tool_fp.as_ref(),
                );
                payload["validator_type"] = json!(validator_type_str(spec.kind));
                payload["duration_ms"] = json!(result.duration_ms);
                (ExecutionEventKind::CompileResult, payload, success)
            }
            ValidationKind::Test => {
                let passed = first_number_before("passed", &result.content).unwrap_or(0);
                let failed = first_number_before("failed", &result.content).unwrap_or(0);
                let skipped = first_number_before("skipped", &result.content)
                    .or_else(|| first_number_before("ignored", &result.content))
                    .unwrap_or(0);
                let total = first_number_before("total", &result.content)
                    .unwrap_or(passed + failed + skipped);
                let success = result.exit_code == 0 && failed == 0;
                let mut payload = test_result_payload_with_fingerprint(
                    spec.validator,
                    total,
                    passed,
                    failed,
                    skipped,
                    success,
                    tool_fp.as_ref(),
                    &result.content,
                );
                payload["duration_ms"] = json!(result.duration_ms);
                (ExecutionEventKind::TestResult, payload, success)
            }
            ValidationKind::Lint => {
                let error_count = first_number_before("error", &result.content)
                    .or_else(|| first_number_before("errors", &result.content))
                    .unwrap_or_else(|| compile_error_count(&result.content) as u64);
                let warning_count = first_number_before("warning", &result.content)
                    .or_else(|| first_number_before("warnings", &result.content))
                    .unwrap_or_else(|| warning_count(&result.content) as u64);
                let success = result.exit_code == 0 && error_count == 0;
                let mut payload = lint_result_payload_with_fingerprint(
                    spec.validator,
                    error_count,
                    warning_count,
                    success,
                    tool_fp.as_ref(),
                    &result.content,
                );
                payload["duration_ms"] = json!(result.duration_ms);
                (ExecutionEventKind::LintResult, payload, success)
            }
            ValidationKind::Schema | ValidationKind::StaticAnalysis | ValidationKind::Other => {
                let success = infer_success(result.exit_code, &result.content);
                let failure_reason = if success {
                    String::new()
                } else {
                    summarize_text(&result.content, 500)
                };
                let payload = inline_signature_payload_from_fingerprint(
                    json!({
                        "validator_name": spec.validator,
                        "pass": success,
                        "failure_reason": failure_reason,
                    }),
                    success,
                    tool_fp.as_ref(),
                );
                let payload = normalize_validation_payload(
                    payload,
                    spec.validator,
                    validator_type_str(spec.kind),
                    spec.validator,
                    success,
                    Some(result.exit_code),
                    Some(result.duration_ms),
                    (!success).then(|| summarize_text(&result.content, 500)),
                );
                (ExecutionEventKind::ValidationResult, payload, success)
            }
        };
        let validation_event = build_execution_event(&validation_ctx, kind, success, payload);
        events.push(validation_event);
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
    pub trajectory_id: Option<Uuid>,
    pub attempt_index: Option<i32>,
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
) -> Result<Vec<AgentEvent>, String> {
    validate_validator_type(report.validator_type.as_deref())?;
    let content = report.content.as_deref().unwrap_or("");
    if let Some(event_type) = report.event_type.as_deref() {
        if let Some(kind) = ExecutionEventKind::from_str(event_type) {
            // Advanced clients can submit an already-normalized artifact event.
            // We still wrap it with the standard event envelope for consistency.
            let mut payload = report
                .payload
                .clone()
                .unwrap_or_else(|| validation_result_payload(&report.validator_name, true, ""));
            let success = report.success.unwrap_or_else(|| {
                payload
                    .get("success")
                    .and_then(Value::as_bool)
                    .unwrap_or(true)
            });
            if is_failure_outcome_event_type(kind.as_str()) {
                payload = inline_signature_payload(payload, success, content);
            }
            return Ok(vec![build_execution_event(ctx, kind, success, payload)]);
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
            "type_check" => Some(ValidatorSpec {
                validator: "explicit",
                kind: ValidationKind::TypeCheck,
            }),
            "schema" => Some(ValidatorSpec {
                validator: "explicit",
                kind: ValidationKind::Schema,
            }),
            "static_analysis" => Some(ValidatorSpec {
                validator: "explicit",
                kind: ValidationKind::StaticAnalysis,
            }),
            "other" => Some(ValidatorSpec {
                validator: "explicit",
                kind: ValidationKind::Other,
            }),
            _ => None,
        })
        .or_else(|| classify_validator(&report.validator_name, content));

    // If the validator maps to a known tool, reuse the stream parser path so
    // Path A and Path B produce equivalent event chains.
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
        // Explicit reports are allowed to override parser-derived success when
        // the client has stronger knowledge than our generic text parser.
        let override_fp = (!success).then(|| fingerprint(content));
        for event in &mut events {
            event.metadata["success"] = json!(success);
            if is_failure_outcome_event_type(&event.event_type) {
                let payload = event.metadata["payload"].clone();
                event.metadata["payload"] = inline_signature_payload_from_fingerprint(
                    payload,
                    success,
                    override_fp.as_ref(),
                );
                event.summary =
                    event_summary(&event.event_type, success, &event.metadata["payload"]);
                event.evidence = Some(summarize_text(&event.metadata["payload"].to_string(), 1000));
            }
        }
    }
    Ok(events)
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
            trajectory: None,
        }
    }

    #[test]
    fn event_types_are_defined_in_one_place() {
        assert_eq!(EXECUTION_EVENT_TYPES.len(), 7);
        assert!(EXECUTION_EVENT_TYPES.contains(&EVENT_TYPE_TOOL_RESULT));
        assert!(!EXECUTION_EVENT_TYPES.contains(&"failure_signature"));
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
        assert!(!events
            .iter()
            .any(|event| event.event_type == "failure_signature"));
        let test_event = events
            .iter()
            .find(|event| event.event_type == EVENT_TYPE_TEST_RESULT)
            .unwrap();
        assert_eq!(test_event.metadata["payload"]["signature"], "unknown");
        assert_eq!(
            test_event.metadata["payload"]["signature_category"],
            "unknown"
        );
        assert_eq!(
            test_event.metadata["payload"]["fingerprint_version"],
            FINGERPRINT_VERSION
        );
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
            if is_failure_outcome_event_type(event_type) {
                assert!(event.metadata["payload"]["signature"].is_null());
                assert!(event.metadata["payload"]["signature_category"].is_null());
                assert!(event.metadata["payload"]["fingerprint_version"].is_null());
            }
        }
    }

    #[test]
    fn failed_outcome_events_carry_inline_signatures() {
        let payload =
            compile_result_payload("rust", "cargo", 101, "error[E0308]: mismatched types");
        let event =
            build_execution_event(&ctx(), ExecutionEventKind::CompileResult, false, payload);

        assert_eq!(event.metadata["payload"]["signature"], "rust:type-mismatch");
        assert_eq!(
            event.metadata["payload"]["signature_category"],
            "type_error"
        );
        assert_eq!(
            event.metadata["payload"]["fingerprint_version"],
            FINGERPRINT_VERSION
        );
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
            trajectory_id: None,
            attempt_index: None,
        };
        let events = events_for_validation_report(&ctx(), &report).unwrap();
        assert!(events
            .iter()
            .any(|event| event.event_type == EVENT_TYPE_TOOL_RESULT));
        assert!(events
            .iter()
            .any(|event| event.event_type == EVENT_TYPE_TEST_RESULT));
        assert!(!events
            .iter()
            .any(|event| event.event_type == "failure_signature"));
        let test_event = events
            .iter()
            .find(|event| event.event_type == EVENT_TYPE_TEST_RESULT)
            .unwrap();
        assert_eq!(test_event.metadata["payload"]["signature"], "unknown");
        assert_eq!(test_event.metadata["payload"]["validator_type"], "test");
        assert!(test_event.metadata["payload"]
            .get("failure_excerpt")
            .is_some());
    }

    #[test]
    fn invalid_validator_type_is_rejected() {
        let report = ValidationReportRequest {
            session_id: Some("session".to_string()),
            repo: "repo".to_string(),
            task: "task".to_string(),
            actor: None,
            event_type: None,
            validator_name: "custom".to_string(),
            validator_type: Some("free_form".to_string()),
            success: Some(true),
            exit_code: Some(0),
            content: Some("ok".to_string()),
            payload: None,
            correlation_id: Some(ctx().correlation_id),
            parent_event_id: None,
            trajectory_id: None,
            attempt_index: None,
        };
        assert!(events_for_validation_report(&ctx(), &report).is_err());
    }

    #[test]
    fn patch_payload_includes_lineage_metadata_fields() {
        let payload = patch_result_payload(vec!["src/lib.rs".to_string()], "applied", vec![]);
        assert_eq!(payload["files_touched"][0], "src/lib.rs");
        assert_eq!(payload["lines_added"], 0);
        assert_eq!(payload["lines_removed"], 0);
        assert_eq!(payload["patch_applied"], true);
        assert_eq!(payload["patch_reverted"], false);
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
