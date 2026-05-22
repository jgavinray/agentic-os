//! Deterministic harness feedback classification.
//!
//! The harness feedback layer records bounded execution-harness signals on
//! `agent_events` metadata. It does not replace the audit log and it never
//! changes prompt assembly directly; quarantine only prevents known poisoned
//! harness artifacts from being reused as memory.

use serde_json::{json, Value};
use std::collections::BTreeSet;

pub const HARNESS_FEEDBACK_SCHEMA_VERSION: u32 = 1;

pub const HARNESS_SIGNAL_TYPES: [&str; 20] = [
    "benchmark_trace",
    "invalid_path",
    "repeated_invalid_path",
    "path_typo_context_ledger",
    "read_missing_path",
    "edit_failed",
    "gate_blocked",
    "gate_recovery_failed",
    "gate_recovered",
    "forbidden_tool_fallback",
    "command_boundary_violation",
    "patch_invalid",
    "fake_success_claim",
    "no_progress_timeout",
    "diagnosis_success",
    "hard_stop_compliance",
    "scope_violation",
    "misleading_green_tests",
    "memory_poison_indicator",
    "unknown",
];

pub const HARNESS_QUARANTINE_REASONS: [&str; 8] = [
    "benchmark_session",
    "poisoned_path",
    "repeated_invalid_path",
    "fake_success_claim",
    "tool_recovery_failed",
    "patch_invalid",
    "manual_quarantine",
    "memory_poison_indicator",
];

pub const HARNESS_LEARNING_STATUSES: [&str; 3] = ["recorded", "quarantined", "ignored"];

const MEMORY_POLICY_INCLUDE: &str = "include";
const MEMORY_POLICY_EXCLUDE: &str = "exclude_from_context";

#[derive(Debug, Clone, Default)]
pub struct BackfillOptions {
    pub repo: Option<String>,
    pub session_id: Option<String>,
    pub since: Option<chrono::DateTime<chrono::Utc>>,
    pub dry_run: bool,
    pub batch_size: i64,
}

#[derive(Debug, Clone, Default)]
pub struct BackfillReport {
    pub events_scanned: i64,
    pub updated: i64,
    pub quarantined: i64,
    pub dry_run: bool,
    pub batch_size: i64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct HarnessClassification {
    signals: BTreeSet<&'static str>,
    quarantined: bool,
    quarantine_reason: Option<&'static str>,
}

impl HarnessClassification {
    fn empty() -> Self {
        Self {
            signals: BTreeSet::new(),
            quarantined: false,
            quarantine_reason: None,
        }
    }

    fn signal(&mut self, signal: &'static str) {
        self.signals.insert(signal);
    }

    fn quarantine(&mut self, reason: &'static str) {
        self.quarantined = true;
        if self.quarantine_reason.is_none() {
            self.quarantine_reason = Some(reason);
        }
    }
}

pub fn annotate_event_metadata(
    event_type: &str,
    summary: &str,
    evidence: Option<&str>,
    metadata: Value,
) -> Value {
    let classification = classify_event(event_type, summary, evidence, &metadata);
    if classification.signals.is_empty()
        && !classification.quarantined
        && metadata.get("harness_feedback").is_none()
    {
        return metadata;
    }

    let mut metadata = object_or_empty(metadata);
    let mut feedback = metadata
        .get("harness_feedback")
        .and_then(Value::as_object)
        .cloned()
        .unwrap_or_default();

    let signals: Vec<Value> = classification
        .signals
        .iter()
        .map(|signal| Value::String((*signal).to_string()))
        .collect();
    let learning_status = if classification.quarantined {
        "quarantined"
    } else if signals.is_empty() {
        "ignored"
    } else {
        "recorded"
    };

    feedback.insert(
        "schema_version".to_string(),
        Value::Number(HARNESS_FEEDBACK_SCHEMA_VERSION.into()),
    );
    feedback.insert("signals".to_string(), Value::Array(signals));
    feedback.insert(
        "signal_count".to_string(),
        Value::Number((classification.signals.len() as u64).into()),
    );
    feedback.insert(
        "quarantined".to_string(),
        Value::Bool(classification.quarantined),
    );
    feedback.insert(
        "quarantine_reason".to_string(),
        classification
            .quarantine_reason
            .map(|reason| Value::String(reason.to_string()))
            .unwrap_or(Value::Null),
    );
    feedback.insert(
        "memory_policy".to_string(),
        Value::String(
            if classification.quarantined {
                MEMORY_POLICY_EXCLUDE
            } else {
                MEMORY_POLICY_INCLUDE
            }
            .to_string(),
        ),
    );
    feedback.insert(
        "learning_status".to_string(),
        Value::String(learning_status.to_string()),
    );

    metadata["harness_feedback"] = Value::Object(feedback);
    metadata
}

pub fn is_quarantined(metadata: &Value) -> bool {
    metadata
        .get("harness_feedback")
        .and_then(|value| value.get("quarantined"))
        .and_then(Value::as_bool)
        .unwrap_or(false)
        || metadata
            .get("harness_feedback")
            .and_then(|value| value.get("memory_policy"))
            .and_then(Value::as_str)
            .is_some_and(|policy| policy == MEMORY_POLICY_EXCLUDE)
}

pub fn record_metadata_metrics(metadata: &Value) {
    let Some(feedback) = metadata.get("harness_feedback") else {
        return;
    };
    let signals = feedback
        .get("signals")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    for signal in signals {
        if let Some(signal) = signal.as_str() {
            crate::telemetry::record_harness_feedback_signal(signal);
        }
    }
    if is_quarantined(metadata) {
        let reason = feedback
            .get("quarantine_reason")
            .and_then(Value::as_str)
            .unwrap_or("manual_quarantine");
        crate::telemetry::record_harness_feedback_quarantine(reason);
    }
    let status = feedback
        .get("learning_status")
        .and_then(Value::as_str)
        .unwrap_or("ignored");
    crate::telemetry::record_harness_feedback_learning(status);
}

pub async fn run_backfill(
    pool: &deadpool_postgres::Pool,
    opts: &BackfillOptions,
) -> Result<BackfillReport, anyhow::Error> {
    let batch_size = opts.batch_size.max(1);
    let mut report = BackfillReport {
        dry_run: opts.dry_run,
        batch_size,
        ..BackfillReport::default()
    };
    let mut last_created_at: Option<chrono::DateTime<chrono::Utc>> = None;
    let mut last_id: Option<String> = None;

    loop {
        let conn = pool.get().await?;
        let rows = conn
            .query(
                "SELECT id, event_type, summary, evidence, metadata, created_at
                 FROM agent_events
                 WHERE ($1::TEXT IS NULL OR repo = $1)
                   AND ($2::TEXT IS NULL OR session_id = $2)
                   AND ($3::TIMESTAMPTZ IS NULL OR created_at >= $3)
                   AND (
                       $4::TIMESTAMPTZ IS NULL
                       OR created_at > $4
                       OR (created_at = $4 AND id > $5)
                   )
                 ORDER BY created_at ASC, id ASC
                 LIMIT $6",
                &[
                    &opts.repo,
                    &opts.session_id,
                    &opts.since,
                    &last_created_at,
                    &last_id,
                    &batch_size,
                ],
            )
            .await?;
        if rows.is_empty() {
            break;
        }

        for row in rows {
            let id: String = row.get("id");
            let event_type: String = row.get("event_type");
            let summary: String = row.get("summary");
            let evidence: Option<String> = row.get("evidence");
            let metadata: Value = row.get("metadata");
            let created_at: chrono::DateTime<chrono::Utc> = row.get("created_at");

            report.events_scanned += 1;
            last_created_at = Some(created_at);
            last_id = Some(id.clone());

            let updated = annotate_event_metadata(
                &event_type,
                &summary,
                evidence.as_deref(),
                metadata.clone(),
            );
            if updated == metadata {
                continue;
            }
            report.updated += 1;
            if is_quarantined(&updated) {
                report.quarantined += 1;
            }
            if opts.dry_run {
                continue;
            }
            conn.execute(
                "UPDATE agent_events
                 SET metadata = $2
                 WHERE id = $1 AND metadata IS DISTINCT FROM $2",
                &[&id, &updated],
            )
            .await?;
            record_metadata_metrics(&updated);
        }
    }

    Ok(report)
}

fn classify_event(
    event_type: &str,
    summary: &str,
    evidence: Option<&str>,
    metadata: &Value,
) -> HarnessClassification {
    let mut classification = HarnessClassification::empty();
    collect_structured_metadata(metadata, &mut classification);
    collect_text_signals(event_type, summary, evidence, &mut classification);
    apply_quarantine_rules(&mut classification);
    classification
}

fn collect_structured_metadata(metadata: &Value, classification: &mut HarnessClassification) {
    if bool_path(metadata, &["benchmark"]) || bool_path(metadata, &["harness", "benchmark"]) {
        classification.signal("benchmark_trace");
    }

    for path in [
        &["harness_feedback", "signals"][..],
        &["harness", "signals"][..],
        &["harness", "failure_signals"][..],
    ] {
        for signal in string_array_path(metadata, path) {
            classification.signal(bounded_signal_type(signal));
        }
    }

    for path in [
        &["harness_feedback", "signal"][..],
        &["harness", "signal"][..],
        &["harness", "failure_class"][..],
    ] {
        if let Some(signal) = string_path(metadata, path) {
            classification.signal(bounded_signal_type(signal));
        }
    }

    if bool_path(metadata, &["harness_feedback", "quarantined"])
        || bool_path(metadata, &["harness", "quarantine"])
        || bool_path(metadata, &["harness", "quarantined"])
    {
        let reason = string_path(metadata, &["harness_feedback", "quarantine_reason"])
            .or_else(|| string_path(metadata, &["harness", "quarantine_reason"]))
            .map(bounded_quarantine_reason)
            .unwrap_or("manual_quarantine");
        classification.quarantine(reason);
    }
}

fn collect_text_signals(
    event_type: &str,
    summary: &str,
    evidence: Option<&str>,
    classification: &mut HarnessClassification,
) {
    let haystack = format!(
        "{}\n{}\n{}",
        event_type,
        summary,
        evidence.unwrap_or_default()
    )
    .to_ascii_lowercase();

    if contains_any(&haystack, &["context_leder", "context_ledler"]) {
        classification.signal("path_typo_context_ledger");
    }
    if contains_any(
        &haystack,
        &[
            "no such file or directory",
            "file not found",
            "read tool: file",
        ],
    ) && contains_any(
        &haystack,
        &["context_ledger", "context_leder", "context_ledler", ".tmp/"],
    ) {
        classification.signal("read_missing_path");
    }
    if contains_any(&haystack, &["error writing file", "error editing file"]) {
        classification.signal("edit_failed");
    }
    if contains_any(
        &haystack,
        &["[fact-forcing gate]", "gateguard", "fact-forcing gate"],
    ) {
        classification.signal("gate_blocked");
    }
    if haystack.contains("gate") && contains_any(&haystack, &["fallback", "failed to recover"]) {
        classification.signal("gate_recovery_failed");
    }
    if haystack.contains("gate") && contains_any(&haystack, &["recovered", "satisfy gateguard"]) {
        classification.signal("gate_recovered");
    }
    if contains_any(&haystack, &["do not use bash", "bash fallback"]) {
        classification.signal("forbidden_tool_fallback");
    }
    if contains_any(
        &haystack,
        &["do not retry", "do not search", "do not guess paths"],
    ) && contains_any(&haystack, &["retry", "searched for", "finds the actual"])
    {
        classification.signal("command_boundary_violation");
    }
    if contains_any(
        &haystack,
        &["invalid patch", "fake hash", "a1b2c3d4e5f60718"],
    ) || (haystack.contains("unified diff patch")
        && contains_any(&haystack, &["context_leder", "context_ledler"]))
    {
        classification.signal("patch_invalid");
    }
    if contains_any(
        &haystack,
        &["claimed success", "false confidence", "misleading"],
    ) && contains_any(&haystack, &["test", "green", "passed"])
    {
        classification.signal("fake_success_claim");
    }
    if contains_any(
        &haystack,
        &["made no progress", "no progress", "still incomplete"],
    ) {
        classification.signal("no_progress_timeout");
    }
    if contains_any(&haystack, &["sanity check: passed", "sanity_check=passed"]) {
        classification.signal("diagnosis_success");
    }
    if contains_any(
        &haystack,
        &["hard-stop compliance", "stopped after the first failure"],
    ) {
        classification.signal("hard_stop_compliance");
    }
    if contains_any(
        &haystack,
        &[
            "scope creep",
            "outside the allowed files",
            "scope violation",
        ],
    ) {
        classification.signal("scope_violation");
    }
    if contains_any(
        &haystack,
        &["enum_variant_hash", "placeholder", "early return"],
    ) && contains_any(&haystack, &["test", "validation"])
    {
        classification.signal("misleading_green_tests");
    }
    if contains_any(
        &haystack,
        &[
            "poisoned memory",
            "poisoned cache",
            "memory/cache is perhaps poisoned",
        ],
    ) {
        classification.signal("memory_poison_indicator");
    }
}

fn apply_quarantine_rules(classification: &mut HarnessClassification) {
    if classification.signals.contains("path_typo_context_ledger") {
        classification.quarantine("poisoned_path");
    } else if classification.signals.contains("repeated_invalid_path") {
        classification.quarantine("repeated_invalid_path");
    } else if classification.signals.contains("patch_invalid") {
        classification.quarantine("patch_invalid");
    } else if classification.signals.contains("fake_success_claim") {
        classification.quarantine("fake_success_claim");
    } else if classification.signals.contains("gate_recovery_failed")
        || classification.signals.contains("forbidden_tool_fallback")
        || classification
            .signals
            .contains("command_boundary_violation")
        || classification.signals.contains("no_progress_timeout")
    {
        classification.quarantine("tool_recovery_failed");
    } else if classification.signals.contains("memory_poison_indicator") {
        classification.quarantine("memory_poison_indicator");
    } else if classification.signals.contains("benchmark_trace") {
        classification.quarantine("benchmark_session");
    }
}

pub fn bounded_signal_type(value: &str) -> &'static str {
    match value {
        "benchmark_trace" | "benchmark" => "benchmark_trace",
        "invalid_path" => "invalid_path",
        "repeated_invalid_path" => "repeated_invalid_path",
        "path_typo_context_ledger" | "context_leder" | "context_ledler" => {
            "path_typo_context_ledger"
        }
        "read_missing_path" | "missing_path" => "read_missing_path",
        "edit_failed" => "edit_failed",
        "gate_blocked" | "gateguard_blocked" => "gate_blocked",
        "gate_recovery_failed" => "gate_recovery_failed",
        "gate_recovered" => "gate_recovered",
        "forbidden_tool_fallback" => "forbidden_tool_fallback",
        "command_boundary_violation" => "command_boundary_violation",
        "patch_invalid" | "invalid_patch" => "patch_invalid",
        "fake_success_claim" => "fake_success_claim",
        "no_progress_timeout" | "no_progress" => "no_progress_timeout",
        "diagnosis_success" => "diagnosis_success",
        "hard_stop_compliance" => "hard_stop_compliance",
        "scope_violation" => "scope_violation",
        "misleading_green_tests" => "misleading_green_tests",
        "memory_poison_indicator" => "memory_poison_indicator",
        _ => "unknown",
    }
}

pub fn bounded_quarantine_reason(value: &str) -> &'static str {
    match value {
        "benchmark_session" => "benchmark_session",
        "poisoned_path" => "poisoned_path",
        "repeated_invalid_path" => "repeated_invalid_path",
        "fake_success_claim" => "fake_success_claim",
        "tool_recovery_failed" => "tool_recovery_failed",
        "patch_invalid" => "patch_invalid",
        "manual_quarantine" => "manual_quarantine",
        "memory_poison_indicator" => "memory_poison_indicator",
        _ => "manual_quarantine",
    }
}

pub fn bounded_learning_status(value: &str) -> &'static str {
    match value {
        "recorded" => "recorded",
        "quarantined" => "quarantined",
        "ignored" => "ignored",
        _ => "ignored",
    }
}

fn object_or_empty(value: Value) -> Value {
    if value.is_object() {
        value
    } else {
        json!({})
    }
}

fn contains_any(haystack: &str, needles: &[&str]) -> bool {
    needles.iter().any(|needle| haystack.contains(needle))
}

fn bool_path(value: &Value, path: &[&str]) -> bool {
    path_value(value, path)
        .and_then(Value::as_bool)
        .unwrap_or(false)
}

fn string_path<'a>(value: &'a Value, path: &[&str]) -> Option<&'a str> {
    path_value(value, path).and_then(Value::as_str)
}

fn string_array_path<'a>(value: &'a Value, path: &[&str]) -> Vec<&'a str> {
    path_value(value, path)
        .and_then(Value::as_array)
        .map(|values| values.iter().filter_map(Value::as_str).collect())
        .unwrap_or_default()
}

fn path_value<'a>(value: &'a Value, path: &[&str]) -> Option<&'a Value> {
    let mut current = value;
    for segment in path {
        current = current.get(*segment)?;
    }
    Some(current)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn context_ledger_path_typo_is_quarantined() {
        let metadata = annotate_event_metadata(
            "failed_attempt",
            "Read failed: /repo/orchestrator/src/context_leder/enums.rs not found",
            Some("No such file or directory"),
            json!({}),
        );

        let feedback = metadata.get("harness_feedback").unwrap();
        assert_eq!(feedback["schema_version"], HARNESS_FEEDBACK_SCHEMA_VERSION);
        assert_eq!(feedback["quarantined"], true);
        assert_eq!(feedback["quarantine_reason"], "poisoned_path");
        assert_eq!(feedback["memory_policy"], MEMORY_POLICY_EXCLUDE);
        assert!(feedback["signals"]
            .as_array()
            .unwrap()
            .contains(&Value::String("path_typo_context_ledger".to_string())));
        assert!(is_quarantined(&metadata));
    }

    #[test]
    fn structured_manual_quarantine_is_preserved_with_bounded_reason() {
        let metadata = annotate_event_metadata(
            "checkpoint",
            "benchmark run",
            None,
            json!({
                "harness": {
                    "signals": ["gate_blocked", "unknown-user-value"],
                    "quarantine": true,
                    "quarantine_reason": "raw user supplied reason"
                }
            }),
        );

        let feedback = metadata.get("harness_feedback").unwrap();
        assert_eq!(feedback["quarantined"], true);
        assert_eq!(feedback["quarantine_reason"], "manual_quarantine");
        assert!(feedback["signals"]
            .as_array()
            .unwrap()
            .contains(&Value::String("gate_blocked".to_string())));
        assert!(feedback["signals"]
            .as_array()
            .unwrap()
            .contains(&Value::String("unknown".to_string())));
    }

    #[test]
    fn ordinary_events_are_not_annotated() {
        let metadata =
            annotate_event_metadata("user_message", "please update docs", None, json!({}));
        assert!(metadata.get("harness_feedback").is_none());
        assert!(!is_quarantined(&metadata));
    }

    #[test]
    fn annotation_is_idempotent() {
        let once = annotate_event_metadata(
            "failed_attempt",
            "context_ledler path typo caused failed read",
            None,
            json!({}),
        );
        let twice = annotate_event_metadata(
            "failed_attempt",
            "context_ledler path typo caused failed read",
            None,
            once.clone(),
        );
        assert_eq!(once, twice);
    }
}
