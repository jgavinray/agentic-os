use std::collections::{BTreeMap, BTreeSet};

use chrono::{DateTime, Utc};
use deadpool_postgres::Pool;
use serde_json::Value;
use tokio_postgres::types::ToSql;
use uuid::Uuid;

use crate::client_capture::RawHttpCapture;
use crate::prompt_intervention_assembly::records_from_capture;
use crate::prompt_intervention_detector::{DETECTOR_VERSION, TAXONOMY_VERSION};
use crate::prompt_intervention_fingerprint::PROMPT_FINGERPRINT_VERSION;
use crate::prompt_intervention_records::{
    insert, insert_backfill_summary, PromptInterventionBackfillSummary, PromptInterventionRecord,
};
use crate::prompt_intervention_taxonomy::LabelerType;

#[derive(Clone, Debug, Default)]
pub struct BackfillOptions {
    pub since: Option<DateTime<Utc>>,
    pub until: Option<DateTime<Utc>>,
    pub requested_model: Option<String>,
    pub response_model: Option<String>,
    pub repo: Option<String>,
    pub namespace: Option<String>,
    pub dry_run: bool,
    pub batch_size: i64,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct BackfillReport {
    pub rows_scanned: u64,
    pub labels_detected: u64,
    pub labels_written: u64,
    pub duplicates_skipped: u64,
    pub assembly_errors: u64,
    pub dry_run: bool,
    pub batch_size: i64,
    pub by_intervention_type: BTreeMap<String, u64>,
    pub by_signal_family: BTreeMap<String, u64>,
    pub by_burden_type: BTreeMap<String, u64>,
    pub by_failure_relation: BTreeMap<String, u64>,
    pub by_confidence_bucket: BTreeMap<String, u64>,
}

#[derive(Clone, Debug)]
struct RawExchangeRow {
    exchange_id: Uuid,
    attempt_id: Option<Uuid>,
    endpoint: String,
    method: String,
    path: String,
    namespace: Option<String>,
    repo: Option<String>,
    task: Option<String>,
    request_headers: Value,
    raw_request_body: Vec<u8>,
    parsed_request_body: Option<Value>,
    forwarded_request_body: Option<Vec<u8>>,
    response_status: Option<i32>,
    response_headers: Option<Value>,
    raw_response_body: Option<Vec<u8>>,
}

pub async fn run_backfill(
    pool: &Pool,
    options: &BackfillOptions,
) -> Result<BackfillReport, anyhow::Error> {
    let result = run_backfill_inner(pool, options).await;
    match &result {
        Ok(report) if report.rows_scanned == 0 => {
            crate::telemetry_prompt_interventions::record_prompt_intervention_backfill_run(
                "skipped",
            );
        }
        Ok(_) => {
            crate::telemetry_prompt_interventions::record_prompt_intervention_backfill_run(
                "success",
            );
        }
        Err(_) => {
            crate::telemetry_prompt_interventions::record_prompt_intervention_backfill_run("error");
        }
    }
    result
}

async fn run_backfill_inner(
    pool: &Pool,
    options: &BackfillOptions,
) -> Result<BackfillReport, anyhow::Error> {
    let started_at = Utc::now();
    let mut report = BackfillReport {
        dry_run: options.dry_run,
        batch_size: options.batch_size,
        ..BackfillReport::default()
    };
    let batch_size = options.batch_size.max(1);
    let mut offset = 0i64;

    loop {
        let rows = load_raw_exchange_batch(pool, options, batch_size, offset).await?;
        if rows.is_empty() {
            break;
        }
        offset += rows.len() as i64;

        for row in rows {
            report.rows_scanned += 1;
            if !response_model_matches(row.raw_response_body.as_deref(), &options.response_model) {
                continue;
            }
            let capture = row.into_capture();
            let records = match records_from_capture(&capture) {
                Ok(records) => records,
                Err(error) => {
                    report.assembly_errors += 1;
                    tracing::warn!(
                        exchange_id = %capture.exchange_id,
                        endpoint = %capture.endpoint,
                        "failed to assemble prompt intervention records during backfill: {error}"
                    );
                    continue;
                }
            };
            for record in records {
                if !record_matches_model_filters(&record, options) {
                    continue;
                }
                observe_record(&mut report, &record);
                if record_exists(pool, &record).await? {
                    report.duplicates_skipped += 1;
                    continue;
                }
                if options.dry_run {
                    report.labels_written += 1;
                    continue;
                }
                insert(pool, &record).await?;
                report.labels_written += 1;
            }
        }
    }

    insert_backfill_summary(
        pool,
        &summary_from_report(options, &report, started_at, Utc::now().max(started_at)),
    )
    .await?;

    Ok(report)
}

fn summary_from_report(
    options: &BackfillOptions,
    report: &BackfillReport,
    started_at: DateTime<Utc>,
    completed_at: DateTime<Utc>,
) -> PromptInterventionBackfillSummary {
    let records_inserted = if options.dry_run {
        0
    } else {
        report.labels_written
    };
    let label_note = if options.dry_run {
        format!("labels_would_write={}", report.labels_written)
    } else {
        format!("labels_written={}", report.labels_written)
    };

    PromptInterventionBackfillSummary {
        run_id: Uuid::new_v4(),
        detector_version: DETECTOR_VERSION.to_string(),
        prompt_fingerprint_version: PROMPT_FINGERPRINT_VERSION,
        filter_summary: filter_summary(options),
        records_inserted,
        exchanges_scanned: report.rows_scanned,
        labels_detected: report.labels_detected,
        taxonomy_version: TAXONOMY_VERSION.to_string(),
        labeler_type: LabelerType::Rule,
        status: "completed".to_string(),
        started_at,
        completed_at,
        notes: Some(format!(
            "dry_run={} {label_note} duplicates_skipped={} assembly_errors={}",
            options.dry_run, report.duplicates_skipped, report.assembly_errors
        )),
    }
}

fn filter_summary(options: &BackfillOptions) -> Value {
    serde_json::json!({
        "since": options.since.as_ref().map(DateTime::to_rfc3339),
        "until": options.until.as_ref().map(DateTime::to_rfc3339),
        "requested_model": options.requested_model.as_deref(),
        "response_model": options.response_model.as_deref(),
        "repo": options.repo.as_deref(),
        "namespace": options.namespace.as_deref(),
        "dry_run": options.dry_run,
        "batch_size": options.batch_size
    })
}

async fn load_raw_exchange_batch(
    pool: &Pool,
    options: &BackfillOptions,
    limit: i64,
    offset: i64,
) -> Result<Vec<RawExchangeRow>, anyhow::Error> {
    let conn = pool.get().await?;
    let params: [&(dyn ToSql + Sync); 6] = [
        &options.since,
        &options.until,
        &options.repo,
        &options.namespace,
        &limit,
        &offset,
    ];
    let rows = conn
        .query(
            "SELECT exchange_id, attempt_id, endpoint, method, path, namespace, repo, task,
                    request_headers, raw_request_body, parsed_request_body, forwarded_request_body,
                    response_status, response_headers, raw_response_body
             FROM raw_http_exchanges
             WHERE ($1::timestamptz IS NULL OR received_at >= $1)
               AND ($2::timestamptz IS NULL OR received_at <= $2)
               AND ($3::text IS NULL OR repo = $3)
               AND ($4::text IS NULL OR namespace = $4)
             ORDER BY received_at ASC, exchange_id ASC
             LIMIT $5 OFFSET $6",
            &params,
        )
        .await?;

    Ok(rows
        .into_iter()
        .map(|row| RawExchangeRow {
            exchange_id: row.get("exchange_id"),
            attempt_id: row.get("attempt_id"),
            endpoint: row.get("endpoint"),
            method: row.get("method"),
            path: row.get("path"),
            namespace: row.get("namespace"),
            repo: row.get("repo"),
            task: row.get("task"),
            request_headers: row.get("request_headers"),
            raw_request_body: row.get("raw_request_body"),
            parsed_request_body: row.get("parsed_request_body"),
            forwarded_request_body: row.get("forwarded_request_body"),
            response_status: row.get("response_status"),
            response_headers: row.get("response_headers"),
            raw_response_body: row.get("raw_response_body"),
        })
        .collect())
}

impl RawExchangeRow {
    fn into_capture(self) -> RawHttpCapture {
        RawHttpCapture {
            exchange_id: self.exchange_id,
            attempt_id: self.attempt_id,
            endpoint: self.endpoint,
            method: self.method,
            path: self.path,
            namespace: self.namespace,
            repo: self.repo,
            task: self.task,
            request_headers: self.request_headers,
            raw_request_body: self.raw_request_body,
            parsed_request_body: self.parsed_request_body,
            forwarded_request_body: self.forwarded_request_body,
            response_status: self.response_status,
            response_headers: self.response_headers,
            raw_response_body: self.raw_response_body,
        }
    }
}

fn record_matches_model_filters(
    record: &PromptInterventionRecord,
    options: &BackfillOptions,
) -> bool {
    optional_string_matches(&record.requested_model, &options.requested_model)
}

fn response_model_matches(body: Option<&[u8]>, expected: &Option<String>) -> bool {
    let Some(expected) = expected else {
        return true;
    };
    response_model(body)
        .as_deref()
        .map(|model| model == expected)
        .unwrap_or(false)
}

fn response_model(body: Option<&[u8]>) -> Option<String> {
    let body = body?;
    if let Ok(json) = serde_json::from_slice::<Value>(body) {
        if let Some(model) = response_model_from_json(&json) {
            return Some(model);
        }
    }

    let text = std::str::from_utf8(body).ok()?;
    let mut model = None;
    for line in text.lines() {
        let trimmed = line.trim();
        let Some(data) = trimmed.strip_prefix("data:") else {
            continue;
        };
        let data = data.trim();
        if data.is_empty() || data == "[DONE]" {
            continue;
        }
        let Ok(json) = serde_json::from_str::<Value>(data) else {
            continue;
        };
        if let Some(found) = response_model_from_json(&json) {
            model = Some(found);
        }
    }
    model
}

fn response_model_from_json(json: &Value) -> Option<String> {
    string_at_path(json, &["model"])
        .or_else(|| string_at_path(json, &["model_name"]))
        .or_else(|| string_at_path(json, &["message", "model"]))
        .map(str::to_string)
}

fn string_at_path<'a>(json: &'a Value, path: &[&str]) -> Option<&'a str> {
    let mut current = json;
    for segment in path {
        current = current.get(*segment)?;
    }
    current.as_str().filter(|value| !value.is_empty())
}

fn optional_string_matches(actual: &Option<String>, expected: &Option<String>) -> bool {
    match expected {
        Some(expected) => actual.as_deref() == Some(expected.as_str()),
        None => true,
    }
}

fn observe_record(report: &mut BackfillReport, record: &PromptInterventionRecord) {
    report.labels_detected += 1;
    increment(
        &mut report.by_intervention_type,
        record.intervention_type.as_str(),
    );
    increment(&mut report.by_signal_family, record.signal_family.as_str());
    increment(&mut report.by_burden_type, record.burden_type.as_str());
    increment(
        &mut report.by_failure_relation,
        record.failure_relation.as_str(),
    );
    let bucket = confidence_bucket(record.confidence);
    increment(&mut report.by_confidence_bucket, bucket);
}

fn increment(map: &mut BTreeMap<String, u64>, key: &str) {
    *map.entry(key.to_string()).or_insert(0) += 1;
}

fn confidence_bucket(confidence: f64) -> &'static str {
    if confidence >= 0.9 {
        "0.9-1.0"
    } else if confidence >= 0.8 {
        "0.8-0.89"
    } else if confidence >= 0.5 {
        "0.5-0.79"
    } else {
        "0.0-0.49"
    }
}

async fn record_exists(
    pool: &Pool,
    record: &PromptInterventionRecord,
) -> Result<bool, anyhow::Error> {
    let conn = pool.get().await?;
    let exists = conn
        .query_opt(
            "SELECT 1
             FROM prompt_interventions
             WHERE exchange_id = $1
               AND intervention_type = $2
               AND evidence_hash = $3
               AND taxonomy_version = $4
             LIMIT 1",
            &[
                &record.exchange_id,
                &record.intervention_type.as_str(),
                &record.evidence_hash,
                &record.taxonomy_version,
            ],
        )
        .await?
        .is_some();
    Ok(exists)
}

pub fn report_lines(report: &BackfillReport) -> Vec<String> {
    let mut lines = vec![format!(
        "prompt-intervention-backfill: rows_scanned={} labels_detected={} labels_written={} duplicates_skipped={} assembly_errors={} dry_run={} batch_size={}",
        report.rows_scanned,
        report.labels_detected,
        report.labels_written,
        report.duplicates_skipped,
        report.assembly_errors,
        report.dry_run,
        report.batch_size
    )];
    push_counts(
        &mut lines,
        "intervention_type",
        &report.by_intervention_type,
    );
    push_counts(&mut lines, "signal_family", &report.by_signal_family);
    push_counts(&mut lines, "burden_type", &report.by_burden_type);
    push_counts(&mut lines, "failure_relation", &report.by_failure_relation);
    push_counts(
        &mut lines,
        "confidence_bucket",
        &report.by_confidence_bucket,
    );
    lines
}

fn push_counts(lines: &mut Vec<String>, label: &str, counts: &BTreeMap<String, u64>) {
    let rendered = counts
        .iter()
        .map(|(key, value)| format!("{key}={value}"))
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect::<Vec<_>>()
        .join(",");
    lines.push(format!("{label}: {rendered}"));
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::prompt_intervention_records::sha256_hex;
    use crate::prompt_intervention_taxonomy::{
        BurdenType, FailureRelation, InterventionType, SignalFamily, SignalStrength, SourceKind,
    };

    #[test]
    fn response_model_filter_matches_json_response() {
        let body = br#"{"model":"qwen3.6-27b","content":[]}"#;
        assert!(response_model_matches(
            Some(body),
            &Some("qwen3.6-27b".to_string())
        ));
        assert!(!response_model_matches(
            Some(body),
            &Some("gemma-31b".to_string())
        ));
    }

    #[test]
    fn response_model_filter_matches_openai_sse_response() {
        let body = br#"event: completion
data: {"id":"chatcmpl-test","model":"qwen36-35b-heretic","choices":[]}
data: [DONE]
"#;

        assert!(response_model_matches(
            Some(body),
            &Some("qwen36-35b-heretic".to_string())
        ));
        assert!(!response_model_matches(
            Some(body),
            &Some("gemma-31b".to_string())
        ));
    }

    #[test]
    fn response_model_filter_matches_anthropic_message_start_sse_response() {
        let body =
            br#"data: {"type":"message_start","message":{"id":"msg_test","model":"gemma-31b"}}
data: {"type":"content_block_delta","delta":{"type":"text_delta","text":"ok"}}
data: [DONE]"#;

        assert!(response_model_matches(
            Some(body),
            &Some("gemma-31b".to_string())
        ));
    }

    #[test]
    fn response_model_filter_tolerates_partial_sse_without_trailing_newline() {
        let body = br#"data: {"model_name":"qwen3.6-27b"}"#;

        assert!(response_model_matches(
            Some(body),
            &Some("qwen3.6-27b".to_string())
        ));
    }

    #[test]
    fn response_model_filter_uses_last_sse_model_value() {
        let body = br#"data: {"model":"old-model"}
data: {"model":"new-model"}
data: [DONE]
"#;

        assert!(!response_model_matches(
            Some(body),
            &Some("old-model".to_string())
        ));
        assert!(response_model_matches(
            Some(body),
            &Some("new-model".to_string())
        ));
    }

    #[test]
    fn response_model_filter_rejects_missing_or_non_string_model() {
        assert!(!response_model_matches(
            Some(br#"{"model":null}"#),
            &Some("qwen3.6-27b".to_string())
        ));
        assert!(!response_model_matches(
            Some(br#"data: {"choices":[]}"#),
            &Some("qwen3.6-27b".to_string())
        ));
        assert!(!response_model_matches(
            Some(b""),
            &Some("qwen3.6-27b".to_string())
        ));
    }

    #[test]
    fn confidence_buckets_are_bounded() {
        assert_eq!(confidence_bucket(0.95), "0.9-1.0");
        assert_eq!(confidence_bucket(0.8), "0.8-0.89");
        assert_eq!(confidence_bucket(0.5), "0.5-0.79");
        assert_eq!(confidence_bucket(0.49), "0.0-0.49");
    }

    #[test]
    fn observe_record_counts_required_dimensions() {
        let mut report = BackfillReport::default();
        observe_record(&mut report, &sample_record());
        assert_eq!(report.labels_detected, 1);
        assert_eq!(report.by_intervention_type["scope_narrowing"], 1);
        assert_eq!(report.by_signal_family["steering"], 1);
        assert_eq!(report.by_burden_type["human_scope_control"], 1);
        assert_eq!(report.by_failure_relation["prevention"], 1);
        assert_eq!(report.by_confidence_bucket["0.8-0.89"], 1);
    }

    #[test]
    fn report_lines_include_dry_run_counts() {
        let mut report = BackfillReport {
            rows_scanned: 2,
            labels_detected: 1,
            assembly_errors: 1,
            dry_run: true,
            batch_size: 10,
            ..BackfillReport::default()
        };
        increment(&mut report.by_intervention_type, "scope_narrowing");
        let lines = report_lines(&report);
        assert!(lines[0].contains("rows_scanned=2"));
        assert!(lines[0].contains("labels_detected=1"));
        assert!(lines[0].contains("assembly_errors=1"));
        assert!(lines[0].contains("dry_run=true"));
        assert!(lines
            .iter()
            .any(|line| line == "intervention_type: scope_narrowing=1"));
    }

    #[test]
    fn dry_run_summary_records_no_inserted_records_and_would_write_note() {
        let started_at = Utc::now();
        let report = BackfillReport {
            rows_scanned: 5,
            labels_detected: 3,
            labels_written: 2,
            duplicates_skipped: 1,
            assembly_errors: 1,
            dry_run: true,
            batch_size: 10,
            ..BackfillReport::default()
        };
        let options = BackfillOptions {
            dry_run: true,
            batch_size: 10,
            response_model: Some("qwen36-35b-heretic".to_string()),
            ..BackfillOptions::default()
        };

        let summary = summary_from_report(&options, &report, started_at, started_at);

        assert_eq!(summary.records_inserted, 0);
        assert_eq!(summary.exchanges_scanned, 5);
        assert_eq!(summary.labels_detected, 3);
        assert_eq!(summary.detector_version, DETECTOR_VERSION);
        assert_eq!(summary.taxonomy_version, TAXONOMY_VERSION);
        assert_eq!(
            summary.prompt_fingerprint_version,
            PROMPT_FINGERPRINT_VERSION
        );
        assert_eq!(
            summary.filter_summary["response_model"],
            "qwen36-35b-heretic"
        );
        let notes = summary.notes.as_deref().unwrap_or_default();
        assert!(notes.contains("dry_run=true"));
        assert!(notes.contains("labels_would_write=2"));
        assert!(notes.contains("duplicates_skipped=1"));
        assert!(notes.contains("assembly_errors=1"));
    }

    #[test]
    fn write_summary_records_inserted_records() {
        let started_at = Utc::now();
        let report = BackfillReport {
            rows_scanned: 5,
            labels_detected: 3,
            labels_written: 2,
            dry_run: false,
            batch_size: 10,
            ..BackfillReport::default()
        };
        let options = BackfillOptions {
            dry_run: false,
            batch_size: 10,
            ..BackfillOptions::default()
        };

        let summary = summary_from_report(&options, &report, started_at, started_at);

        assert_eq!(summary.records_inserted, 2);
        assert_eq!(summary.exchanges_scanned, 5);
        assert_eq!(summary.labels_detected, 3);
        let notes = summary.notes.as_deref().unwrap_or_default();
        assert!(notes.contains("dry_run=false"));
        assert!(notes.contains("labels_written=2"));
    }

    fn sample_record() -> PromptInterventionRecord {
        let evidence_excerpt = "Edit only the LiteLLM config.".to_string();
        PromptInterventionRecord {
            id: Uuid::new_v4(),
            exchange_id: Uuid::new_v4(),
            trajectory_id: None,
            request_event_id: None,
            attempt_id: None,
            requested_model: Some("claude-opus-4-8".to_string()),
            routed_model: Some("qwen3.6-27b".to_string()),
            baseline_arm: None,
            selected_route: Some("claude_opus_tier".to_string()),
            routing_policy_version: None,
            exact_prompt_hash: sha256_hex("exact"),
            normalized_prompt_hash: sha256_hex("normalized"),
            prompt_fingerprint_version: 1,
            source_kind: SourceKind::UserMessage,
            intervention_type: InterventionType::ScopeNarrowing,
            signal_family: SignalFamily::Steering,
            signal_type: "scope_narrowing".to_string(),
            signal_strength: SignalStrength::Explicit,
            burden_type: BurdenType::HumanScopeControl,
            failure_relation: FailureRelation::Prevention,
            target_behavior: None,
            blocked_behavior: None,
            replacement_behavior: None,
            evidence_hash: sha256_hex(&evidence_excerpt),
            evidence_excerpt,
            labeler_type: LabelerType::Rule,
            confidence: 0.85,
            taxonomy_version: TAXONOMY_VERSION.to_string(),
            supersedes_record_id: None,
            created_at: Utc::now(),
        }
    }
}
