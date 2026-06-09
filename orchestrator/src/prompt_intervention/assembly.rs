/// Assembly helpers for prompt intervention records.
///
/// This module converts captured request payloads into validated derived
/// records. It does not persist records or integrate with runtime handlers.
use chrono::Utc;
use serde_json::Value;
use uuid::Uuid;

use crate::client_capture::RawHttpCapture;
use crate::prompt_intervention_detector::detect_prompt_interventions;
use crate::prompt_intervention_extraction::extract_prompt_sources;
use crate::prompt_intervention_records::PromptInterventionRecord;
use crate::prompt_intervention_taxonomy::LabelerType;

/// Build validated prompt intervention records from one raw HTTP capture.
pub fn records_from_capture(
    capture: &RawHttpCapture,
) -> anyhow::Result<Vec<PromptInterventionRecord>> {
    let sources = extract_prompt_sources(
        capture.parsed_request_body.as_ref(),
        &capture.raw_request_body,
    );
    let detections = detect_prompt_interventions(&sources);
    if detections.is_empty() {
        return Ok(Vec::new());
    }

    let forwarded = forwarded_json(capture);
    let metadata = forwarded.as_ref().and_then(agentic_os_metadata);
    let mut records = Vec::with_capacity(detections.len());

    for detection in detections {
        let record = PromptInterventionRecord {
            id: Uuid::new_v4(),
            exchange_id: capture.exchange_id,
            trajectory_id: metadata.and_then(|m| uuid_field(m, "trajectory_id")),
            request_event_id: metadata.and_then(|m| uuid_field(m, "request_event_id")),
            attempt_id: capture
                .attempt_id
                .or_else(|| metadata.and_then(|m| uuid_field(m, "attempt_id"))),
            requested_model: metadata
                .and_then(|m| string_field(m, "requested_model"))
                .or_else(|| parsed_model(capture.parsed_request_body.as_ref())),
            routed_model: metadata
                .and_then(|m| string_field(m, "routed_model"))
                .or_else(|| forwarded_model(forwarded.as_ref())),
            baseline_arm: metadata.and_then(|m| string_field(m, "baseline_arm")),
            selected_route: metadata.and_then(|m| string_field(m, "selected_route")),
            routing_policy_version: metadata.and_then(|m| string_field(m, "policy_version")),
            exact_prompt_hash: detection.exact_prompt_hash,
            normalized_prompt_hash: detection.normalized_prompt_hash,
            prompt_fingerprint_version: detection.prompt_fingerprint_version,
            source_kind: detection.source_kind,
            intervention_type: detection.intervention_type,
            signal_family: detection.signal_family,
            signal_type: detection.signal_type,
            signal_strength: detection.signal_strength,
            burden_type: detection.burden_type,
            failure_relation: detection.failure_relation,
            target_behavior: detection.target_behavior,
            blocked_behavior: detection.blocked_behavior,
            replacement_behavior: detection.replacement_behavior,
            evidence_excerpt: detection.evidence_excerpt,
            evidence_hash: detection.evidence_hash,
            labeler_type: LabelerType::Rule,
            confidence: detection.confidence,
            taxonomy_version: detection.taxonomy_version.to_string(),
            supersedes_record_id: None,
            created_at: Utc::now(),
        };
        record.validate()?;
        records.push(record);
    }

    Ok(records)
}

fn forwarded_json(capture: &RawHttpCapture) -> Option<Value> {
    capture
        .forwarded_request_body
        .as_ref()
        .and_then(|body| serde_json::from_slice(body).ok())
}

fn agentic_os_metadata(forwarded: &Value) -> Option<&Value> {
    forwarded.get("metadata")?.get("agentic_os")
}

fn uuid_field(value: &Value, key: &str) -> Option<Uuid> {
    value.get(key)?.as_str()?.parse().ok()
}

fn string_field(value: &Value, key: &str) -> Option<String> {
    value
        .get(key)?
        .as_str()
        .filter(|s| !s.is_empty())
        .map(str::to_string)
}

fn parsed_model(parsed: Option<&Value>) -> Option<String> {
    parsed
        .and_then(|value| value.get("model"))
        .and_then(Value::as_str)
        .filter(|s| !s.is_empty())
        .map(str::to_string)
}

fn forwarded_model(forwarded: Option<&Value>) -> Option<String> {
    forwarded
        .and_then(|value| value.get("model"))
        .and_then(Value::as_str)
        .filter(|s| !s.is_empty())
        .map(str::to_string)
}

#[cfg(test)]
mod tests {
    use axum::http::HeaderMap;
    use serde_json::json;

    use super::*;
    use crate::client_capture::to_json_bytes;
    use crate::prompt_intervention_taxonomy::{InterventionType, LabelerType, SourceKind};

    fn capture_with_body(body: Value) -> RawHttpCapture {
        let raw = to_json_bytes(&body);
        let mut capture = RawHttpCapture::new("messages", &HeaderMap::new(), raw);
        capture.parsed_request_body = Some(body);
        capture
    }

    #[test]
    fn no_findings_returns_empty_records() {
        let capture = capture_with_body(json!({
            "model": "claude-opus-4-8",
            "messages": [{ "role": "user", "content": "hello" }]
        }));

        assert!(records_from_capture(&capture).unwrap().is_empty());
    }

    #[test]
    fn copies_route_and_lineage_metadata_from_forwarded_body() {
        let attempt_id = Uuid::new_v4();
        let request_event_id = Uuid::new_v4();
        let trajectory_id = Uuid::new_v4();
        let mut capture = capture_with_body(json!({
            "model": "claude-opus-4-8",
            "messages": [{ "role": "user", "content": "Edit only the LiteLLM config" }]
        }));
        capture.forwarded_request_body = Some(to_json_bytes(&json!({
            "model": "qwen3.6-27b",
            "metadata": {
                "agentic_os": {
                    "attempt_id": attempt_id,
                    "request_event_id": request_event_id,
                    "trajectory_id": trajectory_id,
                    "requested_model": "claude-opus-4-8",
                    "routed_model": "qwen3.6-27b",
                    "baseline_arm": "policy_enabled",
                    "selected_route": "claude_opus_tier",
                    "policy_version": "claude-tier-routing-v1"
                }
            }
        })));

        let records = records_from_capture(&capture).unwrap();
        assert_eq!(records.len(), 1);
        let record = &records[0];
        assert_eq!(record.exchange_id, capture.exchange_id);
        assert_eq!(record.attempt_id, Some(attempt_id));
        assert_eq!(record.request_event_id, Some(request_event_id));
        assert_eq!(record.trajectory_id, Some(trajectory_id));
        assert_eq!(record.requested_model.as_deref(), Some("claude-opus-4-8"));
        assert_eq!(record.routed_model.as_deref(), Some("qwen3.6-27b"));
        assert_eq!(record.baseline_arm.as_deref(), Some("policy_enabled"));
        assert_eq!(record.selected_route.as_deref(), Some("claude_opus_tier"));
        assert_eq!(
            record.routing_policy_version.as_deref(),
            Some("claude-tier-routing-v1")
        );
        assert_eq!(record.labeler_type, LabelerType::Rule);
        assert_eq!(record.source_kind, SourceKind::UserMessage);
        assert_eq!(record.intervention_type, InterventionType::ScopeNarrowing);
        record.validate().unwrap();
    }

    #[test]
    fn requested_model_falls_back_to_parsed_model() {
        let mut capture = capture_with_body(json!({
            "model": "claude-opus-4-8",
            "messages": [{ "role": "user", "content": "Do not implement this yet" }]
        }));
        capture.forwarded_request_body = Some(to_json_bytes(&json!({
            "model": "qwen3.6-27b",
            "metadata": { "agentic_os": {} }
        })));

        let records = records_from_capture(&capture).unwrap();
        assert_eq!(
            records[0].requested_model.as_deref(),
            Some("claude-opus-4-8")
        );
        assert_eq!(records[0].routed_model.as_deref(), Some("qwen3.6-27b"));
    }

    #[test]
    fn missing_forwarded_body_does_not_prevent_labeling() {
        let capture = capture_with_body(json!({
            "model": "claude-opus-4-8",
            "messages": [{ "role": "user", "content": "Do not implement this yet" }]
        }));

        let records = records_from_capture(&capture).unwrap();
        assert_eq!(records.len(), 1);
        let record = &records[0];
        assert_eq!(
            record.intervention_type,
            InterventionType::ImplementationBlock
        );
        assert_eq!(record.exchange_id, capture.exchange_id);
        assert_eq!(record.requested_model.as_deref(), Some("claude-opus-4-8"));
        assert_eq!(record.routed_model, None);
        assert_eq!(record.trajectory_id, None);
        assert_eq!(record.request_event_id, None);
        assert_eq!(record.attempt_id, None);
        assert_eq!(record.baseline_arm, None);
        assert_eq!(record.selected_route, None);
        assert_eq!(record.routing_policy_version, None);
        record.validate().unwrap();
    }

    #[test]
    fn capture_attempt_id_precedes_forwarded_metadata_attempt_id() {
        let capture_attempt_id = Uuid::new_v4();
        let metadata_attempt_id = Uuid::new_v4();
        let mut capture = capture_with_body(json!({
            "messages": [{ "role": "user", "content": "Run tests before committing" }]
        }));
        capture.attempt_id = Some(capture_attempt_id);
        capture.forwarded_request_body = Some(to_json_bytes(&json!({
            "metadata": {
                "agentic_os": {
                    "attempt_id": metadata_attempt_id
                }
            }
        })));

        let records = records_from_capture(&capture).unwrap();
        assert_eq!(records[0].attempt_id, Some(capture_attempt_id));
    }

    #[test]
    fn records_validate_and_evidence_is_redacted() {
        let capture = capture_with_body(json!({
            "messages": [{
                "role": "user",
                "content": "Authorization: Bearer secret-token do not implement this"
            }]
        }));

        let records = records_from_capture(&capture).unwrap();
        assert_eq!(records.len(), 1);
        assert!(!records[0].evidence_excerpt.contains("secret-token"));
        assert!(records[0].evidence_excerpt.contains("<redacted>"));
        records[0].validate().unwrap();
    }

    #[test]
    fn copies_detector_behavior_fields_into_records() {
        let capture = capture_with_body(json!({
            "model": "claude-opus-4-8",
            "messages": [{
                "role": "user",
                "content": "No I don't want you to do it - I am explicitly asking you to develop me a prompt and report it here."
            }]
        }));

        let records = records_from_capture(&capture).unwrap();
        assert_eq!(records.len(), 1);
        let record = &records[0];
        assert_eq!(record.intervention_type, InterventionType::StopAndRedirect);
        assert_eq!(record.blocked_behavior.as_deref(), Some("implementation"));
        assert_eq!(
            record.replacement_behavior.as_deref(),
            Some("develop prompt and report it")
        );
        record.validate().unwrap();
    }
}
