use crate::db::AgentEvent;
use serde_json::Value;

pub fn event_payload(event: &AgentEvent) -> &Value {
    event.metadata.get("payload").unwrap_or(&event.metadata)
}

pub fn event_success(event: &AgentEvent) -> bool {
    event
        .metadata
        .get("success")
        .and_then(Value::as_bool)
        .or_else(|| event_payload(event).get("success").and_then(Value::as_bool))
        .or_else(|| event_payload(event).get("pass").and_then(Value::as_bool))
        .unwrap_or(false)
}

pub fn payload_str<'a>(payload: &'a Value, key: &str) -> Option<&'a str> {
    payload.get(key).and_then(Value::as_str)
}

pub fn payload_bool(payload: &Value, key: &str) -> Option<bool> {
    payload.get(key).and_then(Value::as_bool)
}

pub fn metadata_str<'a>(metadata: &'a Value, key: &str) -> Option<&'a str> {
    metadata.get(key).and_then(Value::as_str).or_else(|| {
        metadata
            .get("payload")
            .and_then(|p| p.get(key))
            .and_then(Value::as_str)
    })
}

pub fn metadata_i64(metadata: &Value, key: &str) -> Option<i64> {
    metadata.get(key).and_then(Value::as_i64).or_else(|| {
        metadata
            .get("payload")
            .and_then(|p| p.get(key))
            .and_then(Value::as_i64)
    })
}
