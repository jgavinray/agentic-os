use serde_json::Value;

pub(crate) fn event_text(event: &crate::db::AgentEvent) -> String {
    match event.evidence.as_deref().filter(|value| !value.is_empty()) {
        Some(evidence) => format!("{}\n{}", event.summary, evidence),
        None => event.summary.clone(),
    }
}

pub(crate) fn has_request_text(event: &crate::db::AgentEvent) -> bool {
    !event_text(event).trim().is_empty()
}

pub(crate) fn metadata_key_text(value: &Value) -> String {
    fn collect(value: &Value, keys: &mut Vec<String>) {
        match value {
            Value::Object(map) => {
                for (key, nested) in map {
                    keys.push(key.clone());
                    collect(nested, keys);
                }
            }
            Value::Array(items) => {
                for item in items {
                    collect(item, keys);
                }
            }
            _ => {}
        }
    }

    let mut keys = Vec::new();
    collect(value, &mut keys);
    keys.join(" ")
}
