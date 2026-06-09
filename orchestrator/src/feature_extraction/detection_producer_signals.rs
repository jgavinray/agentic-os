use crate::feature_extraction_types::DetectionTag;
use serde_json::Value;

pub(crate) fn explicit_producer_signal_tags(
    metadata: &Value,
    source_override: Option<&str>,
) -> Vec<DetectionTag> {
    let mut tags = Vec::new();
    let Some(signals) = metadata.get("producer_signals").and_then(Value::as_object) else {
        return tags;
    };
    for (source, value) in signals {
        let source = source_override.unwrap_or(source.as_str());
        match value {
            Value::Array(values) => {
                for value in values {
                    if let Some(tag) = tag_from_signal_value(source, value) {
                        tags.push(tag);
                    }
                }
            }
            Value::Object(_) => {
                if let Some(tag) = tag_from_signal_value(source, value) {
                    tags.push(tag);
                }
            }
            Value::String(tag_type) => tags.push(DetectionTag::new(tag_type, source)),
            _ => {}
        }
    }
    tags
}

fn tag_from_signal_value(source: &str, value: &Value) -> Option<DetectionTag> {
    let obj = value.as_object()?;
    let tag_type = obj.get("type").and_then(Value::as_str)?;
    if tag_type == "tool_loop" {
        let tool = obj.get("tool").and_then(Value::as_str).unwrap_or("unknown");
        Some(DetectionTag::tool_loop(source, tool))
    } else {
        Some(DetectionTag::new(tag_type, source))
    }
}
