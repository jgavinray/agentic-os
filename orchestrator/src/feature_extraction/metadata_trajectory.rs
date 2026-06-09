use serde_json::Value;

use crate::feature_metadata_paths::{i64_path, string_path};

pub(crate) fn trajectory_abandoned_before_model(metadata: &Value) -> bool {
    trajectory_status(metadata) == Some("abandoned")
        && event_model_calls_from_metadata(metadata) == Some(0)
}

pub(crate) fn trajectory_single_model_abandoned_no_tools(metadata: &Value) -> bool {
    trajectory_status(metadata) == Some("abandoned")
        && event_model_calls_from_metadata(metadata) == Some(1)
        && i64_path(metadata, &["total_tool_calls"])
            .or_else(|| i64_path(metadata, &["payload", "total_tool_calls"]))
            .unwrap_or(0)
            == 0
        && i64_path(metadata, &["total_validations"])
            .or_else(|| i64_path(metadata, &["payload", "total_validations"]))
            .unwrap_or(0)
            == 0
}

fn trajectory_status(metadata: &Value) -> Option<&str> {
    string_path(metadata, &["final_status"])
        .or_else(|| string_path(metadata, &["payload", "final_status"]))
}

fn event_model_calls_from_metadata(metadata: &Value) -> Option<i64> {
    i64_path(metadata, &["total_model_calls"])
        .or_else(|| i64_path(metadata, &["payload", "total_model_calls"]))
}
