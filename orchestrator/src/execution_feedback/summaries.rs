use serde_json::Value;

use crate::execution_feedback::{
    EVENT_TYPE_COMPILE_RESULT, EVENT_TYPE_LINT_RESULT, EVENT_TYPE_PATCH_RESULT,
    EVENT_TYPE_REMEDIATION, EVENT_TYPE_TEST_RESULT, EVENT_TYPE_TOOL_RESULT,
    EVENT_TYPE_VALIDATION_RESULT,
};

pub(crate) fn event_summary(event_type: &str, success: bool, payload: &Value) -> String {
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
