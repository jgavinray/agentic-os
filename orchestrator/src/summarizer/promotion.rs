use serde_json::Value;

pub(crate) fn should_promote_to_level(
    target_level: i32,
    event_type: &str,
    summary: &str,
    metadata: &Value,
    source_count: usize,
) -> bool {
    match target_level {
        1 => true,
        2 => {
            has_source_ids(metadata)
                && (source_count >= 2
                    || has_l2_signal(event_type, summary, metadata)
                    || has_failed_or_remediated_content(summary))
        }
        3 => {
            has_source_ids(metadata)
                && (metadata_flag(metadata, "stable")
                    || metadata_flag(metadata, "manually_confirmed")
                    || metadata_flag(metadata, "architecture_truth")
                    || (source_count >= 2 && is_architecture_related(summary)))
        }
        _ => false,
    }
}

pub(crate) fn has_source_ids(metadata: &Value) -> bool {
    metadata
        .get("source_event_ids")
        .or_else(|| metadata.get("summarized_event_ids"))
        .and_then(|v| v.as_array())
        .map(|ids| !ids.is_empty())
        .unwrap_or(false)
}

fn has_l2_signal(event_type: &str, summary: &str, metadata: &Value) -> bool {
    metadata_flag(metadata, "important")
        || metadata_flag(metadata, "referenced")
        || metadata.get("last_referenced_at").is_some()
        || metadata
            .get("referenced_count")
            .and_then(|v| v.as_u64())
            .unwrap_or(0)
            > 0
        || matches!(event_type, "failed_attempt" | "remediation")
        || is_architecture_related(summary)
}

fn metadata_flag(metadata: &Value, key: &str) -> bool {
    metadata.get(key).and_then(|v| v.as_bool()).unwrap_or(false)
}

fn has_failed_or_remediated_content(summary: &str) -> bool {
    let lower = summary.to_lowercase();
    lower.contains("failed")
        || lower.contains("failure")
        || lower.contains("remediation")
        || lower.contains("do not retry")
}

fn is_architecture_related(summary: &str) -> bool {
    let lower = summary.to_lowercase();
    crate::state::TaskCategory::ARCHITECTURE_KEYWORDS
        .iter()
        .any(|kw| lower.contains(kw))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn l1_to_l2_requires_promotion_signal() {
        let no_sources = json!({});
        assert!(!should_promote_to_level(
            2,
            "summary",
            "ordinary session summary",
            &no_sources,
            10
        ));

        let with_sources = json!({"summarized_event_ids": ["e1"]});
        assert!(!should_promote_to_level(
            2,
            "summary",
            "ordinary session summary",
            &with_sources,
            1
        ));

        let important = json!({"summarized_event_ids": ["e1"], "important": true});
        assert!(should_promote_to_level(
            2,
            "summary",
            "repo convention should be reused",
            &important,
            1
        ));

        let repeated = json!({"summarized_event_ids": ["e1", "e2"]});
        assert!(should_promote_to_level(
            2,
            "summary",
            "same retry mistake happened again",
            &repeated,
            2
        ));
    }

    #[test]
    fn l2_to_l3_requires_stability_or_confirmation() {
        let with_sources = json!({"summarized_event_ids": ["s1"]});
        assert!(!should_promote_to_level(
            3,
            "summary",
            "repo-level cache convention",
            &with_sources,
            5
        ));

        let stable = json!({"summarized_event_ids": ["s1"], "stable": true});
        assert!(should_promote_to_level(
            3,
            "summary",
            "repo-level cache convention",
            &stable,
            1
        ));

        let confirmed = json!({"summarized_event_ids": ["s1"], "manually_confirmed": true});
        assert!(should_promote_to_level(
            3,
            "summary",
            "durable project memory",
            &confirmed,
            1
        ));

        let no_sources = json!({"stable": true});
        assert!(!should_promote_to_level(
            3,
            "summary",
            "stable but unverifiable",
            &no_sources,
            5
        ));
    }
}
