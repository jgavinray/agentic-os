use crate::feature_constraint_budget::constraint_priority;
use crate::feature_constraint_failures::{recovery_after_detection, FailureKey};
use crate::feature_extraction_types::SuppressedConstraint;
use chrono::{DateTime, Utc};
use std::collections::{BTreeMap, BTreeSet};

pub(crate) fn active_detection(
    key: &FailureKey,
    latest_detection: &BTreeMap<FailureKey, DateTime<Utc>>,
    latest_recovery: &BTreeMap<FailureKey, DateTime<Utc>>,
    suppressed: &mut Vec<SuppressedConstraint>,
    constraint_type: &str,
) -> bool {
    let Some(detected_at) = latest_detection.get(key) else {
        return false;
    };
    if recovery_after_detection(key, *detected_at, latest_recovery) {
        suppressed.push(SuppressedConstraint {
            constraint_type: constraint_type.to_string(),
            reason: "recovery_detected".to_string(),
        });
        return false;
    }
    true
}

pub(crate) fn suppress_duplicates(suppressed: &mut Vec<SuppressedConstraint>) {
    let mut seen = BTreeSet::new();
    suppressed.retain(|item| seen.insert((item.constraint_type.clone(), item.reason.clone())));
    suppressed.sort_by_key(|item| {
        (
            constraint_priority(&item.constraint_type),
            item.reason.clone(),
        )
    });
}
