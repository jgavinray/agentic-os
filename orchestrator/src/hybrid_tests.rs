use crate::hybrid::{compute_decay, rrf_merge, rrf_merge_decay};
use crate::state::SearchHit;

fn sh(id: &str, event_type: &str, summary: &str) -> SearchHit {
    SearchHit {
        event_id: id.to_string(),
        event_type: event_type.to_string(),
        summary: summary.to_string(),
        created_at: Some(chrono::Utc::now()),
    }
}

#[test]
fn rrf_merge_empty_lists_returns_empty() {
    assert!(rrf_merge(&[], &[], 60.0, 10).is_empty());
}

#[test]
fn rrf_merge_single_list_preserves_order() {
    let sem = vec![sh("a", "edit", "first"), sh("b", "edit", "second")];
    let result = rrf_merge(&sem, &[], 60.0, 10);
    assert_eq!(result[0].event_id, "a");
    assert_eq!(result[1].event_id, "b");
}

#[test]
fn rrf_merge_item_in_both_lists_scores_higher_than_item_in_one() {
    let sem = vec![
        sh("solo", "edit", "only semantic"),
        sh("overlap", "edit", "both"),
    ];
    let fts = vec![sh("overlap", "edit", "both")];
    let result = rrf_merge(&sem, &fts, 60.0, 10);
    assert_eq!(result[0].event_id, "overlap");
}

#[test]
fn rrf_merge_deduplicates_by_event_id() {
    let sem = vec![sh("a", "edit", "x"), sh("a", "edit", "x")];
    let fts = vec![sh("a", "edit", "x")];
    let result = rrf_merge(&sem, &fts, 60.0, 10);
    assert_eq!(result.len(), 1);
}

#[test]
fn rrf_merge_respects_limit() {
    let sem: Vec<SearchHit> = (0..10).map(|i| sh(&i.to_string(), "edit", "x")).collect();
    let result = rrf_merge(&sem, &[], 60.0, 3);
    assert_eq!(result.len(), 3);
}

#[test]
fn decay_no_timestamp_returns_one() {
    let mut hit = sh("a", "edit", "fresh");
    hit.created_at = None;
    let result = rrf_merge_decay(
        &[hit],
        &[],
        60.0,
        10,
        crate::state::DEFAULT_CONTEXT_DECAY_RATE,
    );
    assert_eq!(result.len(), 1);
    assert_eq!(result[0].event_id, "a");
}

#[test]
fn decay_favors_recency() {
    let fresh_ts = chrono::Utc::now();
    let stale_ts = chrono::Utc::now() - chrono::Duration::hours(1);
    let fresh = sh("fresh", "edit", "now");
    let stale = sh("stale", "edit", "old");
    let mut fresh = fresh;
    let mut stale = stale;
    fresh.created_at = Some(fresh_ts);
    stale.created_at = Some(stale_ts);

    let r = rrf_merge_decay(
        &[],
        &[stale, fresh],
        60.0,
        10,
        crate::state::DEFAULT_CONTEXT_DECAY_RATE,
    );

    assert_eq!(r[0].event_id, "fresh");
    assert_eq!(r[1].event_id, "stale");
}

#[test]
fn decay_still_surfaces_stale_if_no_alternative() {
    let mut stale = sh("stale", "edit", "old");
    let stale_ts = chrono::Utc::now() - chrono::Duration::hours(1);
    stale.created_at = Some(stale_ts);
    let r = rrf_merge_decay(
        &[],
        &[stale],
        60.0,
        10,
        crate::state::DEFAULT_CONTEXT_DECAY_RATE,
    );
    assert_eq!(r.len(), 1);
    assert_eq!(r[0].event_id, "stale");
}

#[test]
fn decay_rate_matches_plan_acceptance_thresholds() {
    let lambda = crate::state::DEFAULT_CONTEXT_DECAY_RATE;
    assert!(compute_decay(60.0, lambda) <= 0.70);
    assert!(compute_decay(1440.0, lambda) <= 0.30);
}
