use crate::state::SearchHit;

/// Reciprocal Rank Fusion with k=60 (industry default).
/// Input slices are ordered best-first (index 0 = rank 1).
/// Returns deduplicated SearchHits ordered by combined RRF score, highest first.
#[cfg(test)]
pub fn rrf_merge(
    semantic_hits: &[SearchHit],
    fts_hits: &[SearchHit],
    k: f64,
    limit: usize,
) -> Vec<SearchHit> {
    use std::collections::HashMap;

    // (accumulated_score, first-seen hit)
    let mut scores: HashMap<String, (f64, SearchHit)> = HashMap::new();

    for (rank, hit) in semantic_hits.iter().enumerate() {
        let contribution = 1.0 / (k + (rank + 1) as f64);
        scores
            .entry(hit.event_id.clone())
            .and_modify(|(s, _)| *s += contribution)
            .or_insert((contribution, hit.clone()));
    }

    for (rank, hit) in fts_hits.iter().enumerate() {
        let contribution = 1.0 / (k + (rank + 1) as f64);
        scores
            .entry(hit.event_id.clone())
            .and_modify(|(s, _)| *s += contribution)
            .or_insert((contribution, hit.clone()));
    }

    let mut merged: Vec<(f64, SearchHit)> = scores.into_values().collect();
    merged.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    merged.into_iter().take(limit).map(|(_, hit)| hit).collect()
}

/// Compute time decay: e^(-lambda * age_minutes).
pub fn compute_decay(age_minutes: f64, lambda: f64) -> f64 {
    (-lambda * age_minutes).exp().clamp(0.001, 1.0)
}

/// Reciprocal Rank Fusion with time decay.
/// Stale events get reduced rank proportional to their age.
pub fn rrf_merge_decay(
    semantic_hits: &[SearchHit],
    fts_hits: &[SearchHit],
    k: f64,
    limit: usize,
    lambda: f64,
) -> Vec<SearchHit> {
    use std::collections::HashMap;

    let mut scores: HashMap<String, (f64, SearchHit)> = HashMap::new();

    for hits in [semantic_hits, fts_hits] {
        for (rank, hit) in hits.iter().enumerate() {
            let rrf = 1.0 / (k + (rank + 1) as f64);
            let minutes = hit
                .created_at
                .map(|ts| (chrono::Utc::now() - ts).num_minutes().max(0) as f64)
                .unwrap_or(0.0);
            let contribution = rrf * compute_decay(minutes, lambda);
            scores
                .entry(hit.event_id.clone())
                .and_modify(|(score, existing)| {
                    *score += contribution;
                    if existing.created_at.is_none() && hit.created_at.is_some() {
                        *existing = hit.clone();
                    }
                })
                .or_insert((contribution, hit.clone()));
        }
    }

    let mut merged: Vec<(f64, SearchHit)> = scores.into_values().collect();
    merged.sort_by(|a, b| {
        b.0.partial_cmp(&a.0)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| b.1.created_at.cmp(&a.1.created_at))
    });
    merged.into_iter().take(limit).map(|(_, hit)| hit).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

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

    // ── Time decay tests ─────────────────────────────────────────────

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
}
