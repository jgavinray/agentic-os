use crate::state::SearchHit;

/// Reciprocal Rank Fusion with k=60 (industry default).
/// Input slices are ordered best-first (index 0 = rank 1).
/// Returns deduplicated SearchHits ordered by combined RRF score, highest first.
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
        // Score is accumulated; hit data comes from whichever list first introduced this event_id.
        // Both lists read from the same agent_events table, so summaries are identical in practice.
        scores
            .entry(hit.event_id.clone())
            .and_modify(|(s, _)| *s += contribution)
            .or_insert((contribution, hit.clone()));
    }

    let mut merged: Vec<(f64, SearchHit)> = scores.into_values().collect();
    merged.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
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
        // "solo" is rank-0 in semantic only: score = 1/61
        // "overlap" is rank-1 in semantic AND rank-0 in fts: score = 1/62 + 1/61 > 1/61
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
}
