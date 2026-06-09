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
