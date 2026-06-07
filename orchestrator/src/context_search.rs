use crate::db;
use crate::state::{AppState, SearchHit};

pub(crate) struct HybridSearchResult {
    pub(crate) hits: Vec<SearchHit>,
    pub(crate) semantic_hits: usize,
    pub(crate) fts_hits: usize,
    pub(crate) deduped_hits: usize,
}

async fn semantic_search(state: &AppState, query: &str, limit: usize) -> Vec<SearchHit> {
    crate::qdrant::search(&state.embedder, &state.qdrant_url, query, limit)
        .await
        .unwrap_or_default()
        .into_iter()
        .filter_map(|item| {
            let payload = item.get("payload")?;
            let created_at = payload
                .get("created_at")
                .and_then(|v| v.as_str())
                .and_then(|s| chrono::DateTime::parse_from_rfc3339(s).ok())
                .map(|dt| dt.with_timezone(&chrono::Utc));
            Some(SearchHit {
                event_id: payload.get("event_id")?.as_str()?.to_string(),
                event_type: payload.get("event_type")?.as_str()?.to_string(),
                summary: payload.get("summary")?.as_str()?.to_string(),
                created_at,
            })
        })
        .collect()
}

pub(crate) async fn hybrid_search(
    state: &AppState,
    repo: &str,
    task: &str,
    semantic_limit: usize,
) -> HybridSearchResult {
    let query = format!("{repo} {task}");
    let fts_limit = semantic_limit as i64;

    let (semantic, fts_result) = tokio::join!(
        semantic_search(state, &query, semantic_limit),
        db::search_events_fts(&state.pool, repo, task, fts_limit),
    );

    let semantic = db::hydrate_active_search_hits(&state.pool, repo, semantic)
        .await
        .unwrap_or_else(|e| {
            tracing::warn!("semantic hit hydration failed, falling back to FTS-only: {e}");
            vec![]
        });
    let fts = fts_result.unwrap_or_else(|e| {
        tracing::warn!("FTS search failed, falling back to semantic-only: {e}");
        vec![]
    });
    let hits = crate::hybrid::rrf_merge_decay(
        &semantic,
        &fts,
        60.0,
        semantic_limit,
        state.context_decay_rate,
    );
    let deduped_hits = hits.len();

    HybridSearchResult {
        hits,
        semantic_hits: semantic.len(),
        fts_hits: fts.len(),
        deduped_hits,
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn hybrid_search_uses_tokio_join() {
        let src = include_str!("context_search.rs");
        let hs_start = src
            .find("pub(crate) async fn hybrid_search")
            .expect("hybrid_search not found");
        let body = &src[hs_start..hs_start + 500];
        assert!(
            body.contains("tokio::join!"),
            "hybrid_search should parallelize semantic_search and FTS via tokio::join!"
        );
        assert!(
            body.contains("semantic_search") && body.contains("db::search_events_fts"),
            "hybrid_search should parallelize semantic_search and FTS"
        );
    }
}
