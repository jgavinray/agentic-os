use deadpool_postgres::Pool;

/// Full-text search on agent_events.summary and evidence for a given repo.
/// Returns results ordered by ts_rank DESC.
pub async fn search_events_fts(
    pool: &Pool,
    repo: &str,
    query: &str,
    limit: i64,
) -> Result<Vec<crate::state::SearchHit>, anyhow::Error> {
    let started = std::time::Instant::now();
    let result = async {
        let conn = pool.get().await?;
        let rows = conn
            .query(
                "WITH docs AS (
                     SELECT id, event_type, summary, created_at,
                            to_tsvector('english', coalesce(summary, '') || ' ' || coalesce(evidence, '')) AS tsv
                     FROM agent_events
                     WHERE repo = $1
                       AND summarized = false
                       AND metadata->'harness_feedback'->>'quarantined' IS DISTINCT FROM 'true'
                 )
                 SELECT id, event_type, summary, created_at
                 FROM docs
                 WHERE tsv @@ plainto_tsquery('english', $2)
                 ORDER BY ts_rank(tsv, plainto_tsquery('english', $2)) DESC
                 LIMIT $3",
                &[&repo, &query, &limit],
            )
            .await?;

        Ok(rows
            .into_iter()
            .map(|row| crate::state::SearchHit {
                event_id: row.get("id"),
                event_type: row.get("event_type"),
                summary: row.get("summary"),
                created_at: Some(row.get("created_at")),
            })
            .collect())
    }
    .await;
    crate::telemetry::record_db_query("search_events_fts", started.elapsed(), result.is_ok());
    result
}

pub async fn hydrate_active_search_hits(
    pool: &Pool,
    repo: &str,
    hits: Vec<crate::state::SearchHit>,
) -> Result<Vec<crate::state::SearchHit>, anyhow::Error> {
    if hits.is_empty() {
        return Ok(hits);
    }

    let started = std::time::Instant::now();
    let result = async {
        let ids: Vec<String> = hits.iter().map(|h| h.event_id.clone()).collect();
        let conn = pool.get().await?;
        let rows = conn
            .query(
                "SELECT id, event_type, summary, created_at
                 FROM agent_events
                 WHERE repo = $1
                   AND summarized = false
                   AND metadata->'harness_feedback'->>'quarantined' IS DISTINCT FROM 'true'
                   AND id = ANY($2)",
                &[&repo, &ids],
            )
            .await?;

        let by_id: std::collections::HashMap<String, crate::state::SearchHit> = rows
            .into_iter()
            .map(|row| {
                let id: String = row.get("id");
                (
                    id.clone(),
                    crate::state::SearchHit {
                        event_id: id,
                        event_type: row.get("event_type"),
                        summary: row.get("summary"),
                        created_at: Some(row.get("created_at")),
                    },
                )
            })
            .collect();

        Ok(hits
            .into_iter()
            .filter_map(|hit| by_id.get(&hit.event_id).cloned())
            .collect())
    }
    .await;
    crate::telemetry::record_db_query(
        "hydrate_active_search_hits",
        started.elapsed(),
        result.is_ok(),
    );
    result
}
