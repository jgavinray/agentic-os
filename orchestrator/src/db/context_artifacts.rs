use crate::context_artifacts::ContextArtifact;
use deadpool_postgres::Pool;

pub use crate::db_context_artifact_candidates::{
    get_recent_instruction_candidates, get_recent_repo_decision_candidates,
    get_recent_session_events,
};

pub async fn upsert_context_artifact(
    pool: &Pool,
    artifact: &ContextArtifact,
) -> Result<(), anyhow::Error> {
    let started = std::time::Instant::now();
    let result = async {
        let conn = pool.get().await?;
        conn.execute(
            "INSERT INTO context_artifacts
             (id, repo, scope, artifact_type, status, content_raw, content_compact,
              content_rendered, content_hash, invalidation_key, source_event_ids,
              source_file_paths, token_estimate)
             VALUES
             ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13)
             ON CONFLICT (repo, scope, artifact_type, invalidation_key) DO UPDATE
             SET status = EXCLUDED.status,
                 content_raw = EXCLUDED.content_raw,
                 content_compact = EXCLUDED.content_compact,
                 content_rendered = EXCLUDED.content_rendered,
                 content_hash = EXCLUDED.content_hash,
                 source_event_ids = EXCLUDED.source_event_ids,
                 source_file_paths = EXCLUDED.source_file_paths,
                 token_estimate = EXCLUDED.token_estimate,
                 updated_at = now()",
            &[
                &artifact.id,
                &artifact.repo,
                &artifact.scope,
                &artifact.artifact_type,
                &artifact.status,
                &artifact.content_raw,
                &artifact.content_compact,
                &artifact.content_rendered,
                &artifact.content_hash,
                &artifact.invalidation_key,
                &artifact.source_event_ids,
                &artifact.source_file_paths,
                &artifact.token_estimate,
            ],
        )
        .await?;
        Ok(())
    }
    .await;
    crate::telemetry::record_db_query("upsert_context_artifact", started.elapsed(), result.is_ok());
    result
}

pub async fn get_active_context_artifacts(
    pool: &Pool,
    repo: &str,
    limit: i64,
) -> Result<Vec<ContextArtifact>, anyhow::Error> {
    let started = std::time::Instant::now();
    let result = async {
        let conn = pool.get().await?;
        let rows = conn
            .query(
                "SELECT id, repo, scope, artifact_type, status, content_raw, content_compact,
                        content_rendered, content_hash, invalidation_key, source_event_ids,
                        source_file_paths, token_estimate
                 FROM context_artifacts
                 WHERE repo = $1
                   AND status = 'active'
                 ORDER BY
                   CASE artifact_type
                     WHEN 'service_topology' THEN 0
                     WHEN 'architecture_spec' THEN 1
                     WHEN 'active_instruction' THEN 1
                     WHEN 'repo_map' THEN 2
                     WHEN 'durable_project_memory' THEN 3
                     WHEN 'repo_decisions' THEN 4
                     WHEN 'failure_history' THEN 5
                     ELSE 99
                   END,
                   updated_at DESC
                 LIMIT $2",
                &[&repo, &limit],
            )
            .await?;

        Ok(rows.into_iter().map(row_to_context_artifact).collect())
    }
    .await;
    crate::telemetry::record_db_query(
        "get_active_context_artifacts",
        started.elapsed(),
        result.is_ok(),
    );
    result
}

pub async fn get_context_artifacts_for_repo(
    pool: &Pool,
    repo: &str,
    limit: i64,
) -> Result<Vec<ContextArtifact>, anyhow::Error> {
    let started = std::time::Instant::now();
    let result = async {
        let conn = pool.get().await?;
        let rows = conn
            .query(
                "SELECT id, repo, scope, artifact_type, status, content_raw, content_compact,
                        content_rendered, content_hash, invalidation_key, source_event_ids,
                        source_file_paths, token_estimate
                 FROM context_artifacts
                 WHERE repo = $1
                 ORDER BY updated_at DESC
                 LIMIT $2",
                &[&repo, &limit],
            )
            .await?;

        Ok(rows.into_iter().map(row_to_context_artifact).collect())
    }
    .await;
    crate::telemetry::record_db_query(
        "get_context_artifacts_for_repo",
        started.elapsed(),
        result.is_ok(),
    );
    result
}

fn row_to_context_artifact(row: tokio_postgres::Row) -> ContextArtifact {
    ContextArtifact {
        id: row.get("id"),
        repo: row.get("repo"),
        scope: row.get("scope"),
        artifact_type: row.get("artifact_type"),
        status: row.get("status"),
        content_raw: row.get("content_raw"),
        content_compact: row.get("content_compact"),
        content_rendered: row.get("content_rendered"),
        content_hash: row.get("content_hash"),
        invalidation_key: row.get("invalidation_key"),
        source_event_ids: row.get("source_event_ids"),
        source_file_paths: row.get("source_file_paths"),
        token_estimate: row.get("token_estimate"),
    }
}
