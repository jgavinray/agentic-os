use chrono::{DateTime, Utc};
use deadpool_postgres::Pool;

use crate::request_classification_types::{
    LabelCount, LowMarginIntent, ReportOptions, RequestClassificationReport, SessionRouteCount,
};

/// Intent decisions with a margin at or below this are "contested".
const LOW_MARGIN_THRESHOLD: i64 = 10;

pub async fn request_classification_report(
    pool: &Pool,
    opts: &ReportOptions,
) -> Result<RequestClassificationReport, anyhow::Error> {
    let conn = pool.get().await?;
    let by_route = count_grouped(
        &conn,
        "recommended_route",
        opts.repo.as_deref(),
        opts.since,
        20,
    )
    .await?;
    let unknown_label_counts =
        count_unknown_labels(&conn, opts.repo.as_deref(), opts.since).await?;
    let repeated_guardrail_sessions =
        count_repeated_guardrail_sessions(&conn, opts.repo.as_deref(), opts.since).await?;
    let top_risk_flags = count_risk_flags(&conn, opts.repo.as_deref(), opts.since, 20).await?;
    let low_margin_intents =
        list_low_margin_intents(&conn, opts.repo.as_deref(), opts.since, 20).await?;

    Ok(RequestClassificationReport {
        by_route,
        top_risk_flags,
        unknown_label_counts,
        repeated_guardrail_sessions,
        low_margin_intents,
    })
}

/// Most recent contested intent decisions: the winning weight barely beat the
/// runner-up. These rows are the highest-value candidates to hand-label into
/// the golden corpus (tests/corpus.rs) before changing classifier rules.
async fn list_low_margin_intents(
    conn: &deadpool_postgres::Object,
    repo: Option<&str>,
    since: Option<DateTime<Utc>>,
    limit: i64,
) -> Result<Vec<LowMarginIntent>, anyhow::Error> {
    let rows = conn
        .query(
            "SELECT event_id::TEXT AS event_id,
                    intent,
                    features->>'intent_runner_up' AS runner_up,
                    (features->>'intent_margin')::BIGINT AS margin
             FROM agent_request_classifications
             WHERE features ? 'intent_margin'
               AND (features->>'intent_margin')::BIGINT <= $3
               AND ($1::TEXT IS NULL OR repo = $1)
               AND ($2::TIMESTAMPTZ IS NULL OR event_created_at >= $2)
             ORDER BY event_created_at DESC
             LIMIT $4",
            &[&repo, &since, &LOW_MARGIN_THRESHOLD, &limit],
        )
        .await?;
    Ok(rows
        .into_iter()
        .map(|row| LowMarginIntent {
            event_id: row.get("event_id"),
            intent: row.get("intent"),
            runner_up: row.get("runner_up"),
            margin: row.get("margin"),
        })
        .collect())
}

async fn count_grouped(
    conn: &deadpool_postgres::Object,
    column: &str,
    repo: Option<&str>,
    since: Option<DateTime<Utc>>,
    limit: i64,
) -> Result<Vec<LabelCount>, anyhow::Error> {
    let column = match column {
        "recommended_route" => "recommended_route",
        "intent" => "intent",
        "domain" => "domain",
        "complexity" => "complexity",
        _ => anyhow::bail!("unsupported report column"),
    };
    let sql = format!(
        "SELECT {column} AS label, count(*)::BIGINT AS count
         FROM agent_request_classifications
         WHERE ($1::TEXT IS NULL OR repo = $1)
           AND ($2::TIMESTAMPTZ IS NULL OR event_created_at >= $2)
         GROUP BY {column}
         ORDER BY count DESC, {column} ASC
         LIMIT $3"
    );
    let rows = conn.query(&sql, &[&repo, &since, &limit]).await?;
    Ok(rows
        .into_iter()
        .map(|row| LabelCount {
            label: row.get("label"),
            count: row.get("count"),
        })
        .collect())
}

async fn count_risk_flags(
    conn: &deadpool_postgres::Object,
    repo: Option<&str>,
    since: Option<DateTime<Utc>>,
    limit: i64,
) -> Result<Vec<LabelCount>, anyhow::Error> {
    let rows = conn
        .query(
            "SELECT risk_label AS label, count(*)::BIGINT AS count
             FROM agent_request_classifications,
                  unnest(risk) AS risk_label
             WHERE ($1::TEXT IS NULL OR repo = $1)
               AND ($2::TIMESTAMPTZ IS NULL OR event_created_at >= $2)
             GROUP BY risk_label
             ORDER BY count DESC, risk_label ASC
             LIMIT $3",
            &[&repo, &since, &limit],
        )
        .await?;
    Ok(rows
        .into_iter()
        .map(|row| LabelCount {
            label: row.get("label"),
            count: row.get("count"),
        })
        .collect())
}

async fn count_unknown_labels(
    conn: &deadpool_postgres::Object,
    repo: Option<&str>,
    since: Option<DateTime<Utc>>,
) -> Result<Vec<LabelCount>, anyhow::Error> {
    let rows = conn
        .query(
            "SELECT field, sum(count)::BIGINT AS count
             FROM (
                 SELECT 'intent' AS field, count(*)::BIGINT AS count
                 FROM agent_request_classifications
                 WHERE intent = 'unknown'
                   AND ($1::TEXT IS NULL OR repo = $1)
                   AND ($2::TIMESTAMPTZ IS NULL OR event_created_at >= $2)
                 UNION ALL
                 SELECT 'domain' AS field, count(*)::BIGINT AS count
                 FROM agent_request_classifications
                 WHERE domain = 'unknown'
                   AND ($1::TEXT IS NULL OR repo = $1)
                   AND ($2::TIMESTAMPTZ IS NULL OR event_created_at >= $2)
                 UNION ALL
                 SELECT 'artifact_type' AS field, count(*)::BIGINT AS count
                 FROM agent_request_classifications
                 WHERE artifact_type = 'unknown'
                   AND ($1::TEXT IS NULL OR repo = $1)
                   AND ($2::TIMESTAMPTZ IS NULL OR event_created_at >= $2)
                 UNION ALL
                 SELECT 'complexity' AS field, count(*)::BIGINT AS count
                 FROM agent_request_classifications
                 WHERE complexity = 'unknown'
                   AND ($1::TEXT IS NULL OR repo = $1)
                   AND ($2::TIMESTAMPTZ IS NULL OR event_created_at >= $2)
                 UNION ALL
                 SELECT 'recommended_route' AS field, count(*)::BIGINT AS count
                 FROM agent_request_classifications
                 WHERE recommended_route = 'unknown'
                   AND ($1::TEXT IS NULL OR repo = $1)
                   AND ($2::TIMESTAMPTZ IS NULL OR event_created_at >= $2)
                 UNION ALL
                 SELECT 'response_contract' AS field, count(*)::BIGINT AS count
                 FROM agent_request_classifications
                 WHERE response_contract = 'unknown'
                   AND ($1::TEXT IS NULL OR repo = $1)
                   AND ($2::TIMESTAMPTZ IS NULL OR event_created_at >= $2)
             ) unknowns
             GROUP BY field
             ORDER BY field ASC",
            &[&repo, &since],
        )
        .await?;
    Ok(rows
        .into_iter()
        .map(|row| LabelCount {
            label: row.get("field"),
            count: row.get("count"),
        })
        .collect())
}

async fn count_repeated_guardrail_sessions(
    conn: &deadpool_postgres::Object,
    repo: Option<&str>,
    since: Option<DateTime<Utc>>,
) -> Result<Vec<SessionRouteCount>, anyhow::Error> {
    let rows = conn
        .query(
            "SELECT session_id, count(*)::BIGINT AS count
             FROM agent_request_classifications
             WHERE recommended_route = 'refuse_or_guardrail'
               AND ($1::TEXT IS NULL OR repo = $1)
               AND ($2::TIMESTAMPTZ IS NULL OR event_created_at >= $2)
             GROUP BY session_id
             HAVING count(*) > 1
             ORDER BY count DESC, session_id ASC
             LIMIT 20",
            &[&repo, &since],
        )
        .await?;
    Ok(rows
        .into_iter()
        .map(|row| SessionRouteCount {
            session_id: row.get("session_id"),
            count: row.get("count"),
        })
        .collect())
}
