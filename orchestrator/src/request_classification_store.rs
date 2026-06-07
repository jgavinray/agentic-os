use deadpool_postgres::Pool;

use crate::request_classification_types::{PersistOutcome, RequestClassification};

pub use crate::request_classification_backfill::run_backfill;

pub async fn classify_and_persist_event(
    pool: &Pool,
    event: &crate::db::AgentEvent,
) -> Result<Option<PersistOutcome>, anyhow::Error> {
    if !crate::request_classification::is_classifiable_request_event(event) {
        return Ok(None);
    }
    let classification = crate::request_classification::classify_request_event(event);
    let outcome = persist_classification(pool, &classification).await?;
    Ok(Some(outcome))
}

pub async fn persist_classification(
    pool: &Pool,
    classification: &RequestClassification,
) -> Result<PersistOutcome, anyhow::Error> {
    let result = async {
        let conn = pool.get().await?;
        let secondary_domains = classification
            .secondary_domains
            .iter()
            .map(|domain| domain.as_str().to_string())
            .collect::<Vec<_>>();
        let risk = classification
            .risk
            .iter()
            .map(|risk| risk.as_str().to_string())
            .collect::<Vec<_>>();
        let affected = conn
            .execute(
                "INSERT INTO agent_request_classifications (
                    event_id,
                    repo,
                    session_id,
                    trajectory_id,
                    event_created_at,
                    classified_at,
                    classification_schema_version,
                    routing_policy_version,
                    classifier_source,
                    intent,
                    domain,
                    secondary_domains,
                    artifact_type,
                    risk,
                    complexity,
                    recommended_route,
                    response_contract,
                    features
                )
                VALUES (
                    $1, $2, $3, $4, $5, $6, $7, $8, $9,
                    $10, $11, $12, $13, $14, $15, $16, $17, $18
                )
                ON CONFLICT (
                    event_id,
                    classification_schema_version,
                    routing_policy_version
                )
                DO NOTHING",
                &[
                    &classification.event_id,
                    &classification.repo,
                    &classification.session_id,
                    &classification.trajectory_id,
                    &classification.event_created_at,
                    &classification.classified_at,
                    &classification.classification_schema_version,
                    &classification.routing_policy_version,
                    &classification.classifier_source,
                    &classification.intent.as_str(),
                    &classification.domain.as_str(),
                    &secondary_domains,
                    &classification.artifact_type.as_str(),
                    &risk,
                    &classification.complexity.as_str(),
                    &classification.recommended_route.as_str(),
                    &classification.response_contract.as_str(),
                    &classification.features,
                ],
            )
            .await?;
        Ok::<PersistOutcome, anyhow::Error>(if affected == 1 {
            PersistOutcome::Inserted
        } else {
            PersistOutcome::Skipped
        })
    }
    .await;

    match &result {
        Ok(outcome) => {
            crate::telemetry::record_request_classification_write(outcome.as_str());
            if matches!(outcome, PersistOutcome::Inserted) {
                crate::request_classification::record_classification_metrics(classification);
            }
        }
        Err(_) => crate::telemetry::record_request_classification_write("error"),
    }
    result
}

pub async fn update_classification_if_changed(
    pool: &Pool,
    classification: &RequestClassification,
) -> Result<PersistOutcome, anyhow::Error> {
    let result = async {
        let conn = pool.get().await?;
        let secondary_domains = classification
            .secondary_domains
            .iter()
            .map(|domain| domain.as_str().to_string())
            .collect::<Vec<_>>();
        let risk = classification
            .risk
            .iter()
            .map(|risk| risk.as_str().to_string())
            .collect::<Vec<_>>();
        let affected = conn
            .execute(
                "UPDATE agent_request_classifications
                 SET
                    repo = $2,
                    session_id = $3,
                    trajectory_id = $4,
                    event_created_at = $5,
                    classified_at = $6,
                    classifier_source = $9,
                    intent = $10,
                    domain = $11,
                    secondary_domains = $12,
                    artifact_type = $13,
                    risk = $14,
                    complexity = $15,
                    recommended_route = $16,
                    response_contract = $17,
                    features = $18
                 WHERE event_id = $1
                   AND classification_schema_version = $7
                   AND routing_policy_version = $8
                   AND (
                       repo IS DISTINCT FROM $2
                       OR session_id IS DISTINCT FROM $3
                       OR trajectory_id IS DISTINCT FROM $4
                       OR event_created_at IS DISTINCT FROM $5
                       OR classified_at IS DISTINCT FROM $6
                       OR classifier_source IS DISTINCT FROM $9
                       OR intent IS DISTINCT FROM $10
                       OR domain IS DISTINCT FROM $11
                       OR secondary_domains IS DISTINCT FROM $12
                       OR artifact_type IS DISTINCT FROM $13
                       OR risk IS DISTINCT FROM $14
                       OR complexity IS DISTINCT FROM $15
                       OR recommended_route IS DISTINCT FROM $16
                       OR response_contract IS DISTINCT FROM $17
                       OR features IS DISTINCT FROM $18
                   )",
                &[
                    &classification.event_id,
                    &classification.repo,
                    &classification.session_id,
                    &classification.trajectory_id,
                    &classification.event_created_at,
                    &classification.classified_at,
                    &classification.classification_schema_version,
                    &classification.routing_policy_version,
                    &classification.classifier_source,
                    &classification.intent.as_str(),
                    &classification.domain.as_str(),
                    &secondary_domains,
                    &classification.artifact_type.as_str(),
                    &risk,
                    &classification.complexity.as_str(),
                    &classification.recommended_route.as_str(),
                    &classification.response_contract.as_str(),
                    &classification.features,
                ],
            )
            .await?;
        Ok::<PersistOutcome, anyhow::Error>(if affected == 1 {
            PersistOutcome::Updated
        } else {
            PersistOutcome::Skipped
        })
    }
    .await;

    match &result {
        Ok(outcome) => {
            crate::telemetry::record_request_classification_write(outcome.as_str());
            if matches!(outcome, PersistOutcome::Updated) {
                crate::request_classification::record_classification_metrics(classification);
            }
        }
        Err(_) => crate::telemetry::record_request_classification_write("error"),
    }
    result
}
