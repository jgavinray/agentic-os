use crate::orchestration_policy_types::{
    OrchestrationPolicy, POLICY_SCHEMA_VERSION, POLICY_SOURCE_DETERMINISTIC_RULES,
};
use crate::request_classification::RequestClassification;

/// Persist an orchestration policy to the append-only ledger.
///
/// This is a pure INSERT with no ON CONFLICT. Every call appends a new row.
pub async fn persist_orchestration_policy(
    pool: &deadpool_postgres::Pool,
    classification: &RequestClassification,
    policy: &OrchestrationPolicy,
) -> Result<uuid::Uuid, anyhow::Error> {
    use chrono::Utc;
    let started = std::time::Instant::now();

    let policy_id = uuid::Uuid::new_v4();

    // event_id: NULL when classification.event_id == "live-request"
    let event_id = if classification.event_id == "live-request" {
        None
    } else {
        Some(classification.event_id.clone())
    };

    let context_sources: serde_json::Value = serde_json::json!(policy
        .context_sources
        .iter()
        .map(|s| s.as_str())
        .collect::<Vec<_>>());
    let allowed_tools: serde_json::Value = serde_json::json!(policy
        .allowed_tools
        .iter()
        .map(|t| t.as_str())
        .collect::<Vec<_>>());
    let required_tools: serde_json::Value = serde_json::json!(policy
        .required_tools
        .iter()
        .map(|t| t.as_str())
        .collect::<Vec<_>>());
    let blocked_tools: serde_json::Value = serde_json::json!(policy
        .blocked_tools
        .iter()
        .map(|t| t.as_str())
        .collect::<Vec<_>>());
    let scope_policy: serde_json::Value = serde_json::json!(policy
        .scope_policy
        .iter()
        .map(|s| s.as_str())
        .collect::<Vec<_>>());
    let risk_policy: serde_json::Value = serde_json::json!(policy
        .risk_policy
        .iter()
        .map(|r| r.as_str())
        .collect::<Vec<_>>());

    let result = async {
        let conn = pool.get().await?;
        conn.execute(
            "INSERT INTO agent_orchestration_policies (
                policy_id,
                event_id,
                session_id,
                repo,
                created_at,
                classification_schema_version,
                routing_policy_version,
                policy_schema_version,
                intent,
                recommended_route,
                context_sources,
                allowed_tools,
                required_tools,
                blocked_tools,
                edit_policy,
                validation_policy,
                git_policy,
                runtime_policy,
                scope_policy,
                prompt_refinement_policy,
                risk_policy,
                source
            ) VALUES (
                $1, $2, $3, $4, $5, $6, $7, $8, $9, $10,
                $11, $12, $13, $14, $15, $16, $17, $18, $19, $20, $21, $22
            )",
            &[
                &policy_id,
                &event_id,
                &classification.session_id,
                &classification.repo,
                &Utc::now(),
                &classification.classification_schema_version,
                &classification.routing_policy_version,
                &POLICY_SCHEMA_VERSION,
                &classification.intent.as_str(),
                &classification.recommended_route.as_str(),
                &context_sources,
                &allowed_tools,
                &required_tools,
                &blocked_tools,
                &policy.edit_policy.as_str(),
                &policy.validation_policy.as_str(),
                &policy.git_policy.as_str(),
                &policy.runtime_policy.as_str(),
                &scope_policy,
                &policy.prompt_refinement_policy.as_str(),
                &risk_policy,
                &POLICY_SOURCE_DETERMINISTIC_RULES,
            ],
        )
        .await?;
        Ok::<uuid::Uuid, anyhow::Error>(policy_id)
    }
    .await;

    crate::telemetry::record_db_query(
        "persist_orchestration_policy",
        started.elapsed(),
        result.is_ok(),
    );

    result
}

/// Build a compact JSON representation of an orchestration policy.
///
/// Used for request/tool event metadata and telemetry. The append-only policy
/// ledger still persists the normalized columns separately.
pub fn compact_policy_metadata(
    classification: &RequestClassification,
    policy: &OrchestrationPolicy,
) -> serde_json::Value {
    serde_json::json!({
        "intent": classification.intent.as_str(),
        "recommended_route": classification.recommended_route.as_str(),
        "context_sources": policy.context_sources.iter().map(|s| s.as_str()).collect::<Vec<_>>(),
        "allowed_tools": policy.allowed_tools.iter().map(|t| t.as_str()).collect::<Vec<_>>(),
        "required_tools": policy.required_tools.iter().map(|t| t.as_str()).collect::<Vec<_>>(),
        "blocked_tools": policy.blocked_tools.iter().map(|t| t.as_str()).collect::<Vec<_>>(),
        "edit_policy": policy.edit_policy.as_str(),
        "validation_policy": policy.validation_policy.as_str(),
        "git_policy": policy.git_policy.as_str(),
        "runtime_policy": policy.runtime_policy.as_str(),
        "scope_policy": policy.scope_policy.iter().map(|s| s.as_str()).collect::<Vec<_>>(),
        "prompt_refinement_policy": policy.prompt_refinement_policy.as_str(),
        "risk_policy": policy.risk_policy.iter().map(|r| r.as_str()).collect::<Vec<_>>(),
        "policy_schema_version": POLICY_SCHEMA_VERSION,
        "source": POLICY_SOURCE_DETERMINISTIC_RULES,
    })
}
