use deadpool_postgres::Pool;
use serde_json::json;
use uuid::Uuid;

use crate::orchestration_policy::OrchestrationPolicy;
use crate::request_classification::RequestClassification;
use crate::tool_mediation_types::ToolMenuOutcome;

/// Initialize capture-side storage for shaped tool-menu decisions.
pub async fn init(pool: &Pool) -> Result<(), anyhow::Error> {
    let conn = pool.get().await?;
    conn.batch_execute(
        "CREATE TABLE IF NOT EXISTS tool_mediation_decisions (
            id UUID PRIMARY KEY,
            exchange_id UUID NOT NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            endpoint TEXT NOT NULL,
            request_intent TEXT NOT NULL,
            request_domain TEXT NOT NULL,
            recommended_route TEXT NOT NULL,
            edit_policy TEXT NOT NULL,
            validation_policy TEXT NOT NULL,
            runtime_policy TEXT NOT NULL,
            tool_policy_version TEXT NOT NULL,
            endpoint_format TEXT NOT NULL,
            tool_intent TEXT NOT NULL,
            decision TEXT NOT NULL,
            reason TEXT NOT NULL,
            offered_tools JSONB NOT NULL DEFAULT '[]'::jsonb,
            allowed_tools JSONB NOT NULL DEFAULT '[]'::jsonb,
            hidden_tools JSONB NOT NULL DEFAULT '[]'::jsonb,
            offered_capabilities JSONB NOT NULL DEFAULT '[]'::jsonb,
            allowed_capabilities JSONB NOT NULL DEFAULT '[]'::jsonb,
            hidden_capabilities JSONB NOT NULL DEFAULT '[]'::jsonb,
            missing_required_capabilities JSONB NOT NULL DEFAULT '[]'::jsonb,
            tool_choice_changed BOOLEAN NOT NULL DEFAULT false
        );
        CREATE INDEX IF NOT EXISTS tool_mediation_decisions_exchange_id_idx
            ON tool_mediation_decisions(exchange_id);
        CREATE INDEX IF NOT EXISTS tool_mediation_decisions_created_at_idx
            ON tool_mediation_decisions(created_at DESC);
        CREATE INDEX IF NOT EXISTS tool_mediation_decisions_request_intent_idx
            ON tool_mediation_decisions(request_intent);
        CREATE INDEX IF NOT EXISTS tool_mediation_decisions_decision_reason_idx
            ON tool_mediation_decisions(decision, reason);",
    )
    .await?;
    Ok(())
}

pub async fn insert(
    pool: &Pool,
    endpoint: &'static str,
    exchange_id: Uuid,
    classification: &RequestClassification,
    policy: &OrchestrationPolicy,
    outcome: &ToolMenuOutcome,
) -> Result<(), anyhow::Error> {
    let missing_required_capabilities =
        if crate::tool_mediation_types::policy_requires_implementation_tool_surface(policy) {
            crate::tool_mediation_types::missing_implementation_tool_capabilities(outcome)
        } else {
            Vec::new()
        };
    let offered_capabilities = outcome.offered_capabilities().collect::<Vec<_>>();
    let allowed_capabilities = outcome.allowed_capabilities().collect::<Vec<_>>();
    let hidden_capabilities = outcome.hidden_capabilities().collect::<Vec<_>>();

    let conn = pool.get().await?;
    conn.execute(
        "INSERT INTO tool_mediation_decisions
         (id, exchange_id, endpoint, request_intent, request_domain,
          recommended_route, edit_policy, validation_policy, runtime_policy,
          tool_policy_version, endpoint_format, tool_intent, decision, reason,
          offered_tools, allowed_tools, hidden_tools, offered_capabilities,
          allowed_capabilities, hidden_capabilities, missing_required_capabilities,
          tool_choice_changed)
         VALUES
         ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14,
          $15, $16, $17, $18, $19, $20, $21, $22)",
        &[
            &Uuid::new_v4(),
            &exchange_id,
            &endpoint,
            &classification.intent.as_str(),
            &classification.domain.as_str(),
            &classification.recommended_route.as_str(),
            &policy.edit_policy.as_str(),
            &policy.validation_policy.as_str(),
            &policy.runtime_policy.as_str(),
            &outcome.policy_version,
            &outcome.endpoint_format,
            &outcome.intent,
            &outcome.decision,
            &outcome.reason,
            &json!(outcome.offered_tools),
            &json!(outcome.allowed_tools),
            &json!(outcome.hidden_tools),
            &json!(offered_capabilities),
            &json!(allowed_capabilities),
            &json!(hidden_capabilities),
            &json!(missing_required_capabilities),
            &outcome.tool_choice_changed,
        ],
    )
    .await?;
    Ok(())
}

pub fn insert_best_effort(
    pool: Option<&Pool>,
    endpoint: &'static str,
    exchange_id: Uuid,
    classification: &RequestClassification,
    policy: &OrchestrationPolicy,
    outcome: &ToolMenuOutcome,
) {
    let Some(pool) = pool.cloned() else {
        return;
    };
    let classification = classification.clone();
    let policy = policy.clone();
    let outcome = outcome.clone();
    tokio::spawn(async move {
        if let Err(error) = insert(
            &pool,
            endpoint,
            exchange_id,
            &classification,
            &policy,
            &outcome,
        )
        .await
        {
            tracing::warn!(
                exchange_id = %exchange_id,
                endpoint,
                "failed to persist tool mediation decision: {error}"
            );
        }
    });
}

#[cfg(test)]
mod tests {
    #[test]
    fn decision_table_contains_queryable_audit_columns() {
        let source = include_str!("decision_store.rs");
        for column in [
            "exchange_id",
            "request_intent",
            "recommended_route",
            "edit_policy",
            "offered_tools",
            "allowed_tools",
            "hidden_tools",
            "missing_required_capabilities",
        ] {
            assert!(source.contains(column), "missing column {column}");
        }
    }
}
