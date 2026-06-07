use deadpool_postgres::Pool;

pub use crate::context_rendering::{build_context, build_layered_context, estimate_tokens};
pub use crate::db_connection::{acquire_single_writer_lock, create_pool, SingleWriterGuard};
pub use crate::db_context_artifacts::{
    get_active_context_artifacts, get_context_artifacts_for_repo, get_context_compiler_ledger,
    get_recent_instruction_candidates, get_recent_repo_decision_candidates,
    get_recent_session_events, insert_context_compiler_ledger_entry, upsert_context_artifact,
};
pub use crate::db_context_events::{
    count_events_for_repo, get_context_events_for_repo, get_context_evidence_for_policy,
    get_events_for_repo, hydrate_active_search_hits, preferred_summary_levels, search_events_fts,
};
pub use crate::db_execution_events::{
    append_execution_event, insert_compile_result_event, insert_execution_artifact_event,
    insert_lint_result_event, insert_patch_result_event, insert_remediation_event,
    insert_test_result_event, insert_tool_result_event, insert_validation_result_event,
};
pub use crate::db_failures::{
    get_active_errors, get_failure_history_for_signatures, get_recent_failure_history,
    insert_error_record, warn_if_legacy_signature_backfill_pending,
};
pub use crate::db_sessions::{
    append_event_from_request, create_session, event_from_append_request, find_or_create_session,
    start_session_from_request,
};
pub use crate::db_trajectory::{
    emit_trajectory_result_once, get_event_chain_by_event_id, get_trajectory,
    get_trajectory_attempts, get_trajectory_result, idle_trajectory_ids,
    latest_trajectory_event_for_session, order_event_chain,
};
pub use crate::db_types::{
    AgentEvent, ContextCompilerLedgerEntry, ContextEvidence, FailureHistoryItem,
    VllmCacheObservationInput, VllmCacheStats,
};
pub use crate::db_usage::{
    get_vllm_cache_stats, insert_vllm_cache_observation, record_token_usage,
};

// ── DB query functions ────────────────────────────────────────

pub async fn insert_event(pool: &Pool, event: &AgentEvent) -> Result<(), anyhow::Error> {
    crate::trajectory::validate_event_role(event.event_role.as_deref())?;
    if matches!(event.attempt_index, Some(attempt) if attempt < 1) {
        anyhow::bail!("attempt_index must be positive");
    }
    let lineage_fields = [
        event.trajectory_id.is_some(),
        event.attempt_index.is_some(),
        event.event_role.is_some(),
    ];
    if lineage_fields.iter().any(|present| *present)
        && !lineage_fields.iter().all(|present| *present)
    {
        anyhow::bail!("trajectory_id, attempt_index, and event_role must be written together");
    }
    let metadata = crate::harness_feedback::annotate_event_metadata(
        &event.event_type,
        &event.summary,
        event.evidence.as_deref(),
        event.metadata.clone(),
    );
    let metadata = crate::feature_extraction::annotate_event_metadata(
        &event.event_type,
        &event.summary,
        event.evidence.as_deref(),
        metadata,
    );
    let started = std::time::Instant::now();
    let result = async {
        let conn = pool.get().await?;
        conn.execute(
            "INSERT INTO agent_events
             (id, session_id, repo, actor, event_type, summary, evidence, metadata, correlation_id, parent_event_id, trajectory_id, attempt_index, event_role, summary_level)
             VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14)",
            &[
                &event.id,
                &event.session_id,
                &event.repo,
                &event.actor,
                &event.event_type,
                &event.summary,
                &event.evidence,
                &metadata,
                &event.correlation_id,
                &event.parent_event_id,
                &event.trajectory_id,
                &event.attempt_index,
                &event.event_role,
                &event.summary_level,
            ],
        )
        .await?;
        Ok(())
    }
    .await;
    crate::telemetry::record_db_query("insert_event", started.elapsed(), result.is_ok());
    if result.is_ok() {
        crate::harness_feedback::record_metadata_metrics(&metadata);
        let classification_event = AgentEvent {
            metadata,
            ..event.clone()
        };
        if let Err(e) =
            crate::request_classification::classify_and_persist_event(pool, &classification_event)
                .await
        {
            tracing::warn!(
                target: "request_classification",
                event_id = %event.id,
                event_type = %event.event_type,
                "failed to persist request classification for new event: {e}"
            );
        }
        tracing::info!(
            target: "execution_feedback",
            event_type = %event.event_type,
            event_id = %event.id,
            correlation_id = ?event.correlation_id,
            "event written"
        );
    }
    result
}

// ── Context pack builder ──────────────────────────────────────

pub async fn check_ready(pool: &deadpool_postgres::Pool) -> Result<(), anyhow::Error> {
    let started = std::time::Instant::now();
    let result = async {
        let conn = pool.get().await?;
        let _ = conn.query_one("SELECT 1", &[]).await?;
        Ok(())
    }
    .await;
    crate::telemetry::record_db_query("check_ready", started.elapsed(), result.is_ok());
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state::{ErrorRecord, SearchHit};
    use chrono::Utc;

    fn mem(event_type: &str, summary: &str, evidence: Option<&str>) -> crate::state::EventMemory {
        crate::state::EventMemory {
            event_type: event_type.to_string(),
            summary: summary.to_string(),
            evidence: evidence.map(str::to_string),
            metadata: serde_json::json!({}),
            created_at: Utc::now(),
            summary_level: 0,
        }
    }

    fn mem_with_meta(
        event_type: &str,
        summary: &str,
        metadata: serde_json::Value,
    ) -> crate::state::EventMemory {
        crate::state::EventMemory {
            event_type: event_type.to_string(),
            summary: summary.to_string(),
            evidence: None,
            metadata,
            created_at: Utc::now(),
            summary_level: 0,
        }
    }

    fn hit(event_type: &str, summary: &str) -> SearchHit {
        SearchHit {
            event_id: uuid::Uuid::new_v4().to_string(),
            event_type: event_type.to_string(),
            summary: summary.to_string(),
            created_at: Some(Utc::now()),
        }
    }

    fn err_rec(error_type: &str, description: &str, frequency: i64) -> ErrorRecord {
        ErrorRecord {
            id: "eid".to_string(),
            repo: "r".to_string(),
            task: "t".to_string(),
            error_type: error_type.to_string(),
            description: description.to_string(),
            severity: "medium".to_string(),
            frequency,
            last_seen: Utc::now(),
        }
    }

    fn event(summary: &str) -> AgentEvent {
        AgentEvent {
            id: "tid".to_string(),
            session_id: "tsession".to_string(),
            repo: "trepo".to_string(),
            actor: "tactor".to_string(),
            event_type: "decision".to_string(),
            summary: summary.to_string(),
            evidence: None,
            metadata: serde_json::json!({}),
            correlation_id: None,
            parent_event_id: None,
            trajectory_id: None,
            attempt_index: None,
            event_role: None,
            created_at: Utc::now(),
            summary_level: 0,
        }
    }

    fn event_at_level(event_type: &str, summary: &str, level: i32) -> AgentEvent {
        AgentEvent {
            id: uuid::Uuid::new_v4().to_string(),
            session_id: "session".to_string(),
            repo: "repo".to_string(),
            actor: "actor".to_string(),
            event_type: event_type.to_string(),
            summary: summary.to_string(),
            evidence: None,
            metadata: serde_json::json!({}),
            correlation_id: None,
            parent_event_id: None,
            trajectory_id: None,
            attempt_index: None,
            event_role: None,
            created_at: Utc::now(),
            summary_level: level,
        }
    }

    fn failed_event(summary: &str, evidence: &str, outcome: &str) -> AgentEvent {
        AgentEvent {
            evidence: Some(evidence.to_string()),
            metadata: serde_json::json!({
                "source_event_ids": ["a", "b"],
                "outcome": outcome,
            }),
            ..event_at_level("failed_attempt", summary, 0)
        }
    }

    #[test]
    fn build_context_header_contains_repo_and_task() {
        let out = build_context("my-repo", "my-task", &[], &[], &[], 8000);
        assert!(out.starts_with("Repository: my-repo\nTask: my-task\n"));
    }

    #[test]
    fn build_context_empty_shows_no_prior_memory() {
        let out = build_context("r", "t", &[], &[], &[], 8000);
        assert!(out.contains("No relevant prior memory found"));
    }

    #[test]
    fn build_context_hybrid_hit_appears_in_relevant_section() {
        let hits = vec![hit("edit", "changed auth flow")];
        let out = build_context("r", "t", &[], &hits, &[], 8000);
        assert!(out.contains("== Relevant Prior Memory =="));
        assert!(out.contains("changed auth flow"));
    }

    #[test]
    fn build_context_decision_event_appears_in_decisions_section() {
        let recent = vec![mem(
            "decision",
            "chose retry logic",
            Some("503 intermittent"),
        )];
        let out = build_context("r", "t", &recent, &[], &[], 8000);
        assert!(out.contains("== Recent Decisions =="));
        assert!(out.contains("chose retry logic"));
        assert!(out.contains("Evidence: 503 intermittent"));
    }

    #[test]
    fn build_context_non_decision_recent_not_in_decisions_section() {
        let recent = vec![mem("edit", "changed a file", None)];
        let out = build_context("r", "t", &recent, &[], &[], 8000);
        assert!(!out.contains("== Recent Decisions =="));
    }

    #[test]
    fn build_context_error_appears_in_failed_approaches_section() {
        let errs = vec![err_rec("auth_500", "OAuth callback 500", 3)];
        let out = build_context("r", "t", &[], &[], &errs, 8000);
        assert!(out.contains("== Failed Approaches (do not retry) =="));
        assert!(out.contains("OAuth callback 500"));
        assert!(out.contains("seen 3 times"));
    }

    #[test]
    fn build_context_open_questions_from_checkpoint_metadata() {
        let recent = vec![mem_with_meta(
            "checkpoint",
            "end of sprint",
            serde_json::json!({
                "open_questions": ["Is retry threshold right?", "Do we need caching?"]
            }),
        )];
        let out = build_context("r", "t", &recent, &[], &[], 8000);
        assert!(out.contains("== Open Questions =="));
        assert!(out.contains("Is retry threshold right?"));
    }

    #[test]
    fn build_context_empty_sections_omitted() {
        let hits = vec![hit("edit", "some memory")];
        let out = build_context("r", "t", &[], &hits, &[], 8000);
        assert!(!out.contains("== Recent Decisions =="));
        assert!(!out.contains("== Failed Approaches"));
        assert!(!out.contains("== Open Questions =="));
    }

    #[test]
    fn build_context_deduplicates_same_summary_across_sections() {
        let shared = "shared summary text";
        let hits = vec![hit("edit", shared)];
        let recent = vec![mem("decision", shared, None)];
        let out = build_context("r", "t", &recent, &hits, &[], 8000);
        assert_eq!(out.matches(shared).count(), 1);
    }

    #[test]
    fn build_context_truncates_hybrid_section_at_budget() {
        let hits: Vec<SearchHit> = (0..20)
            .map(|i| {
                hit(
                    "edit",
                    &format!(
                        "event {i} with enough text to consume the hybrid budget allocation here"
                    ),
                )
            })
            .collect();
        let out = build_context("r", "t", &[], &hits, &[], 500);
        assert!(out.contains("truncated"));
    }

    #[test]
    fn context_policy_small_task_includes_l0_and_l1() {
        let mut evidence = ContextEvidence::default();
        evidence.l0_recent = vec![event_at_level("edit", "fresh implementation detail", 0)];
        evidence.l1_matching = vec![event_at_level("summary", "matching prior task summary", 1)];
        evidence.l2_repo = vec![event_at_level("summary", "repo convention", 2)];

        let policy = crate::state::ContextPolicy::for_category(crate::state::TaskCategory::Narrow);
        let (out, stats) =
            build_layered_context("r", "fix bug", &evidence, &[], &[], &policy, 8000, 4000);

        assert!(out.contains("== Recent Evidence =="));
        assert!(out.contains("fresh implementation detail"));
        assert!(out.contains("== Relevant Session Summaries =="));
        assert!(out.contains("matching prior task summary"));
        assert_eq!(stats.l0_items_injected, 1);
        assert_eq!(stats.l1_items_injected, 1);
    }

    #[test]
    fn context_policy_broad_task_includes_l1_l2_and_l0_tail() {
        let mut evidence = ContextEvidence::default();
        evidence.l0_recent = vec![event_at_level("edit", "latest local change", 0)];
        evidence.l1_matching = vec![event_at_level("summary", "session summary", 1)];
        evidence.l2_repo = vec![event_at_level("summary", "repo-level pattern", 2)];

        let policy = crate::state::ContextPolicy::for_category(crate::state::TaskCategory::Broad);
        let (out, _stats) = build_layered_context(
            "r",
            "large refactor",
            &evidence,
            &[],
            &[],
            &policy,
            8000,
            4000,
        );

        assert!(out.contains("latest local change"));
        assert!(out.contains("session summary"));
        assert!(out.contains("repo-level pattern"));
    }

    #[test]
    fn context_policy_architecture_task_prioritizes_l2_l3() {
        let mut evidence = ContextEvidence::default();
        evidence.l2_repo = vec![event_at_level("summary", "repository decision record", 2)];
        evidence.l3_project = vec![event_at_level("summary", "durable architecture truth", 3)];

        let policy =
            crate::state::ContextPolicy::for_category(crate::state::TaskCategory::Architecture);
        let (out, stats) = build_layered_context(
            "r",
            "architecture review",
            &evidence,
            &[],
            &[],
            &policy,
            8000,
            4000,
        );

        let l3_pos = out.find("durable architecture truth").unwrap();
        let l2_pos = out.find("repository decision record").unwrap();
        assert!(l3_pos < l2_pos);
        assert_eq!(stats.l2_items_injected, 1);
        assert_eq!(stats.l3_items_injected, 1);
    }

    #[test]
    fn higher_levels_do_not_eliminate_l0_tail() {
        let mut evidence = ContextEvidence::default();
        evidence.l0_recent = vec![event_at_level("edit", "raw event still included", 0)];
        evidence.l3_project = vec![event_at_level("summary", "durable compressed memory", 3)];

        let policy =
            crate::state::ContextPolicy::for_category(crate::state::TaskCategory::Architecture);
        let (out, _stats) = build_layered_context(
            "r",
            "architecture",
            &evidence,
            &[],
            &[],
            &policy,
            8000,
            4000,
        );

        assert!(out.contains("raw event still included"));
        assert!(out.contains("durable compressed memory"));
    }

    #[test]
    fn failed_attempt_preserves_evidence_and_outcome() {
        let mut evidence = ContextEvidence::default();
        evidence.failures = vec![failed_event(
            "retrying the old cache key caused stale context",
            "observed repeated stale pack after append_event",
            "new key includes event count",
        )];

        let policy = crate::state::ContextPolicy::for_category(crate::state::TaskCategory::Narrow);
        let (out, stats) =
            build_layered_context("r", "fix cache", &evidence, &[], &[], &policy, 8000, 4000);

        assert!(out.contains("== Failed Attempts and Remediations =="));
        assert!(out.contains("observed repeated stale pack after append_event"));
        assert!(out.contains("Outcome: new key includes event count"));
        assert!(out.contains("Source IDs: 2"));
        assert_eq!(stats.failed_attempts_injected, 1);
    }

    #[test]
    fn failure_history_section_sits_between_failures_and_open_questions() {
        let mut evidence = ContextEvidence::default();
        evidence.failures = vec![failed_event(
            "old patch broke cache invalidation",
            "cargo test failed",
            "invalidate by repo prefix",
        )];
        evidence.failure_history = vec![FailureHistoryItem {
            signature: "rust:type-mismatch".to_string(),
            category: "type_error".to_string(),
            failure: AgentEvent {
                event_type: crate::execution_feedback::EVENT_TYPE_COMPILE_RESULT.to_string(),
                summary: "failed compile `rust` target `cargo` errors=1 warnings=0".to_string(),
                metadata: serde_json::json!({
                    "payload": {
                        "signature": "rust:type-mismatch",
                        "signature_category": "type_error",
                        "fingerprint_version": 1,
                        "failure_summaries": ["error[E0308]: mismatched types"]
                    }
                }),
                ..event_at_level(
                    crate::execution_feedback::EVENT_TYPE_COMPILE_RESULT,
                    "failed compile `rust` target `cargo` errors=1 warnings=0",
                    0,
                )
            },
            remediation: Some(AgentEvent {
                event_type: crate::execution_feedback::EVENT_TYPE_REMEDIATION.to_string(),
                summary: "changed generic bound to match caller".to_string(),
                metadata: serde_json::json!({
                    "payload": {"signature": "rust:type-mismatch"}
                }),
                ..event_at_level(
                    crate::execution_feedback::EVENT_TYPE_REMEDIATION,
                    "changed generic bound to match caller",
                    0,
                )
            }),
        }];
        evidence.l0_recent = vec![AgentEvent {
            event_type: "checkpoint".to_string(),
            metadata: serde_json::json!({"open_questions": ["Should cache TTL change?"]}),
            ..event_at_level("checkpoint", "checkpoint", 0)
        }];

        let policy = crate::state::ContextPolicy::for_category(crate::state::TaskCategory::Narrow);
        let (out, stats) = build_layered_context(
            "r",
            "error[E0308]",
            &evidence,
            &[],
            &[],
            &policy,
            8000,
            4000,
        );

        let failed = out.find("== Failed Attempts and Remediations ==").unwrap();
        let history = out.find("== Failure History ==").unwrap();
        let open = out.find("== Open Questions ==").unwrap();
        assert!(failed < history);
        assert!(history < open);
        assert!(out.contains("changed generic bound to match caller"));
        assert_eq!(stats.failure_history_items_injected, 1);
        assert_eq!(
            stats.failure_history_remediation_signatures,
            vec!["rust:type-mismatch".to_string()]
        );
    }

    #[test]
    fn operational_constraints_section_sits_above_failure_history() {
        let mut evidence = ContextEvidence::default();
        evidence.operational_constraints = vec![crate::feature_extraction::OperationalConstraint {
            constraint_type: "use_known_endpoint".to_string(),
            text: "Do not use `localhost` for host-side orchestrator testing. The correct endpoint for this environment is `http://host.docker.internal:8088`.".to_string(),
        }];
        evidence.failure_history = vec![FailureHistoryItem {
            signature: "process:non-zero-exit".to_string(),
            category: "unknown".to_string(),
            failure: event_at_level(
                crate::execution_feedback::EVENT_TYPE_TOOL_RESULT,
                "failed tool `Bash` exit_code=1",
                0,
            ),
            remediation: None,
        }];

        let policy = crate::state::ContextPolicy::for_category(crate::state::TaskCategory::Narrow);
        let (out, stats) =
            build_layered_context("r", "debug", &evidence, &[], &[], &policy, 8000, 4000);

        let constraints = out.find("Operational Constraints:").unwrap();
        let history = out.find("== Failure History ==").unwrap();
        assert!(constraints < history);
        assert!(out.contains("- Do not use `localhost`"));
        assert_eq!(stats.operational_constraints_injected, 1);
    }

    #[test]
    fn failure_history_section_omitted_when_empty() {
        let evidence = ContextEvidence::default();
        let policy = crate::state::ContextPolicy::for_category(crate::state::TaskCategory::Narrow);
        let (out, stats) =
            build_layered_context("r", "no failures", &evidence, &[], &[], &policy, 8000, 4000);

        assert!(!out.contains("== Failure History =="));
        assert_eq!(stats.failure_history_items_injected, 0);
    }

    #[test]
    fn failure_history_section_respects_budget() {
        let mut evidence = ContextEvidence::default();
        evidence.failure_history = vec![FailureHistoryItem {
            signature: "unknown".to_string(),
            category: "unknown".to_string(),
            failure: AgentEvent {
                event_type: crate::execution_feedback::EVENT_TYPE_VALIDATION_RESULT.to_string(),
                summary: "x".repeat(200),
                metadata: serde_json::json!({
                    "payload": {
                        "signature": "unknown",
                        "signature_category": "unknown",
                        "fingerprint_version": 1,
                        "failure_reason": "y".repeat(200)
                    }
                }),
                ..event_at_level(
                    crate::execution_feedback::EVENT_TYPE_VALIDATION_RESULT,
                    "long failure",
                    0,
                )
            },
            remediation: None,
        }];
        let policy = crate::state::ContextPolicy::for_category(crate::state::TaskCategory::Narrow);
        let (out, _stats) =
            build_layered_context("r", "unknown", &evidence, &[], &[], &policy, 8000, 50);
        assert!(out.contains("truncated: failure history budget exceeded"));
    }

    #[test]
    fn event_chain_orders_from_root_to_leaf() {
        let correlation_id = uuid::Uuid::new_v4();
        let root_id = uuid::Uuid::new_v4();
        let response_id = uuid::Uuid::new_v4();
        let tool_id = uuid::Uuid::new_v4();
        let validation_id = uuid::Uuid::new_v4();

        let mut root = event_at_level("user_message", "request", 0);
        root.id = root_id.to_string();
        root.correlation_id = Some(correlation_id);
        root.parent_event_id = None;

        let mut response = event_at_level("assistant_message", "patch", 0);
        response.id = response_id.to_string();
        response.correlation_id = Some(correlation_id);
        response.parent_event_id = Some(root_id);

        let mut tool = event_at_level(
            crate::execution_feedback::EVENT_TYPE_TOOL_RESULT,
            "apply",
            0,
        );
        tool.id = tool_id.to_string();
        tool.correlation_id = Some(correlation_id);
        tool.parent_event_id = Some(response_id);

        let mut validation = event_at_level(
            crate::execution_feedback::EVENT_TYPE_VALIDATION_RESULT,
            "schema ok",
            0,
        );
        validation.id = validation_id.to_string();
        validation.correlation_id = Some(correlation_id);
        validation.parent_event_id = Some(tool_id);

        let ordered = order_event_chain(
            vec![
                validation.clone(),
                tool.clone(),
                root.clone(),
                response.clone(),
            ],
            &validation.id,
        );
        let ids: Vec<String> = ordered.into_iter().map(|event| event.id).collect();
        assert_eq!(
            ids,
            vec![
                root_id.to_string(),
                response_id.to_string(),
                tool_id.to_string(),
                validation_id.to_string()
            ]
        );
    }

    #[test]
    fn failure_history_context_build_p99_under_five_ms() {
        let mut durations = Vec::new();
        let policy = crate::state::ContextPolicy::for_category(crate::state::TaskCategory::Narrow);
        let mut evidence = ContextEvidence::default();
        evidence.failure_history = (0..50)
            .map(|idx| FailureHistoryItem {
                signature: format!("typescript:TS{idx:04}"),
                category: "typescript".to_string(),
                failure: event_at_level(
                    crate::execution_feedback::EVENT_TYPE_COMPILE_RESULT,
                    &format!("failure {idx}"),
                    0,
                ),
                remediation: Some(event_at_level(
                    crate::execution_feedback::EVENT_TYPE_REMEDIATION,
                    &format!("remediation {idx}"),
                    0,
                )),
            })
            .collect();

        for _ in 0..100 {
            let started = std::time::Instant::now();
            let _ = build_layered_context(
                "r",
                "typescript:TS2322",
                &evidence,
                &[],
                &[],
                &policy,
                8000,
                4000,
            );
            durations.push(started.elapsed());
        }
        durations.sort();
        let p99 = durations[durations.len() - 1];
        let threshold = if cfg!(debug_assertions) {
            std::time::Duration::from_millis(20)
        } else {
            std::time::Duration::from_millis(5)
        };
        assert!(p99 < threshold, "p99 was {p99:?}");
    }

    #[test]
    fn promoted_memory_requires_source_ids() {
        assert!(!crate::summarizer::has_source_ids(&serde_json::json!({})));
        assert!(!crate::summarizer::has_source_ids(
            &serde_json::json!({"summarized_event_ids": []})
        ));
        assert!(crate::summarizer::has_source_ids(
            &serde_json::json!({"summarized_event_ids": ["e1"]})
        ));
        assert!(crate::summarizer::has_source_ids(
            &serde_json::json!({"source_event_ids": ["e1"]})
        ));
    }

    #[test]
    fn vector_text_contains_all_fields() {
        let mut e = event("summary text");
        e.evidence = Some("evidence text".to_string());
        let t = e.vector_text();
        assert!(t.contains("trepo"));
        assert!(t.contains("decision"));
        assert!(t.contains("summary text"));
        assert!(t.contains("evidence text"));
    }

    #[test]
    fn vector_text_with_no_evidence_does_not_panic() {
        let e = event("summary text");
        let t = e.vector_text();
        assert!(t.contains("summary text"));
    }

    #[test]
    fn payload_contains_required_fields() {
        let e = event("payload summary");
        let p = e.payload();
        assert_eq!(p["event_id"], "tid");
        assert_eq!(p["repo"], "trepo");
        assert_eq!(p["event_type"], "decision");
        assert_eq!(p["summary"], "payload summary");
        assert!(p["created_at"].is_string());
    }

    #[test]
    fn to_memory_copies_fields_correctly() {
        let mut e = event("mem summary");
        e.evidence = Some("proof".to_string());
        let m = e.to_memory();
        assert_eq!(m.event_type, "decision");
        assert_eq!(m.summary, "mem summary");
        assert_eq!(m.evidence, Some("proof".to_string()));
    }

    #[test]
    fn error_record_frequency_field_accessible() {
        let rec = crate::state::ErrorRecord {
            id: "id".to_string(),
            repo: "r".to_string(),
            task: "t".to_string(),
            error_type: "e".to_string(),
            description: "d".to_string(),
            severity: "medium".to_string(),
            frequency: 3,
            last_seen: Utc::now(),
        };
        assert_eq!(rec.frequency, 3);
    }

    #[test]
    fn summary_level_filters_correctly() {
        assert_eq!(preferred_summary_levels(0), vec![0]);
        assert_eq!(preferred_summary_levels(19), vec![0]);
        assert_eq!(preferred_summary_levels(20), vec![1]);
        assert_eq!(preferred_summary_levels(199), vec![1]);
        assert_eq!(preferred_summary_levels(200), vec![2]);
        assert_eq!(preferred_summary_levels(1999), vec![2]);
        assert_eq!(preferred_summary_levels(2000), vec![3, 2]);
    }
}
