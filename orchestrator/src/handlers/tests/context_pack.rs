use super::*;

#[test]
fn retry_backoff_durations_are_correct() {
    let delays: Vec<u64> = (0u32..2).map(|a| 200 * 2u64.pow(a)).collect();
    assert_eq!(delays, vec![200, 400]);
}

#[test]
fn metrics_include_context_pack_counts() {
    let metrics = telemetry::MetricsRegistry::new();
    telemetry::record_context_pack(
        &metrics,
        &ContextPackStats {
            context_chars: 400,
            context_tokens_estimate: 100,
            l0_items_injected: 1,
            l1_items_injected: 2,
            l2_items_injected: 3,
            l3_items_injected: 4,
            failed_attempts_injected: 1,
            remediations_injected: 1,
            retrieval_semantic_hits: 5,
            retrieval_fts_hits: 6,
            retrieval_deduped_hits: 7,
            cache_hit: false,
            ..Default::default()
        },
    );
    telemetry::record_tokens(
        &metrics,
        &TokenUsage {
            processed_tokens: 120,
            cached_tokens: 80,
            generated_tokens: 40,
        },
        "test-model",
    );
    telemetry::record_promotion(&metrics, false, false);
    telemetry::record_promotion(&metrics, true, true);

    let snapshot = metrics.snapshot();
    assert_eq!(snapshot.context_pack_requests, 1);
    assert_eq!(snapshot.context_cache_misses, 1);
    assert_eq!(snapshot.l3_items_injected, 4);
    assert_eq!(snapshot.retrieval_deduped_hits, 7);
    assert_eq!(snapshot.processed_tokens, 120);
    assert_eq!(snapshot.cached_tokens, 80);
    assert_eq!(snapshot.generated_tokens, 40);
    assert_eq!(snapshot.promotion_attempts, 2);
    assert_eq!(snapshot.promotion_accepted, 1);
    assert_eq!(snapshot.promotion_rejected, 1);
    assert_eq!(snapshot.memory_source_coverage, 0.5);
}

#[test]
fn context_pack_sections_are_stable() {
    let mut evidence = db::ContextEvidence::default();
    evidence.l3_project = vec![test_event("summary", "L3 architecture truth", 3)];
    evidence.l2_repo = vec![test_event("summary", "L2 repo pattern", 2)];
    evidence.l1_matching = vec![test_event("summary", "L1 task summary", 1)];
    evidence.l0_recent = vec![test_event("edit", "L0 raw evidence", 0)];
    evidence.failures = vec![test_event("failed_attempt", "do not repeat this", 0)];
    let policy = ContextPolicy::for_category(TaskCategory::Architecture);

    let (context, _stats) = db::build_layered_context(
        "repo",
        "architecture task",
        &evidence,
        &[],
        &[],
        &policy,
        16_000,
        4000,
    );

    let l3 = context.find("== Durable Project Memory ==").unwrap();
    let l2 = context.find("== Repo Patterns and Decisions ==").unwrap();
    let l1 = context.find("== Relevant Session Summaries ==").unwrap();
    let l0 = context.find("== Recent Evidence ==").unwrap();
    let failures = context
        .find("== Failed Attempts and Remediations ==")
        .unwrap();

    assert!(l3 < l2);
    assert!(l2 < l1);
    assert!(l1 < l0);
    assert!(l0 < failures);
}

#[test]
fn failed_attempts_are_injected_for_debug_tasks() {
    let mut evidence = db::ContextEvidence::default();
    evidence.failures = vec![db::AgentEvent {
        evidence: Some("the old retry loop timed out".to_string()),
        metadata: json!({"outcome": "bounded retry fixed the lag"}),
        ..test_event("failed_attempt", "unbounded retries caused lag", 0)
    }];
    let policy = ContextPolicy::for_category(TaskCategory::Narrow);

    let (context, stats) = db::build_layered_context(
        "repo",
        "debug lag",
        &evidence,
        &[],
        &[],
        &policy,
        8000,
        4000,
    );

    assert!(context.contains("unbounded retries caused lag"));
    assert!(context.contains("Evidence: the old retry loop timed out"));
    assert!(context.contains("Outcome: bounded retry fixed the lag"));
    assert_eq!(stats.failed_attempts_injected, 1);
}

#[test]
fn cache_hit_rate_updates() {
    let metrics = telemetry::MetricsRegistry::new();
    telemetry::record_context_pack(
        &metrics,
        &ContextPackStats {
            cache_hit: false,
            ..Default::default()
        },
    );
    telemetry::record_context_pack(
        &metrics,
        &ContextPackStats {
            cache_hit: true,
            ..Default::default()
        },
    );

    let snapshot = metrics.snapshot();
    assert_eq!(snapshot.context_pack_requests, 2);
    assert_eq!(snapshot.context_cache_misses, 1);
    assert_eq!(snapshot.context_cache_hits, 1);
}
