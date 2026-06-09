use crate::telemetry::MetricsRegistry;
use metrics::{counter, gauge, histogram};

pub fn record_context_pack(registry: &MetricsRegistry, stats: &crate::state::ContextPackStats) {
    counter!("context_pack_requests_total").increment(1);
    if stats.cache_hit {
        counter!("context_pack_cache_hits_total").increment(1);
    } else {
        counter!("context_pack_cache_misses_total").increment(1);
    }
    histogram!("context_pack_build_duration_seconds").record(stats.build_ms as f64 / 1000.0);
    histogram!("context_pack_tokens_estimate").record(stats.context_tokens_estimate as f64);
    record_items("l0", stats.l0_items_injected);
    record_items("l1", stats.l1_items_injected);
    record_items("l2", stats.l2_items_injected);
    record_items("l3", stats.l3_items_injected);
    record_items("failed_attempt", stats.failed_attempts_injected);
    record_items("remediation", stats.remediations_injected);
    record_items("failure_history", stats.failure_history_items_injected);
    record_items(
        "operational_constraints",
        stats.operational_constraints_injected,
    );
    for signature in &stats.failure_history_remediation_signatures {
        counter!(
            "remediation_reuse_total",
            "signature" => crate::execution_feedback::bounded_failure_signature_label(signature)
        )
        .increment(1);
    }
    counter!("retrieval_hits_total", "source" => "semantic")
        .increment(stats.retrieval_semantic_hits as u64);
    counter!("retrieval_hits_total", "source" => "fts").increment(stats.retrieval_fts_hits as u64);
    counter!("retrieval_hits_total", "source" => "deduped")
        .increment(stats.retrieval_deduped_hits as u64);

    registry.with_snapshot_mut(|metrics| {
        metrics.context_pack_requests += 1;
        if stats.cache_hit {
            metrics.context_cache_hits += 1;
        } else {
            metrics.context_cache_misses += 1;
        }
        metrics.context_pack_build_ms_total += stats.build_ms;
        metrics.context_pack_chars_total += stats.context_chars as u64;
        metrics.context_pack_tokens_estimate_total += stats.context_tokens_estimate as u64;
        metrics.l0_items_injected += stats.l0_items_injected as u64;
        metrics.l1_items_injected += stats.l1_items_injected as u64;
        metrics.l2_items_injected += stats.l2_items_injected as u64;
        metrics.l3_items_injected += stats.l3_items_injected as u64;
        metrics.failed_attempts_injected += stats.failed_attempts_injected as u64;
        metrics.remediations_injected += stats.remediations_injected as u64;
        metrics.failure_history_items_injected += stats.failure_history_items_injected as u64;
        metrics.operational_constraints_injected += stats.operational_constraints_injected as u64;
        metrics.remediation_reuse += stats.failure_history_remediation_signatures.len() as u64;
        metrics.retrieval_semantic_hits += stats.retrieval_semantic_hits as u64;
        metrics.retrieval_fts_hits += stats.retrieval_fts_hits as u64;
        metrics.retrieval_deduped_hits += stats.retrieval_deduped_hits as u64;
    });
}

pub fn record_cache_invalidation(registry: &MetricsRegistry) {
    counter!("context_cache_stale_invalidations_total").increment(1);
    registry.with_snapshot_mut(|metrics| {
        metrics.stale_cache_invalidations += 1;
    });
}

pub fn record_context_cache_replacement(registry: &MetricsRegistry, replaced: usize) {
    counter!("context_cache_replacements_total").increment(replaced as u64);
    registry.with_snapshot_mut(|metrics| {
        metrics.context_cache_replacements += replaced as u64;
    });
}

pub fn record_promotion(registry: &MetricsRegistry, accepted: bool, has_sources: bool) {
    let result = if accepted { "accepted" } else { "rejected" };
    counter!("memory_promotions_total", "result" => result).increment(1);

    registry.with_snapshot_mut(|metrics| {
        metrics.promotion_attempts += 1;
        if accepted {
            metrics.promotion_accepted += 1;
        } else {
            metrics.promotion_rejected += 1;
        }
        metrics.memory_source_items += 1;
        if has_sources {
            metrics.memory_source_items_with_sources += 1;
        }
        metrics.memory_source_coverage =
            metrics.memory_source_items_with_sources as f64 / metrics.memory_source_items as f64;
        gauge!("memory_source_coverage").set(metrics.memory_source_coverage);
    });
}

fn record_items(layer: &'static str, count: usize) {
    counter!("context_pack_items_injected_total", "layer" => layer).increment(count as u64);
}
