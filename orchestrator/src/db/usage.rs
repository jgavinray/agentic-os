use deadpool_postgres::Pool;

use crate::db_types::{VllmCacheObservationInput, VllmCacheStats};

pub async fn record_token_usage(
    pool: &Pool,
    requested_model: &str,
    actual_model: &str,
    namespace: &str,
    repo: &str,
    usage: &crate::state::TokenUsage,
) -> Result<(), anyhow::Error> {
    let started = std::time::Instant::now();
    let result: Result<(), anyhow::Error> = async {
        let conn = pool.get().await?;
        conn.execute(
            "INSERT INTO token_usage \
             (requested_model, actual_model, namespace, repo, processed_tokens, cached_tokens, generated_tokens) \
             VALUES ($1, $2, $3, $4, $5, $6, $7)",
            &[
                &requested_model,
                &actual_model,
                &namespace,
                &repo,
                &(usage.processed_tokens as i64),
                &(usage.cached_tokens as i64),
                &(usage.generated_tokens as i64),
            ],
        )
        .await?;
        Ok(())
    }
    .await;
    crate::telemetry::record_db_query("record_token_usage", started.elapsed(), result.is_ok());
    result
}

pub async fn insert_vllm_cache_observation(
    pool: &Pool,
    observation: &VllmCacheObservationInput,
) -> Result<(), anyhow::Error> {
    let started = std::time::Instant::now();
    let result: Result<(), anyhow::Error> = async {
        let conn = pool.get().await?;
        conn.execute(
            "INSERT INTO vllm_cache_observations
             (session_id, namespace, repo, task, endpoint, requested_model, routed_model,
              request_event_id, context_pack_id, attempt_id, metrics_url,
              prefix_cache_queries_delta, prefix_cache_hits_delta,
              prompt_tokens_total_delta, prompt_tokens_cached_delta,
              prompt_tokens_local_compute_delta, prompt_tokens_local_cache_hit_delta,
              prompt_tokens_external_kv_delta, request_input_tokens, request_output_tokens,
              provider_cached_tokens, provider_cache_created_tokens, provider_cache_read_tokens,
              kv_cache_usage_before, kv_cache_usage_after)
             VALUES
             ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11,
              $12, $13, $14, $15, $16, $17, $18, $19, $20,
              $21, $22, $23, $24, $25)",
            &[
                &observation.session_id,
                &observation.namespace,
                &observation.repo,
                &observation.task,
                &observation.endpoint,
                &observation.requested_model,
                &observation.routed_model,
                &observation.request_event_id,
                &observation.context_pack_id,
                &observation.attempt_id,
                &observation.metrics_url,
                &observation.delta.prefix_cache_queries_delta,
                &observation.delta.prefix_cache_hits_delta,
                &observation.delta.prompt_tokens_total_delta,
                &observation.delta.prompt_tokens_cached_delta,
                &observation.delta.prompt_tokens_local_compute_delta,
                &observation.delta.prompt_tokens_local_cache_hit_delta,
                &observation.delta.prompt_tokens_external_kv_delta,
                &observation.request_input_tokens,
                &observation.request_output_tokens,
                &observation.provider_cache.provider_cached_tokens,
                &observation.provider_cache.provider_cache_created_tokens,
                &observation.provider_cache.provider_cache_read_tokens,
                &observation.delta.kv_cache_usage_before,
                &observation.delta.kv_cache_usage_after,
            ],
        )
        .await?;
        Ok(())
    }
    .await;
    crate::telemetry::record_db_query(
        "insert_vllm_cache_observation",
        started.elapsed(),
        result.is_ok(),
    );
    result
}

pub async fn get_vllm_cache_stats(
    pool: &Pool,
    repo: Option<&str>,
    session_id: Option<&str>,
) -> Result<VllmCacheStats, anyhow::Error> {
    let started = std::time::Instant::now();
    let result: Result<VllmCacheStats, anyhow::Error> = async {
        let conn = pool.get().await?;
        let row = conn
            .query_one(
                "SELECT
                    count(*)::BIGINT AS observations,
                    coalesce(sum(prefix_cache_queries_delta), 0)::BIGINT AS prefix_cache_queries,
                    coalesce(sum(prefix_cache_hits_delta), 0)::BIGINT AS prefix_cache_hits,
                    coalesce(sum(prompt_tokens_total_delta), 0)::BIGINT AS prompt_tokens_total,
                    coalesce(sum(prompt_tokens_cached_delta), 0)::BIGINT AS prompt_tokens_cached,
                    coalesce(sum(prompt_tokens_local_compute_delta), 0)::BIGINT AS prompt_tokens_local_compute,
                    coalesce(sum(prompt_tokens_local_cache_hit_delta), 0)::BIGINT AS prompt_tokens_local_cache_hit,
                    coalesce(sum(prompt_tokens_external_kv_delta), 0)::BIGINT AS prompt_tokens_external_kv,
                    coalesce(sum(request_input_tokens), 0)::BIGINT AS request_input_tokens,
                    coalesce(sum(request_output_tokens), 0)::BIGINT AS request_output_tokens,
                    coalesce(sum(provider_cached_tokens), 0)::BIGINT AS provider_cached_tokens,
                    coalesce(sum(provider_cache_created_tokens), 0)::BIGINT AS provider_cache_created_tokens,
                    coalesce(sum(provider_cache_read_tokens), 0)::BIGINT AS provider_cache_read_tokens
                 FROM vllm_cache_observations
                 WHERE ($1::TEXT IS NULL OR repo = $1)
                   AND ($2::TEXT IS NULL OR session_id = $2)",
                &[&repo, &session_id],
            )
            .await?;
        let prefix_cache_queries = row.get::<_, i64>("prefix_cache_queries");
        let prefix_cache_hits = row.get::<_, i64>("prefix_cache_hits");
        let prompt_tokens_total = row.get::<_, i64>("prompt_tokens_total");
        let prompt_tokens_cached = row.get::<_, i64>("prompt_tokens_cached");
        Ok(VllmCacheStats {
            observations: row.get("observations"),
            prefix_cache_queries,
            prefix_cache_hits,
            prompt_tokens_total,
            prompt_tokens_cached,
            prompt_tokens_local_compute: row.get("prompt_tokens_local_compute"),
            prompt_tokens_local_cache_hit: row.get("prompt_tokens_local_cache_hit"),
            prompt_tokens_external_kv: row.get("prompt_tokens_external_kv"),
            request_input_tokens: row.get("request_input_tokens"),
            request_output_tokens: row.get("request_output_tokens"),
            provider_cached_tokens: row.get("provider_cached_tokens"),
            provider_cache_created_tokens: row.get("provider_cache_created_tokens"),
            provider_cache_read_tokens: row.get("provider_cache_read_tokens"),
            prefix_cache_hit_rate: ratio(prefix_cache_hits, prefix_cache_queries),
            prompt_cached_rate: ratio(prompt_tokens_cached, prompt_tokens_total),
        })
    }
    .await;
    crate::telemetry::record_db_query("get_vllm_cache_stats", started.elapsed(), result.is_ok());
    result
}

fn ratio(numerator: i64, denominator: i64) -> f64 {
    if denominator <= 0 {
        0.0
    } else {
        numerator as f64 / denominator as f64
    }
}
