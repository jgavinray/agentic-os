use axum::extract::MatchedPath;
use axum::extract::Request;
use axum::http::{header, HeaderMap};
use axum::middleware::Next;
use axum::response::Response;
use deadpool_postgres::Pool;
use metrics::{counter, describe_counter, describe_gauge, describe_histogram, gauge, histogram};
use metrics_exporter_prometheus::{Matcher, PrometheusBuilder, PrometheusHandle};
use serde::Serialize;
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

const HTTP_BUCKETS: &[f64] = &[
    0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0,
];
const UPSTREAM_BUCKETS: &[f64] = &[
    0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0, 600.0,
];
const IO_BUCKETS: &[f64] = &[
    0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0,
];
const EMBEDDER_BUCKETS: &[f64] = &[
    0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0,
];
const TOKEN_BUCKETS: &[f64] = &[
    16.0, 32.0, 64.0, 128.0, 256.0, 512.0, 1024.0, 2048.0, 4096.0, 8192.0, 16384.0, 32768.0,
];

#[derive(Debug, Clone, Serialize, Default)]
pub struct MetricsSnapshot {
    pub context_pack_requests: u64,
    pub context_cache_hits: u64,
    pub context_cache_misses: u64,
    pub context_pack_build_ms_total: u64,
    pub context_pack_chars_total: u64,
    pub context_pack_tokens_estimate_total: u64,
    pub l0_items_injected: u64,
    pub l1_items_injected: u64,
    pub l2_items_injected: u64,
    pub l3_items_injected: u64,
    pub failed_attempts_injected: u64,
    pub remediations_injected: u64,
    pub retrieval_semantic_hits: u64,
    pub retrieval_fts_hits: u64,
    pub retrieval_deduped_hits: u64,
    pub processed_tokens: u64,
    pub cached_tokens: u64,
    pub generated_tokens: u64,
    pub promotion_attempts: u64,
    pub promotion_accepted: u64,
    pub promotion_rejected: u64,
    pub memory_source_items: u64,
    pub memory_source_items_with_sources: u64,
    pub memory_source_coverage: f64,
    pub stale_cache_invalidations: u64,
}

#[derive(Clone)]
pub struct MetricsRegistry {
    inner: Arc<RwLock<MetricsSnapshot>>,
}

impl MetricsRegistry {
    pub fn new() -> Self {
        Self {
            inner: Arc::new(RwLock::new(MetricsSnapshot::default())),
        }
    }

    pub fn snapshot(&self) -> MetricsSnapshot {
        self.inner.read().unwrap().clone()
    }
}

impl Default for MetricsRegistry {
    fn default() -> Self {
        Self::new()
    }
}

pub fn install_recorder() -> Result<PrometheusHandle, anyhow::Error> {
    let handle = PrometheusBuilder::new()
        .set_buckets_for_metric(
            Matcher::Full("http_request_duration_seconds".to_string()),
            HTTP_BUCKETS,
        )?
        .set_buckets_for_metric(
            Matcher::Full("upstream_litellm_duration_seconds".to_string()),
            UPSTREAM_BUCKETS,
        )?
        .set_buckets_for_metric(
            Matcher::Full("stream_first_token_seconds".to_string()),
            UPSTREAM_BUCKETS,
        )?
        .set_buckets_for_metric(
            Matcher::Full("stream_duration_seconds".to_string()),
            UPSTREAM_BUCKETS,
        )?
        .set_buckets_for_metric(
            Matcher::Full("db_query_duration_seconds".to_string()),
            IO_BUCKETS,
        )?
        .set_buckets_for_metric(
            Matcher::Full("qdrant_request_duration_seconds".to_string()),
            IO_BUCKETS,
        )?
        .set_buckets_for_metric(
            Matcher::Full("embedder_inference_duration_seconds".to_string()),
            EMBEDDER_BUCKETS,
        )?
        .set_buckets_for_metric(
            Matcher::Full("embedder_input_tokens".to_string()),
            TOKEN_BUCKETS,
        )?
        .set_buckets_for_metric(
            Matcher::Full("sentiment_inference_duration_seconds".to_string()),
            EMBEDDER_BUCKETS,
        )?
        .set_buckets_for_metric(
            Matcher::Full("summarizer_duration_seconds".to_string()),
            UPSTREAM_BUCKETS,
        )?
        .set_buckets_for_metric(
            Matcher::Full("context_pack_build_duration_seconds".to_string()),
            HTTP_BUCKETS,
        )?
        .set_buckets_for_metric(
            Matcher::Full("context_pack_tokens_estimate".to_string()),
            TOKEN_BUCKETS,
        )?
        .install_recorder()?;
    describe_metrics();
    Ok(handle)
}

fn describe_metrics() {
    describe_counter!(
        "http_requests_total",
        "HTTP requests served by the orchestrator."
    );
    describe_histogram!(
        "http_request_duration_seconds",
        "HTTP request latency in seconds."
    );
    describe_gauge!(
        "http_requests_in_flight",
        "HTTP requests currently in flight."
    );
    describe_counter!("auth_attempts_total", "API authentication attempts.");
    describe_counter!(
        "upstream_litellm_requests_total",
        "Requests sent from the orchestrator to LiteLLM."
    );
    describe_histogram!(
        "upstream_litellm_duration_seconds",
        "LiteLLM upstream request latency in seconds."
    );
    describe_counter!(
        "upstream_litellm_errors_total",
        "LiteLLM upstream request errors by path and kind."
    );
    describe_histogram!(
        "stream_first_token_seconds",
        "Seconds from request handling to first streamed byte."
    );
    describe_histogram!(
        "stream_duration_seconds",
        "Total stream duration in seconds."
    );
    describe_counter!(
        "stream_disconnects_total",
        "Streams interrupted before completion."
    );
    describe_gauge!("db_pool_size", "Configured Postgres pool capacity.");
    describe_gauge!(
        "db_pool_available",
        "Postgres pool connections currently available."
    );
    describe_gauge!(
        "db_pool_waiters",
        "Tasks waiting for a Postgres pool connection."
    );
    describe_histogram!(
        "db_query_duration_seconds",
        "Postgres query latency in seconds."
    );
    describe_counter!("db_query_errors_total", "Postgres query failures.");
    describe_counter!("qdrant_requests_total", "Qdrant API requests.");
    describe_histogram!(
        "qdrant_request_duration_seconds",
        "Qdrant API latency in seconds."
    );
    describe_counter!(
        "embedder_inferences_total",
        "Embedding model inference calls."
    );
    describe_histogram!(
        "embedder_inference_duration_seconds",
        "Embedding model inference latency in seconds."
    );
    describe_histogram!("embedder_input_tokens", "Tokenized embedder input length.");
    describe_counter!(
        "sentiment_inferences_total",
        "Sentiment classifier inferences by verdict."
    );
    describe_histogram!(
        "sentiment_inference_duration_seconds",
        "Sentiment classifier inference latency in seconds."
    );
    describe_counter!("summarizer_ticks_total", "Summarizer loop ticks.");
    describe_counter!(
        "summarizer_candidates_found_total",
        "Summarizer candidates found by target level."
    );
    describe_counter!(
        "summarizer_summaries_written_total",
        "Summarizer write attempts by target level and result."
    );
    describe_histogram!(
        "summarizer_duration_seconds",
        "Summarizer pass latency in seconds."
    );
    describe_counter!(
        "context_pack_requests_total",
        "Context pack build requests."
    );
    describe_counter!("context_pack_cache_hits_total", "Context pack cache hits.");
    describe_counter!(
        "context_pack_cache_misses_total",
        "Context pack cache misses."
    );
    describe_histogram!(
        "context_pack_build_duration_seconds",
        "Context pack build latency in seconds."
    );
    describe_histogram!(
        "context_pack_tokens_estimate",
        "Estimated context pack token count."
    );
    describe_counter!(
        "context_pack_items_injected_total",
        "Context pack items injected by memory layer."
    );
    describe_counter!("retrieval_hits_total", "Retrieval hits by source.");
    describe_counter!(
        "inference_tokens_total",
        "Inference tokens by kind and model."
    );
    describe_counter!(
        "context_cache_stale_invalidations_total",
        "Context cache invalidations triggered by new memory."
    );
    describe_counter!("memory_promotions_total", "Memory promotion decisions.");
    describe_gauge!(
        "memory_source_coverage",
        "Fraction of promoted memory with sources."
    );
    describe_counter!(
        "rate_limited_total",
        "Requests rejected by the per-key rate limiter."
    );
    describe_counter!(
        "process_cpu_seconds_total",
        "CPU seconds consumed by this process."
    );
    describe_gauge!(
        "process_resident_memory_bytes",
        "Resident memory used by this process."
    );
    describe_gauge!(
        "process_start_time_seconds",
        "Process start time since Unix epoch."
    );
    describe_gauge!(
        "process_open_fds",
        "Open file descriptors for this process."
    );
}

pub fn prime_metrics(registry: &MetricsRegistry, default_model: &str, sentiment_loaded: bool) {
    for result in ["accepted", "rejected"] {
        counter!("auth_attempts_total", "result" => result).increment(0);
        counter!("memory_promotions_total", "result" => result).increment(0);
    }
    for endpoint in [
        "/v1/chat/completions",
        "/v1/messages",
        "/v1/models",
        "/sessions/start",
        "/events/append",
        "/context/pack",
        "/search",
        "/metrics",
    ] {
        gauge!("http_requests_in_flight", "endpoint" => endpoint).set(0.0);
    }
    for path in [
        "chat_completions",
        "messages",
        "models",
        "summarizer_chat_completions",
    ] {
        counter!("upstream_litellm_requests_total", "path" => path, "status" => "none")
            .increment(0);
        histogram!("upstream_litellm_duration_seconds", "path" => path).record(0.0);
        for kind in ["timeout", "connection", "4xx", "5xx", "parse"] {
            counter!("upstream_litellm_errors_total", "path" => path, "kind" => kind).increment(0);
        }
    }
    for path in ["chat_completions", "messages"] {
        histogram!("stream_first_token_seconds", "path" => path).record(0.0);
        histogram!("stream_duration_seconds", "path" => path).record(0.0);
        counter!("stream_disconnects_total", "path" => path, "reason" => "client_disconnect")
            .increment(0);
    }
    for op in [
        "create_session",
        "find_or_create_session",
        "insert_event",
        "count_events_for_repo",
        "get_context_evidence",
        "search_events_fts",
        "hydrate_active_search_hits",
        "insert_error_record",
        "get_active_errors",
        "record_token_usage",
        "check_ready",
    ] {
        histogram!("db_query_duration_seconds", "op" => op).record(0.0);
        counter!("db_query_errors_total", "op" => op).increment(0);
    }
    for op in ["upsert", "search", "health", "create_collection"] {
        counter!("qdrant_requests_total", "op" => op, "status" => "none").increment(0);
        histogram!("qdrant_request_duration_seconds", "op" => op).record(0.0);
    }
    counter!("embedder_inferences_total").increment(0);
    histogram!("embedder_inference_duration_seconds").record(0.0);
    histogram!("embedder_input_tokens").record(0.0);
    if sentiment_loaded {
        for verdict in ["negative", "non_negative"] {
            counter!("sentiment_inferences_total", "verdict" => verdict).increment(0);
        }
        histogram!("sentiment_inference_duration_seconds").record(0.0);
    }
    for level in ["1", "2", "3"] {
        counter!("summarizer_candidates_found_total", "target_level" => level).increment(0);
        histogram!("summarizer_duration_seconds", "target_level" => level).record(0.0);
        for result in ["success", "failure"] {
            counter!(
                "summarizer_summaries_written_total",
                "target_level" => level,
                "result" => result
            )
            .increment(0);
        }
    }
    counter!("summarizer_ticks_total").increment(0);
    counter!("context_pack_requests_total").increment(0);
    counter!("context_pack_cache_hits_total").increment(0);
    counter!("context_pack_cache_misses_total").increment(0);
    histogram!("context_pack_build_duration_seconds").record(0.0);
    histogram!("context_pack_tokens_estimate").record(0.0);
    for layer in ["l0", "l1", "l2", "l3", "failed_attempt", "remediation"] {
        counter!("context_pack_items_injected_total", "layer" => layer).increment(0);
    }
    for source in ["semantic", "fts", "deduped"] {
        counter!("retrieval_hits_total", "source" => source).increment(0);
    }
    for kind in ["processed", "cached", "generated"] {
        counter!("inference_tokens_total", "kind" => kind, "model" => default_model.to_string())
            .increment(0);
    }
    counter!("context_cache_stale_invalidations_total").increment(0);
    gauge!("memory_source_coverage").set(registry.snapshot().memory_source_coverage);
}

pub async fn http_metrics_middleware(req: Request, next: Next) -> Response {
    let endpoint = req
        .extensions()
        .get::<MatchedPath>()
        .map(|path| path.as_str().to_string())
        .unwrap_or_else(|| req.uri().path().to_string());
    let method = req.method().as_str().to_string();
    let started = Instant::now();

    gauge!("http_requests_in_flight", "endpoint" => endpoint.clone()).increment(1.0);
    let response = next.run(req).await;
    gauge!("http_requests_in_flight", "endpoint" => endpoint.clone()).decrement(1.0);

    let status = response.status().as_u16().to_string();
    counter!(
        "http_requests_total",
        "endpoint" => endpoint.clone(),
        "method" => method.clone(),
        "status" => status
    )
    .increment(1);
    histogram!(
        "http_request_duration_seconds",
        "endpoint" => endpoint,
        "method" => method
    )
    .record(started.elapsed().as_secs_f64());

    response
}

pub fn prometheus_content_type() -> HeaderMap {
    let mut headers = HeaderMap::new();
    headers.insert(
        header::CONTENT_TYPE,
        "text/plain; version=0.0.4".parse().unwrap(),
    );
    headers
}

pub fn record_auth_attempt(accepted: bool) {
    let result = if accepted { "accepted" } else { "rejected" };
    counter!("auth_attempts_total", "result" => result).increment(1);
}

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
    counter!("retrieval_hits_total", "source" => "semantic")
        .increment(stats.retrieval_semantic_hits as u64);
    counter!("retrieval_hits_total", "source" => "fts").increment(stats.retrieval_fts_hits as u64);
    counter!("retrieval_hits_total", "source" => "deduped")
        .increment(stats.retrieval_deduped_hits as u64);

    let mut metrics = registry.inner.write().unwrap();
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
    metrics.retrieval_semantic_hits += stats.retrieval_semantic_hits as u64;
    metrics.retrieval_fts_hits += stats.retrieval_fts_hits as u64;
    metrics.retrieval_deduped_hits += stats.retrieval_deduped_hits as u64;
}

fn record_items(layer: &'static str, count: usize) {
    counter!("context_pack_items_injected_total", "layer" => layer).increment(count as u64);
}

pub fn record_tokens(registry: &MetricsRegistry, usage: &crate::state::TokenUsage, model: &str) {
    if usage.is_empty() {
        return;
    }
    counter!("inference_tokens_total", "kind" => "processed", "model" => model.to_string())
        .increment(usage.processed_tokens);
    counter!("inference_tokens_total", "kind" => "cached", "model" => model.to_string())
        .increment(usage.cached_tokens);
    counter!("inference_tokens_total", "kind" => "generated", "model" => model.to_string())
        .increment(usage.generated_tokens);

    let mut metrics = registry.inner.write().unwrap();
    metrics.processed_tokens += usage.processed_tokens;
    metrics.cached_tokens += usage.cached_tokens;
    metrics.generated_tokens += usage.generated_tokens;
}

pub fn record_cache_invalidation(registry: &MetricsRegistry) {
    counter!("context_cache_stale_invalidations_total").increment(1);
    registry.inner.write().unwrap().stale_cache_invalidations += 1;
}

pub fn record_promotion(registry: &MetricsRegistry, accepted: bool, has_sources: bool) {
    let result = if accepted { "accepted" } else { "rejected" };
    counter!("memory_promotions_total", "result" => result).increment(1);

    let mut metrics = registry.inner.write().unwrap();
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
}

pub fn record_rate_limited(key_hash: &str) {
    counter!("rate_limited_total", "key_hash" => key_hash.to_string()).increment(1);
}

pub fn record_db_query(op: &'static str, elapsed: Duration, success: bool) {
    histogram!("db_query_duration_seconds", "op" => op).record(elapsed.as_secs_f64());
    if !success {
        counter!("db_query_errors_total", "op" => op).increment(1);
    }
}

pub fn record_qdrant_request(op: &'static str, elapsed: Duration, status: &str) {
    histogram!("qdrant_request_duration_seconds", "op" => op).record(elapsed.as_secs_f64());
    counter!("qdrant_requests_total", "op" => op, "status" => status.to_string()).increment(1);
}

pub fn record_upstream_litellm(path: &'static str, elapsed: Duration, status: &str) {
    histogram!("upstream_litellm_duration_seconds", "path" => path).record(elapsed.as_secs_f64());
    counter!(
        "upstream_litellm_requests_total",
        "path" => path,
        "status" => status.to_string()
    )
    .increment(1);
}

pub fn record_upstream_litellm_error(path: &'static str, kind: &'static str) {
    counter!("upstream_litellm_errors_total", "path" => path, "kind" => kind).increment(1);
}

pub fn upstream_error_kind(status: reqwest::StatusCode) -> &'static str {
    if status.is_client_error() {
        "4xx"
    } else if status.is_server_error() {
        "5xx"
    } else {
        "parse"
    }
}

pub fn reqwest_error_kind(err: &reqwest::Error) -> &'static str {
    if err.is_timeout() {
        "timeout"
    } else if err.is_connect() {
        "connection"
    } else if err.is_status() {
        "5xx"
    } else {
        "parse"
    }
}

pub fn record_embedder_input_tokens(tokens: usize) {
    histogram!("embedder_input_tokens").record(tokens as f64);
}

pub fn record_embedder_inference(elapsed: Duration, success: bool) {
    if success {
        counter!("embedder_inferences_total").increment(1);
    }
    histogram!("embedder_inference_duration_seconds").record(elapsed.as_secs_f64());
}

pub fn record_sentiment(verdict: bool, elapsed: Duration) {
    let verdict = if verdict { "negative" } else { "non_negative" };
    counter!("sentiment_inferences_total", "verdict" => verdict).increment(1);
    histogram!("sentiment_inference_duration_seconds").record(elapsed.as_secs_f64());
}

pub fn record_summarizer_tick() {
    counter!("summarizer_ticks_total").increment(1);
}

pub fn record_summarizer_candidate(target_level: i32, count: usize) {
    counter!(
        "summarizer_candidates_found_total",
        "target_level" => target_level.to_string()
    )
    .increment(count as u64);
}

pub fn record_summarizer_duration(target_level: i32, elapsed: Duration) {
    histogram!(
        "summarizer_duration_seconds",
        "target_level" => target_level.to_string()
    )
    .record(elapsed.as_secs_f64());
}

pub fn record_summarizer_written(target_level: i32, success: bool) {
    let result = if success { "success" } else { "failure" };
    counter!(
        "summarizer_summaries_written_total",
        "target_level" => target_level.to_string(),
        "result" => result
    )
    .increment(1);
}

pub struct StreamTracker {
    path: &'static str,
    started: Instant,
    first_token_seen: bool,
    completed: bool,
}

impl StreamTracker {
    pub fn new(path: &'static str, started: Instant) -> Self {
        Self {
            path,
            started,
            first_token_seen: false,
            completed: false,
        }
    }

    pub fn first_token(&mut self) {
        if !self.first_token_seen {
            histogram!("stream_first_token_seconds", "path" => self.path)
                .record(self.started.elapsed().as_secs_f64());
            self.first_token_seen = true;
        }
    }

    pub fn finish(&mut self) {
        self.completed = true;
        histogram!("stream_duration_seconds", "path" => self.path)
            .record(self.started.elapsed().as_secs_f64());
    }

    pub fn fail(&mut self, reason: &'static str) {
        self.completed = true;
        histogram!("stream_duration_seconds", "path" => self.path)
            .record(self.started.elapsed().as_secs_f64());
        counter!("stream_disconnects_total", "path" => self.path, "reason" => reason).increment(1);
    }
}

impl Drop for StreamTracker {
    fn drop(&mut self) {
        if !self.completed {
            histogram!("stream_duration_seconds", "path" => self.path)
                .record(self.started.elapsed().as_secs_f64());
            counter!(
                "stream_disconnects_total",
                "path" => self.path,
                "reason" => "client_disconnect"
            )
            .increment(1);
        }
    }
}

pub fn record_pool_gauges(pool: &Pool) {
    let status = pool.status();
    gauge!("db_pool_size").set(status.max_size as f64);
    gauge!("db_pool_available").set(status.available as f64);
    gauge!("db_pool_waiters").set(status.waiting as f64);
}

pub fn record_process_metrics() {
    let metrics = process_metrics();
    counter!("process_cpu_seconds_total").absolute(metrics.cpu_seconds.floor() as u64);
    gauge!("process_resident_memory_bytes").set(metrics.resident_memory_bytes as f64);
    gauge!("process_start_time_seconds").set(metrics.start_time_seconds);
    gauge!("process_open_fds").set(metrics.open_fds as f64);
}

#[derive(Default)]
struct ProcessMetrics {
    cpu_seconds: f64,
    resident_memory_bytes: u64,
    start_time_seconds: f64,
    open_fds: u64,
}

fn process_metrics() -> ProcessMetrics {
    #[cfg(target_os = "linux")]
    {
        linux_process_metrics()
    }
    #[cfg(not(target_os = "linux"))]
    {
        ProcessMetrics {
            start_time_seconds: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|d| d.as_secs_f64())
                .unwrap_or_default(),
            ..Default::default()
        }
    }
}

#[cfg(target_os = "linux")]
fn linux_process_metrics() -> ProcessMetrics {
    const CLOCK_TICKS_PER_SECOND: f64 = 100.0;
    const PAGE_SIZE_BYTES: u64 = 4096;

    let stat = std::fs::read_to_string("/proc/self/stat").unwrap_or_default();
    let after_comm = stat.rsplit_once(") ").map(|(_, rest)| rest).unwrap_or("");
    let fields: Vec<&str> = after_comm.split_whitespace().collect();
    let utime = fields
        .get(11)
        .and_then(|v| v.parse::<f64>().ok())
        .unwrap_or_default();
    let stime = fields
        .get(12)
        .and_then(|v| v.parse::<f64>().ok())
        .unwrap_or_default();
    let start_ticks = fields
        .get(19)
        .and_then(|v| v.parse::<f64>().ok())
        .unwrap_or_default();
    let cpu_seconds = (utime + stime) / CLOCK_TICKS_PER_SECOND;

    let resident_memory_bytes = std::fs::read_to_string("/proc/self/statm")
        .ok()
        .and_then(|s| {
            s.split_whitespace()
                .nth(1)
                .and_then(|pages| pages.parse::<u64>().ok())
        })
        .map(|pages| pages * PAGE_SIZE_BYTES)
        .unwrap_or_default();

    let boot_time = std::fs::read_to_string("/proc/stat")
        .ok()
        .and_then(|s| {
            s.lines().find_map(|line| {
                line.strip_prefix("btime ")
                    .and_then(|value| value.parse::<f64>().ok())
            })
        })
        .unwrap_or_default();

    let open_fds = std::fs::read_dir("/proc/self/fd")
        .map(|entries| entries.count() as u64)
        .unwrap_or_default();

    ProcessMetrics {
        cpu_seconds,
        resident_memory_bytes,
        start_time_seconds: boot_time + start_ticks / CLOCK_TICKS_PER_SECOND,
        open_fds,
    }
}
