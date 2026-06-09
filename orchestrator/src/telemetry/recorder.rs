use metrics_exporter_prometheus::{Matcher, PrometheusBuilder, PrometheusHandle};

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

pub fn install_prometheus_recorder() -> Result<PrometheusHandle, anyhow::Error> {
    Ok(PrometheusBuilder::new()
        .set_buckets_for_metric(
            Matcher::Full("http_request_duration_seconds".to_string()),
            HTTP_BUCKETS,
        )?
        .set_buckets_for_metric(
            Matcher::Full("upstream_litellm_duration_seconds".to_string()),
            UPSTREAM_BUCKETS,
        )?
        .set_buckets_for_metric(
            Matcher::Full("summarizer_upstream_duration_seconds".to_string()),
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
        .set_buckets_for_metric(
            Matcher::Full("feature_extraction_duration_seconds".to_string()),
            IO_BUCKETS,
        )?
        .install_recorder()?)
}
