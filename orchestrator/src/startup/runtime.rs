use std::sync::Arc;
use std::time::Duration;

use crate::{embedder, sentiment};

pub fn load_embedder(embed_model_path: &str) -> Arc<embedder::Embedder> {
    Arc::new(
        embedder::Embedder::load(embed_model_path)
            .expect("failed to load embedding model — run setup-models.sh first"),
    )
}

pub fn load_sentiment_classifier_from_env() -> Option<Arc<sentiment::SentimentClassifier>> {
    std::env::var("SENTIMENT_MODEL_PATH").ok().and_then(|path| {
        let threshold = std::env::var("SENTIMENT_THRESHOLD")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(0.70_f32);
        match sentiment::SentimentClassifier::load(&path, threshold) {
            Ok(c) => Some(Arc::new(c)),
            Err(e) => {
                tracing::warn!(
                    "sentiment classifier unavailable, negative feedback detection disabled: {e}"
                );
                None
            }
        }
    })
}

pub fn build_http_clients(
    litellm_request_timeout_secs: u64,
) -> Result<(reqwest::Client, reqwest::Client), anyhow::Error> {
    let http = reqwest::Client::builder()
        .connect_timeout(Duration::from_secs(10))
        .pool_idle_timeout(Duration::from_secs(90))
        .timeout(Duration::from_secs(litellm_request_timeout_secs))
        .build()?;

    let http_stream = reqwest::Client::builder()
        .connect_timeout(Duration::from_secs(10))
        .pool_idle_timeout(Duration::from_secs(90))
        // No overall timeout: streaming responses are long-lived.
        .build()?;

    Ok((http, http_stream))
}
