use metrics::{counter, histogram};
use std::time::Duration;

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
