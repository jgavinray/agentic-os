use crate::state::{AppState, TokenUsage};
use crate::telemetry;

pub async fn request_summary(state: &AppState, prompt: String) -> Result<String, anyhow::Error> {
    let request_body = serde_json::json!({
        "model": state.summarizer_model.clone(),
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": state.summarizer_max_tokens,
        "temperature": 0.1,
    });

    let url = format!("{}/chat/completions", state.summarizer_url);
    let upstream_started = std::time::Instant::now();
    let mut request = state.http.post(&url).json(&request_body);
    if let Some(key) = &state.summarizer_key {
        request = request.bearer_auth(key);
    }

    let resp = request.send().await?;
    let status = resp.status();
    telemetry::record_upstream_summarizer(
        "chat_completions",
        upstream_started.elapsed(),
        &status.as_u16().to_string(),
    );
    if !status.is_success() {
        telemetry::record_upstream_summarizer_error(
            "chat_completions",
            telemetry::upstream_error_kind(status),
        );
    }

    let val: serde_json::Value = resp.json().await.inspect_err(|_| {
        telemetry::record_upstream_summarizer_error("chat_completions", "parse");
    })?;
    telemetry::record_tokens(
        &state.metrics,
        &TokenUsage::from_openai_value(&val),
        &state.summarizer_model,
    );

    summary_text_from_response(&val)
}

fn summary_text_from_response(val: &serde_json::Value) -> Result<String, anyhow::Error> {
    let content_val = &val["choices"][0]["message"]["content"];
    match content_val {
        serde_json::Value::String(s) => Ok(s.clone()),
        serde_json::Value::Array(blocks) => Ok(blocks
            .iter()
            .filter_map(|b| b.get("text").and_then(|t| t.as_str()))
            .collect::<Vec<_>>()
            .join("")),
        _ => anyhow::bail!("no content in summarization response"),
    }
}
