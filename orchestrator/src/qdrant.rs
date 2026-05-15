use serde_json::Value;

const VECTOR_SIZE: usize = 384;
const EVENT_COLLECTION: &str = "agent_events";

// BUG-1: Real embedding via LiteLLM — replaces the previous SHA-256 hash stub.
async fn embed_text(
    http: &reqwest::Client,
    litellm_url: &str,
    litellm_key: &str,
    model: &str,
    text: &str,
) -> Result<Vec<f32>, anyhow::Error> {
    let url = format!("{}/embeddings", litellm_url);
    let body = serde_json::json!({"input": text, "model": model});

    let resp: Value = http
        .post(&url)
        .bearer_auth(litellm_key)
        .json(&body)
        .send()
        .await?
        .json()
        .await?;

    let vector: Vec<f32> = resp["data"][0]["embedding"]
        .as_array()
        .ok_or_else(|| anyhow::anyhow!("embedding response missing data[0].embedding — check EMBEDDING_MODEL is configured in LiteLLM"))?
        .iter()
        .filter_map(|v| v.as_f64().map(|f| f as f32))
        .collect();

    anyhow::ensure!(!vector.is_empty(), "empty embedding vector returned from {model}");
    anyhow::ensure!(
        vector.len() == VECTOR_SIZE,
        "embedding model {model} returned {} dims but Qdrant collection expects {VECTOR_SIZE}",
        vector.len()
    );

    Ok(vector)
}

pub async fn init(qdrant_url: &str) -> Result<(), anyhow::Error> {
    let url = format!("{qdrant_url}/collections/{EVENT_COLLECTION}");
    let body = serde_json::json!({
        "vectors": {"size": VECTOR_SIZE, "distance": "Cosine"}
    });
    let http = reqwest::Client::new();
    let resp = http.put(&url).json(&body).send().await?;
    // 200/201 = created, 409 = already exists — both are fine
    if resp.status().is_success() || resp.status() == 409 {
        Ok(())
    } else {
        resp.error_for_status()?;
        Ok(())
    }
}

pub async fn store_event(
    http: &reqwest::Client,
    litellm_url: &str,
    litellm_key: &str,
    embedding_model: &str,
    qdrant_url: &str,
    event: &crate::db::AgentEvent,
) -> Result<(), anyhow::Error> {
    let vector = embed_text(http, litellm_url, litellm_key, embedding_model, &event.vector_text()).await?;
    let url = format!("{qdrant_url}/collections/{EVENT_COLLECTION}/points?wait=true");

    let body = serde_json::json!({
        "points": [{"id": event.id, "vector": vector, "payload": event.payload()}]
    });

    let resp = http.put(&url).json(&body).send().await?;
    if resp.status().is_success() || resp.status() == 409 {
        Ok(())
    } else {
        resp.error_for_status()?;
        Ok(())
    }
}

/// Returns the Qdrant `result` array (scored points with payloads) directly.
pub async fn search(
    http: &reqwest::Client,
    litellm_url: &str,
    litellm_key: &str,
    embedding_model: &str,
    qdrant_url: &str,
    query: &str,
    limit: usize,
) -> Result<Vec<Value>, anyhow::Error> {
    let vector = embed_text(http, litellm_url, litellm_key, embedding_model, query).await?;
    let url = format!("{qdrant_url}/collections/{EVENT_COLLECTION}/points/search");

    let body = serde_json::json!({
        "vector": vector,
        "limit": limit,
        "with_payload": true,
        "score_threshold": 0.4
    });

    let resp: Value = http.post(&url).json(&body).send().await?.json().await?;

    Ok(resp["result"].as_array().cloned().unwrap_or_default())
}
