use serde_json::Value;

const VECTOR_SIZE: usize = 1024;
const EVENT_COLLECTION: &str = "agent_events";

pub async fn init(qdrant_url: &str) -> Result<(), anyhow::Error> {
    let url = format!("{qdrant_url}/collections/{EVENT_COLLECTION}");
    let body = serde_json::json!({
        "vectors": {"size": VECTOR_SIZE, "distance": "Cosine"}
    });
    let http = reqwest::Client::new();
    let resp = http.put(&url).json(&body).send().await?;
    if resp.status().is_success() || resp.status() == 409 {
        Ok(())
    } else {
        resp.error_for_status()?;
        Ok(())
    }
}

pub async fn store_event(
    embedder: &crate::embedder::Embedder,
    qdrant_url: &str,
    event: &crate::db::AgentEvent,
) -> Result<(), anyhow::Error> {
    let vector = embedder.embed(&event.vector_text()).await?;
    let http = reqwest::Client::new();
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

pub async fn search(
    embedder: &crate::embedder::Embedder,
    qdrant_url: &str,
    query: &str,
    limit: usize,
) -> Result<Vec<Value>, anyhow::Error> {
    let vector = embedder.embed(query).await?;
    let http = reqwest::Client::new();
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
