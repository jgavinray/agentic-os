use serde_json::Value;

const VECTOR_SIZE: usize = 1024;
const EVENT_COLLECTION: &str = "agent_events";

pub async fn init(qdrant_url: &str) -> Result<(), anyhow::Error> {
    let url = format!("{qdrant_url}/collections/{EVENT_COLLECTION}");
    let body = serde_json::json!({
        "vectors": {"size": VECTOR_SIZE, "distance": "Cosine"}
    });
    let http = reqwest::Client::new();
    let started = std::time::Instant::now();
    let resp = http.put(&url).json(&body).send().await?;
    let status = resp.status();
    crate::telemetry::record_qdrant_request(
        "create_collection",
        started.elapsed(),
        &status.as_u16().to_string(),
    );
    if status.is_success() || status == 409 {
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

    let started = std::time::Instant::now();
    let resp = http.put(&url).json(&body).send().await?;
    let status = resp.status();
    crate::telemetry::record_qdrant_request(
        "upsert",
        started.elapsed(),
        &status.as_u16().to_string(),
    );
    if status.is_success() || status == 409 {
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

    let started = std::time::Instant::now();
    let response = http.post(&url).json(&body).send().await?;
    let status = response.status();
    crate::telemetry::record_qdrant_request(
        "search",
        started.elapsed(),
        &status.as_u16().to_string(),
    );
    let resp: Value = response.json().await?;

    Ok(resp["result"].as_array().cloned().unwrap_or_default())
}
