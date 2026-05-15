use serde_json::Value;
use sha2::{Digest, Sha256};

const VECTOR_SIZE: usize = 384;
const EVENT_COLLECTION: &str = "agent_events";

pub fn embedding(text: &str) -> Vec<f32> {
    let digest = Sha256::digest(text.as_bytes());
    let mut vector = Vec::with_capacity(VECTOR_SIZE);
    for i in 0..VECTOR_SIZE {
        let byte = digest[i % digest.len()];
        vector.push((byte as f32 / 255.0) - 0.5);
    }
    vector
}

pub async fn init(qdrant_url: &str) -> Result<(), anyhow::Error> {
    let url = format!("{qdrant_url}/collections/{EVENT_COLLECTION}");
    let body = serde_json::json!({
        "vectors": {
            "size": VECTOR_SIZE,
            "distance": "Cosine"
        }
    });
    let http = reqwest::Client::new();
    http.put(&url)
        .json(&body)
        .send()
        .await?
        .error_for_status()?;
    Ok(())
}

pub async fn store_event(qdrant_url: &str, event: &crate::db::AgentEvent) -> Result<(), anyhow::Error> {
    let url = format!("{qdrant_url}/collections/{EVENT_COLLECTION}/points?wait=true");
    let vector_text = event.vector_text();
    let vector = embedding(&vector_text);
    let payload = event.payload();

    let body = serde_json::json!({
        "points": [{
            "id": event.id,
            "vector": vector,
            "payload": payload
        }]
    });

    let http = reqwest::Client::new();
    http.put(&url)
        .json(&body)
        .send()
        .await?
        .error_for_status()?;

    Ok(())
}

pub async fn search(qdrant_url: &str, query: &str, limit: usize) -> Result<Value, anyhow::Error> {
    let url = format!("{qdrant_url}/collections/{EVENT_COLLECTION}/points/search");
    let vector = embedding(query);

    let body = serde_json::json!({
        "vector": vector,
        "limit": limit,
        "with_payload": true
    });

    let http = reqwest::Client::new();
    let resp = http
        .post(&url)
        .json(&body)
        .send()
        .await?
        .json::<Value>()
        .await?;

    Ok(resp)
}
