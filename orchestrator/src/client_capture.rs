use axum::http::HeaderMap;
use deadpool_postgres::Pool;
use serde_json::Value;
use sha2::{Digest, Sha256};
use uuid::Uuid;

#[derive(Clone, Debug)]
pub struct RawHttpCapture {
    pub exchange_id: Uuid,
    pub attempt_id: Option<Uuid>,
    pub endpoint: String,
    pub method: String,
    pub path: String,
    pub namespace: Option<String>,
    pub repo: Option<String>,
    pub task: Option<String>,
    pub request_headers: Value,
    pub raw_request_body: Vec<u8>,
    pub parsed_request_body: Option<Value>,
    pub forwarded_request_body: Option<Vec<u8>>,
    pub response_status: Option<i32>,
    pub response_headers: Option<Value>,
    pub raw_response_body: Option<Vec<u8>>,
}

impl RawHttpCapture {
    pub fn new(endpoint: &str, headers: &HeaderMap, raw_request_body: Vec<u8>) -> Self {
        Self {
            exchange_id: Uuid::new_v4(),
            attempt_id: None,
            endpoint: endpoint.to_string(),
            method: "POST".to_string(),
            path: format!("/v1/{endpoint}"),
            namespace: None,
            repo: None,
            task: None,
            request_headers: headers_to_json(headers),
            raw_request_body,
            parsed_request_body: None,
            forwarded_request_body: None,
            response_status: None,
            response_headers: None,
            raw_response_body: None,
        }
    }

    pub fn request_sha256(&self) -> String {
        sha256_hex(&self.raw_request_body)
    }

    pub fn forwarded_sha256(&self) -> Option<String> {
        self.forwarded_request_body
            .as_ref()
            .map(|body| sha256_hex(body))
    }

    pub fn response_sha256(&self) -> Option<String> {
        self.raw_response_body.as_ref().map(|body| sha256_hex(body))
    }
}

pub async fn init(pool: &Pool) -> Result<(), anyhow::Error> {
    let conn = pool.get().await?;
    conn.batch_execute(
        "CREATE TABLE IF NOT EXISTS raw_http_exchanges (
            exchange_id UUID PRIMARY KEY,
            attempt_id UUID,
            received_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            endpoint TEXT NOT NULL,
            method TEXT NOT NULL,
            path TEXT NOT NULL,
            namespace TEXT,
            repo TEXT,
            task TEXT,
            request_headers JSONB NOT NULL,
            raw_request_body BYTEA NOT NULL,
            parsed_request_body JSONB,
            forwarded_request_body BYTEA,
            raw_request_body_sha256 TEXT NOT NULL,
            forwarded_request_body_sha256 TEXT,
            response_status INT,
            response_headers JSONB,
            raw_response_body BYTEA,
            raw_response_body_sha256 TEXT
        );
        CREATE INDEX IF NOT EXISTS raw_http_exchanges_received_at_idx
            ON raw_http_exchanges(received_at DESC);
        CREATE INDEX IF NOT EXISTS raw_http_exchanges_attempt_id_idx
            ON raw_http_exchanges(attempt_id);
        CREATE INDEX IF NOT EXISTS raw_http_exchanges_endpoint_idx
            ON raw_http_exchanges(endpoint);",
    )
    .await?;
    Ok(())
}

pub async fn record(pool: &Pool, capture: &RawHttpCapture) -> Result<(), anyhow::Error> {
    let conn = pool.get().await?;
    let request_sha = capture.request_sha256();
    let forwarded_sha = capture.forwarded_sha256();
    let response_sha = capture.response_sha256();
    conn.execute(
        "INSERT INTO raw_http_exchanges
         (exchange_id, attempt_id, endpoint, method, path, namespace, repo, task,
          request_headers, raw_request_body, parsed_request_body, forwarded_request_body,
          raw_request_body_sha256, forwarded_request_body_sha256, response_status,
          response_headers, raw_response_body, raw_response_body_sha256)
         VALUES
         ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18)",
        &[
            &capture.exchange_id,
            &capture.attempt_id,
            &capture.endpoint,
            &capture.method,
            &capture.path,
            &capture.namespace,
            &capture.repo,
            &capture.task,
            &capture.request_headers,
            &capture.raw_request_body,
            &capture.parsed_request_body,
            &capture.forwarded_request_body,
            &request_sha,
            &forwarded_sha,
            &capture.response_status,
            &capture.response_headers,
            &capture.raw_response_body,
            &response_sha,
        ],
    )
    .await?;
    Ok(())
}

pub fn headers_to_json(headers: &HeaderMap) -> Value {
    let mut object = serde_json::Map::new();
    for (name, value) in headers {
        let entry = object
            .entry(name.as_str().to_string())
            .or_insert_with(|| Value::Array(vec![]));
        if let Value::Array(values) = entry {
            values.push(Value::String(
                value
                    .to_str()
                    .map(str::to_string)
                    .unwrap_or_else(|_| format!("{:?}", value.as_bytes())),
            ));
        }
    }
    Value::Object(object)
}

pub fn to_json_bytes(value: &Value) -> Vec<u8> {
    serde_json::to_vec(value).unwrap_or_default()
}

fn sha256_hex(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    format!("{digest:x}")
}
