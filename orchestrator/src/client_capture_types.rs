use axum::http::HeaderMap;
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
