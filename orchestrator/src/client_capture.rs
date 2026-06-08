use axum::http::StatusCode;
use deadpool_postgres::Pool;

pub use crate::client_capture_types::{headers_to_json, to_json_bytes, RawHttpCapture};

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

pub async fn record_best_effort(pool: Option<&Pool>, capture: RawHttpCapture) {
    let Some(pool) = pool else {
        return;
    };
    match record(pool, &capture).await {
        Ok(()) => match crate::prompt_intervention_assembly::records_from_capture(&capture) {
            Ok(records) => {
                for record in records {
                    crate::prompt_intervention_records::spawn_insert_best_effort(
                        Some(pool),
                        record,
                    );
                }
            }
            Err(e) => {
                crate::telemetry_prompt_interventions::record_prompt_intervention_runtime_write_attempt(
                    "assembly_error",
                );
                tracing::warn!(
                    exchange_id = %capture.exchange_id,
                    endpoint = %capture.endpoint,
                    "failed to assemble prompt intervention records: {e}"
                );
            }
        },
        Err(e) => {
            tracing::warn!(
                exchange_id = %capture.exchange_id,
                endpoint = %capture.endpoint,
                "failed to record raw client capture: {e}"
            );
        }
    }
}

pub async fn record_response_best_effort(
    pool: Option<&Pool>,
    mut capture: RawHttpCapture,
    status: StatusCode,
    content_type: &'static str,
    body: Vec<u8>,
) {
    capture.response_status = Some(status.as_u16() as i32);
    capture.response_headers = Some(serde_json::json!({"content-type": [content_type]}));
    capture.raw_response_body = Some(body);
    record_best_effort(pool, capture).await;
}
