use axum::extract::MatchedPath;
use axum::extract::Request;
use axum::http::{header, HeaderMap};
use axum::middleware::Next;
use axum::response::Response;
use metrics::{counter, gauge, histogram};
use std::time::Instant;

pub async fn http_metrics_middleware(req: Request, next: Next) -> Response {
    let endpoint = req
        .extensions()
        .get::<MatchedPath>()
        .map(|path| path.as_str().to_string())
        .unwrap_or_else(|| req.uri().path().to_string());
    let method = req.method().as_str().to_string();
    let started = Instant::now();

    gauge!("http_requests_in_flight", "endpoint" => endpoint.clone()).increment(1.0);
    let response = next.run(req).await;
    gauge!("http_requests_in_flight", "endpoint" => endpoint.clone()).decrement(1.0);

    let status = response.status().as_u16().to_string();
    counter!(
        "http_requests_total",
        "endpoint" => endpoint.clone(),
        "method" => method.clone(),
        "status" => status
    )
    .increment(1);
    histogram!(
        "http_request_duration_seconds",
        "endpoint" => endpoint,
        "method" => method
    )
    .record(started.elapsed().as_secs_f64());

    response
}

pub fn prometheus_content_type() -> HeaderMap {
    let mut headers = HeaderMap::new();
    headers.insert(
        header::CONTENT_TYPE,
        "text/plain; version=0.0.4".parse().unwrap(),
    );
    headers
}
