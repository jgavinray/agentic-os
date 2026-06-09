use super::*;
use crate::sse::extract_assistant_from_anthropic_response;
use serde_json::json;

#[test]
fn extract_sse_assembles_content_across_chunks() {
    let sse = "data: {\"choices\":[{\"delta\":{\"content\":\"Hello\"}}]}\n\
                   data: {\"choices\":[{\"delta\":{\"content\":\" world\"}}]}\n\
                   data: [DONE]\n";
    assert_eq!(extract_assistant_from_sse(sse), "Hello world");
}

#[test]
fn extract_sse_stops_at_done() {
    let sse = "data: {\"choices\":[{\"delta\":{\"content\":\"before\"}}]}\n\
                   data: [DONE]\n\
                   data: {\"choices\":[{\"delta\":{\"content\":\"after\"}}]}\n";
    assert_eq!(extract_assistant_from_sse(sse), "before");
}

#[test]
fn extract_sse_returns_empty_for_blank_input() {
    assert_eq!(extract_assistant_from_sse(""), "");
}

#[test]
fn extract_sse_ignores_malformed_json_data_lines() {
    let sse = "data: not-valid-json\n\
                   data: {\"choices\":[{\"delta\":{\"content\":\"ok\"}}]}\n\
                   data: [DONE]\n";
    assert_eq!(extract_assistant_from_sse(sse), "ok");
}

#[test]
fn extract_sse_ignores_non_data_lines() {
    let sse = ": keep-alive\n\
                   data: {\"choices\":[{\"delta\":{\"content\":\"hi\"}}]}\n\
                   data: [DONE]\n";
    assert_eq!(extract_assistant_from_sse(sse), "hi");
}

#[test]
fn extract_sse_usage_records_processed_cached_and_generated_tokens() {
    let sse = "data: {\"choices\":[{\"delta\":{\"content\":\"hi\"}}]}\n\
                   data: {\"choices\":[{\"delta\":{}}],\"usage\":{\"prompt_tokens\":120,\"completion_tokens\":40,\"prompt_tokens_details\":{\"cached_tokens\":80}}}\n\
                   data: [DONE]\n";
    let usage = extract_token_usage_from_sse(sse);
    assert_eq!(usage.processed_tokens, 120);
    assert_eq!(usage.cached_tokens, 80);
    assert_eq!(usage.generated_tokens, 40);
}

#[test]
fn extract_sse_handles_missing_content_field_gracefully() {
    let sse = "data: {\"choices\":[{\"delta\":{}}]}\n\
                   data: {\"choices\":[{\"delta\":{\"content\":\"real\"}}]}\n\
                   data: [DONE]\n";
    assert_eq!(extract_assistant_from_sse(sse), "real");
}

#[test]
fn anthropic_response_extracts_text_block() {
    let resp = json!({"content": [{"type": "text", "text": "hello"}]});
    assert_eq!(extract_assistant_from_anthropic_response(&resp), "hello");
}

#[test]
fn anthropic_response_skips_tool_use_blocks() {
    let resp = json!({"content": [
        {"type": "tool_use", "id": "c1", "name": "bash", "input": {}},
        {"type": "text", "text": "done"}
    ]});
    assert_eq!(extract_assistant_from_anthropic_response(&resp), "done");
}

#[test]
fn anthropic_response_concatenates_multiple_text_blocks() {
    let resp = json!({"content": [
        {"type": "text", "text": "foo"},
        {"type": "text", "text": "bar"}
    ]});
    assert_eq!(extract_assistant_from_anthropic_response(&resp), "foobar");
}

#[test]
fn anthropic_response_returns_empty_when_no_content() {
    assert_eq!(extract_assistant_from_anthropic_response(&json!({})), "");
}

#[test]
fn anthropic_response_truncates_at_500_chars() {
    let long = "x".repeat(600);
    let resp = json!({"content": [{"type": "text", "text": long}]});
    assert_eq!(extract_assistant_from_anthropic_response(&resp).len(), 500);
}

#[test]
fn anthropic_sse_usage_reads_message_start_and_delta() {
    let raw = concat!(
        "event: message_start\n",
        "data: {\"type\":\"message_start\",\"message\":{\"usage\":{\"input_tokens\":100,\"cache_read_input_tokens\":40}}}\n\n",
        "event: message_delta\n",
        "data: {\"type\":\"message_delta\",\"usage\":{\"output_tokens\":25}}\n\n",
    );
    let usage = extract_token_usage_from_anthropic_sse(raw);
    assert_eq!(usage.processed_tokens, 100);
    assert_eq!(usage.cached_tokens, 40);
    assert_eq!(usage.generated_tokens, 25);
}

#[test]
fn anthropic_sse_usage_is_zero_for_empty_stream() {
    let usage = extract_token_usage_from_anthropic_sse("");
    assert_eq!(usage.processed_tokens, 0);
    assert_eq!(usage.generated_tokens, 0);
}

#[test]
fn anthropic_sse_collects_text_deltas() {
    let raw = concat!(
        "event: content_block_delta\n",
        "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"hel\"}}\n\n",
        "event: content_block_delta\n",
        "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"lo\"}}\n\n",
    );
    assert_eq!(extract_assistant_from_anthropic_sse(raw), "hello");
}

#[test]
fn anthropic_sse_skips_non_text_deltas() {
    let raw = concat!(
        "event: content_block_delta\n",
        "data: {\"type\":\"content_block_delta\",\"index\":1,\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":\"{}\"}}\n\n",
    );
    assert_eq!(extract_assistant_from_anthropic_sse(raw), "");
}

#[test]
fn anthropic_sse_truncates_at_500_chars() {
    let chunk_text = "x".repeat(300);
    let make_line = |t: &str| {
        format!(
            "event: content_block_delta\ndata: {{\"type\":\"content_block_delta\",\"index\":0,\"delta\":{{\"type\":\"text_delta\",\"text\":\"{t}\"}}}}\n\n"
        )
    };
    let raw = make_line(&chunk_text) + &make_line(&chunk_text);
    assert_eq!(extract_assistant_from_anthropic_sse(&raw).len(), 500);
}
