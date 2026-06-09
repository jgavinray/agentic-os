use super::*;

#[test]
fn rate_limited_response_sets_429_and_retry_after() {
    let response = rate_limited_response("secret-token", 3);
    assert_eq!(response.status(), StatusCode::TOO_MANY_REQUESTS);
    assert_eq!(
        response
            .headers()
            .get(header::RETRY_AFTER)
            .and_then(|v| v.to_str().ok()),
        Some("3")
    );
}

#[test]
fn parse_simple_single_entry() {
    let input = "agent-os,project-alpha";
    let mut parts = input.splitn(2, ',');
    let token = parts.next().unwrap_or(input).trim().to_string();
    let namespace = parts.next().unwrap_or(&token).trim().to_string();
    assert_eq!(token, "agent-os");
    assert_eq!(namespace, "project-alpha");
}

#[test]
fn parse_multiple_semicolon_entries() {
    let input = "agent-os,project-alpha;agent-os,project-beta;sk-work,work";
    let entries: Vec<(String, String)> = input
        .split(';')
        .map(|s| {
            let s = s.trim();
            let mut parts = s.splitn(2, ',');
            let token = parts.next().unwrap_or(s).trim().to_string();
            let namespace = parts.next().unwrap_or(&token).trim().to_string();
            (token, namespace)
        })
        .filter(|(t, _)| !t.is_empty())
        .collect();

    assert_eq!(entries.len(), 3);
    assert_eq!(
        entries[0],
        ("agent-os".to_string(), "project-alpha".to_string())
    );
    assert_eq!(
        entries[1],
        ("agent-os".to_string(), "project-beta".to_string())
    );
    assert_eq!(entries[2], ("sk-work".to_string(), "work".to_string()));
}

#[test]
fn parse_empty_token_filtered() {
    let input = "token1,ns1;;token2,ns2";
    let entries: Vec<(String, String)> = input
        .split(';')
        .map(|s| {
            let s = s.trim();
            let mut parts = s.splitn(2, ',');
            let token = parts.next().unwrap_or(s).trim().to_string();
            let namespace = parts.next().unwrap_or(&token).trim().to_string();
            (token, namespace)
        })
        .filter(|(t, _)| !t.is_empty())
        .collect();

    assert_eq!(entries.len(), 2);
}

#[test]
fn parse_fallback_to_token_when_no_namespace() {
    let input = "my-token";
    let mut parts = input.splitn(2, ',');
    let token = parts.next().unwrap_or(input).trim().to_string();
    let namespace = parts.next().unwrap_or(&token).trim().to_string();
    assert_eq!(token, "my-token");
    assert_eq!(namespace, "my-token");
}

#[test]
fn parse_default_value() {
    let input = "agent-os,agentic-os";
    let entries: Vec<(String, String)> = input
        .split(';')
        .map(|s| {
            let s = s.trim();
            let mut parts = s.splitn(2, ',');
            let token = parts.next().unwrap_or(s).trim().to_string();
            let namespace = parts.next().unwrap_or(&token).trim().to_string();
            (token, namespace)
        })
        .filter(|(t, _)| !t.is_empty())
        .collect();

    assert_eq!(entries.len(), 1);
    assert_eq!(
        entries[0],
        ("agent-os".to_string(), "agentic-os".to_string())
    );
}
