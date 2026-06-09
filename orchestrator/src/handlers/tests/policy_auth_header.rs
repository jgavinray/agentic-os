use super::*;

#[test]
fn provided_api_token_accepts_bearer_authorization() {
    let mut headers = HeaderMap::new();
    headers.insert(
        header::AUTHORIZATION,
        "Bearer sk-test".parse().expect("valid header"),
    );

    assert_eq!(provided_api_token(&headers), "sk-test");
}

#[test]
fn provided_api_token_accepts_anthropic_x_api_key() {
    let mut headers = HeaderMap::new();
    headers.insert("x-api-key", "sk-test".parse().expect("valid header"));

    assert_eq!(provided_api_token(&headers), "sk-test");
}
