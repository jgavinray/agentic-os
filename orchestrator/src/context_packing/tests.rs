#[test]
fn context_pack_parallelizes_context_io_calls() {
    let src = include_str!("build.rs");
    let ctx_start = src
        .find("async fn build_cached_context")
        .expect("build_cached_context not found in source");
    let ctx_body: String = src[ctx_start..].chars().take(6500).collect();

    assert!(ctx_body.contains("tokio::join!"));
    assert!(ctx_body.contains("db::get_context_evidence_for_policy"));
    assert!(ctx_body.contains("hybrid_search"));
    assert!(ctx_body.contains("db::get_active_errors"));
    assert!(ctx_body.contains("db::get_failure_history_for_signatures"));

    let join_block_start = ctx_body
        .find("tokio::join!")
        .expect("tokio::join! not found");
    let join_block: String = ctx_body[join_block_start..].chars().take(1500).collect();
    assert!(
        join_block.contains("get_context_evidence_for_policy")
            && join_block.contains("hybrid_search")
            && join_block.contains("get_active_errors")
            && join_block.contains("get_failure_history_for_signatures")
    );
}

#[test]
fn context_pack_preserves_error_propagation_for_events() {
    let src = include_str!("endpoint.rs");
    let ctx_start = src
        .find("pub async fn context_pack")
        .expect("context_pack not found");
    let ctx_end = (ctx_start + 2000).min(src.len());
    let ctx_body = &src[ctx_start..ctx_end];
    assert!(ctx_body.contains("INTERNAL_SERVER_ERROR"));
}
