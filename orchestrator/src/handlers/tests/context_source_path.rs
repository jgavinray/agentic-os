#[test]
fn failure_history_and_validation_capture_are_feature_flagged() {
    let context_src = include_str!("../../context_packing/build.rs");
    let validations_src = include_str!("../../routes/validations.rs");
    assert!(context_src.contains("state.execution_feedback_enabled"));
    let ctx_start = context_src
        .find("async fn build_cached_context")
        .expect("build_cached_context not found");
    let ctx_body: String = context_src[ctx_start..].chars().take(4500).collect();
    assert!(ctx_body.contains("state.execution_feedback_enabled"));

    let validation_start = validations_src
        .find("pub async fn validations")
        .expect("validations handler not found");
    let validation_body: String = validations_src[validation_start..]
        .chars()
        .take(1200)
        .collect();
    assert!(validation_body.contains("!state.execution_feedback_enabled"));
}

#[test]
fn trajectory_capture_is_feature_flagged() {
    let sessions_src = include_str!("../../routes/sessions.rs");
    let trajectory_src = include_str!("../../background/trajectory.rs");
    assert!(sessions_src.contains("state.trajectory_capture_enabled"));
    let append_start = sessions_src
        .find("pub async fn append_event")
        .expect("append_event handler not found");
    let append_body: String = sessions_src[append_start..].chars().take(1400).collect();
    assert!(append_body.contains("req.trajectory_id = None"));
    assert!(append_body.contains("req.attempt_index = None"));
    assert!(append_body.contains("req.event_role = None"));

    let sweep_start = trajectory_src
        .find("pub async fn run_trajectory_idle_sweep")
        .expect("trajectory idle sweep not found");
    let sweep_body: String = trajectory_src[sweep_start..].chars().take(800).collect();
    assert!(sweep_body.contains("!state.trajectory_capture_enabled"));
}

#[test]
fn pack_context_into_req_uses_async_cache_refresh() {
    let src = include_str!("../context.rs");
    let pctr_start = src
        .find("async fn pack_context_into_req")
        .expect("pack_context_into_req not found");
    let body: String = src[pctr_start..].chars().take(1500).collect();
    assert!(
        body.contains("cached_context_for_request"),
        "pack_context_into_req should use cached/minimal context immediately"
    );
    assert!(
        !body.contains("get_or_build_cached_context("),
        "pack_context_into_req should not await full context construction"
    );
    assert!(
        include_str!("../../context_packing/cache.rs").contains("fn spawn_context_cache_refresh")
            && include_str!("../../context_packing/mod.rs")
                .contains("get_or_build_cached_context_inner")
            && include_str!("../../background/mod.rs").contains("tokio::spawn(async move"),
        "context cache refresh should run in the background"
    );
}
