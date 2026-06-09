use super::*;

static MIGRATION_LOCK: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());

fn test_attempt(status_label: &str) -> LiteLlmCallAttempt {
    let route = RouteSelection {
        routed_model: "agentic/strong".to_string(),
        selected_route: "default".to_string(),
        selection_reason: "default_model".to_string(),
        policy_version: "default-routing-v1".to_string(),
    };
    new_attempt(
        None,
        None,
        None,
        "test-ns".to_string(),
        "test-repo".to_string(),
        "test-task".to_string(),
        "chat_completions",
        format!("requested-{status_label}"),
        &route,
        exact_cache_decision("chat_completions", &json!({"temperature": 0}), false),
        Some(context_pack_hash("stable context")),
        None,
    )
}

async fn test_pool_with_migrations() -> deadpool_postgres::Pool {
    let database_url = std::env::var("TEST_DATABASE_URL").expect("TEST_DATABASE_URL must be set");
    let pool = crate::db::create_pool(&database_url).expect("pool");
    let _guard = MIGRATION_LOCK.lock().await;
    crate::migrations::run(&pool).await.expect("migrations");
    pool
}

#[tokio::test]
#[ignore = "requires TEST_DATABASE_URL pointing at a disposable Postgres database"]
async fn db_ledger_preinsert_and_terminal_updates_are_single_row() {
    let pool = test_pool_with_migrations().await;

    let network = test_attempt("network");
    let network_id = network.attempt_id;
    let mut network_finalizer = LiteLlmCallFinalizer::begin(pool.clone(), network).await;
    {
        let conn = pool.get().await.expect("conn");
        let row = conn
            .query_one(
                "SELECT count(*)::BIGINT, max(terminal_status) FROM litellm_call_ledger WHERE attempt_id = $1",
                &[&network_id],
            )
            .await
            .expect("preinsert row");
        let count: i64 = row.get(0);
        let terminal_status: Option<String> = row.get(1);
        assert_eq!(count, 1);
        assert_eq!(terminal_status, None);
    }
    network_finalizer.attempt_mut().completed_at = Some(Instant::now());
    network_finalizer
        .finalize(
            TerminalStatus::NetworkError,
            Some("connect"),
            Some("connection refused"),
            ProviderCacheCounters::default(),
        )
        .await;
    {
        let conn = pool.get().await.expect("conn");
        let row = conn
            .query_one(
                "SELECT count(*)::BIGINT, max(terminal_status), max(total_latency_ms) FROM litellm_call_ledger WHERE attempt_id = $1",
                &[&network_id],
            )
            .await
            .expect("network row");
        let count: i64 = row.get(0);
        let terminal_status: Option<String> = row.get(1);
        let total_latency_ms: Option<i64> = row.get(2);
        assert_eq!(count, 1);
        assert_eq!(terminal_status.as_deref(), Some("network_error"));
        assert!(total_latency_ms.is_some());
    }

    let parse = test_attempt("parse");
    let parse_id = parse.attempt_id;
    let parse_finalizer = LiteLlmCallFinalizer::begin(pool.clone(), parse).await;
    parse_finalizer
        .finalize(
            TerminalStatus::ParseError,
            Some("parse"),
            Some("invalid upstream response"),
            ProviderCacheCounters::default(),
        )
        .await;
    let conn = pool.get().await.expect("conn");
    let row = conn
        .query_one(
            "SELECT count(*)::BIGINT, max(terminal_status) FROM litellm_call_ledger WHERE attempt_id = $1",
            &[&parse_id],
        )
        .await
        .expect("parse row");
    let count: i64 = row.get(0);
    let terminal_status: Option<String> = row.get(1);
    assert_eq!(count, 1);
    assert_eq!(terminal_status.as_deref(), Some("parse_error"));
}

#[tokio::test]
#[ignore = "requires TEST_DATABASE_URL pointing at a disposable Postgres database"]
async fn db_ledger_streaming_first_token_ms_null_and_populated() {
    let pool = test_pool_with_migrations().await;

    let no_token = test_attempt("stream-no-token");
    let no_token_id = no_token.attempt_id;
    let no_token_finalizer = LiteLlmCallFinalizer::begin(pool.clone(), no_token).await;
    no_token_finalizer
        .finalize(
            TerminalStatus::Success,
            None,
            None,
            ProviderCacheCounters::default(),
        )
        .await;

    let mut with_token = test_attempt("stream-token");
    let with_token_id = with_token.attempt_id;
    with_token.first_token_at = Some(with_token.started_at + std::time::Duration::from_millis(42));
    with_token.completed_at = Some(with_token.started_at + std::time::Duration::from_millis(99));
    let with_token_finalizer = LiteLlmCallFinalizer::begin(pool.clone(), with_token).await;
    with_token_finalizer
        .finalize(
            TerminalStatus::Success,
            None,
            None,
            ProviderCacheCounters::default(),
        )
        .await;

    let conn = pool.get().await.expect("conn");
    let row = conn
        .query_one(
            "SELECT
                (SELECT first_token_ms FROM litellm_call_ledger WHERE attempt_id = $1),
                (SELECT first_token_ms FROM litellm_call_ledger WHERE attempt_id = $2)",
            &[&no_token_id, &with_token_id],
        )
        .await
        .expect("stream rows");
    let no_token_ms: Option<i64> = row.get(0);
    let with_token_ms: Option<i64> = row.get(1);
    assert_eq!(no_token_ms, None);
    assert_eq!(with_token_ms, Some(42));
}
