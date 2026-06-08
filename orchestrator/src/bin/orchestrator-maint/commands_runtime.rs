use orchestrator::{db, migrations};
use std::env;

pub(super) async fn open_migrated_pool() -> Result<deadpool_postgres::Pool, anyhow::Error> {
    let db_url = env::var("DATABASE_URL").expect("DATABASE_URL must be set");
    let pool = db::create_pool(&db_url)?;
    migrations::run(&pool).await?;
    Ok(pool)
}

pub(super) async fn open_capture_pool() -> Result<deadpool_postgres::Pool, anyhow::Error> {
    let db_url = env::var("CAPTURE_DATABASE_URL").expect("CAPTURE_DATABASE_URL must be set");
    let pool = db::create_pool(&db_url)?;
    orchestrator::client_capture::init(&pool).await?;
    orchestrator::prompt_intervention_records::init(&pool).await?;
    Ok(pool)
}

pub(super) fn execution_feedback_enabled() -> bool {
    env::var("EXECUTION_FEEDBACK_ENABLED")
        .map(|v| {
            !matches!(
                v.to_ascii_lowercase().as_str(),
                "0" | "false" | "no" | "off"
            )
        })
        .unwrap_or(true)
}
