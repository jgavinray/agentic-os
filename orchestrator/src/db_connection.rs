use deadpool_postgres::{Config, Pool, PoolConfig};
use tokio_postgres::NoTls;

const SINGLE_WRITER_LOCK_KEY: i64 = 0x4167_656e_7469_634f;

pub fn create_pool(database_url: &str) -> Result<Pool, anyhow::Error> {
    let mut cfg = Config::new();
    cfg.url = Some(database_url.to_string());
    // BUG-9: assign explicit PoolConfig. Mutating cfg.pool while it is None
    // leaves deadpool's default in place.
    cfg.pool = Some(PoolConfig::new(16));
    let pool = cfg.create_pool(None, NoTls)?;
    Ok(pool)
}

pub struct SingleWriterGuard {
    conn: deadpool_postgres::Object,
}

pub async fn acquire_single_writer_lock(pool: &Pool) -> Result<SingleWriterGuard, anyhow::Error> {
    let conn = pool.get().await?;
    let acquired: bool = conn
        .query_one(
            "SELECT pg_try_advisory_lock($1)",
            &[&SINGLE_WRITER_LOCK_KEY],
        )
        .await?
        .get(0);
    if !acquired {
        anyhow::bail!(
            "another orchestrator process already owns this Postgres database; \
             single-writer advisory lock {SINGLE_WRITER_LOCK_KEY} is held"
        );
    }
    tracing::info!(
        target: "db",
        lock_key = SINGLE_WRITER_LOCK_KEY,
        "acquired single-writer advisory lock"
    );
    Ok(SingleWriterGuard { conn })
}

impl SingleWriterGuard {
    pub async fn release(self) {
        match self
            .conn
            .execute("SELECT pg_advisory_unlock($1)", &[&SINGLE_WRITER_LOCK_KEY])
            .await
        {
            Ok(_) => tracing::info!(
                target: "db",
                lock_key = SINGLE_WRITER_LOCK_KEY,
                "released single-writer advisory lock"
            ),
            Err(e) => tracing::warn!(
                target: "db",
                lock_key = SINGLE_WRITER_LOCK_KEY,
                "failed to release single-writer advisory lock: {e}"
            ),
        }
    }
}
