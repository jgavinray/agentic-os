use deadpool_postgres::Pool;
use refinery::Target;
use std::ops::DerefMut;

mod embedded {
    use refinery::embed_migrations;
    embed_migrations!("./migrations");
}

pub async fn run(pool: &Pool) -> Result<(), anyhow::Error> {
    let mut conn = pool.get().await?;
    let client = conn.deref_mut().deref_mut();

    if should_fake_baseline(client).await? {
        tracing::info!(
            target: "migrations",
            "legacy schema detected without migration history; marking baseline migration applied"
        );
        embedded::migrations::runner()
            .set_target(Target::FakeVersion(1))
            .run_async(client)
            .await?;
    }

    let report = embedded::migrations::runner()
        .set_grouped(true)
        .run_async(client)
        .await?;
    for migration in report.applied_migrations() {
        tracing::info!(
            target: "migrations",
            version = migration.version(),
            name = migration.name(),
            "applied migration"
        );
    }

    Ok(())
}

async fn should_fake_baseline(client: &mut tokio_postgres::Client) -> Result<bool, anyhow::Error> {
    if table_exists(client, "refinery_schema_history").await? {
        return Ok(false);
    }

    let legacy_tables = [
        "agent_sessions",
        "agent_events",
        "error_index",
        "token_usage",
    ];
    for table in legacy_tables {
        if !table_exists(client, table).await? {
            return Ok(false);
        }
    }

    Ok(true)
}

async fn table_exists(
    client: &mut tokio_postgres::Client,
    table_name: &str,
) -> Result<bool, anyhow::Error> {
    let row = client
        .query_one(
            "SELECT to_regclass($1)::text IS NOT NULL AS exists",
            &[&format!("public.{table_name}")],
        )
        .await?;
    Ok(row.get("exists"))
}
