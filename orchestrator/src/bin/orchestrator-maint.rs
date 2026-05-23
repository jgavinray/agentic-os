use orchestrator::{
    db, execution_feedback, feature_extraction, harness_feedback, logging, migrations,
    request_classification,
};
use std::env;
use std::ops::DerefMut;
use uuid::Uuid;

const DEFAULT_BATCH_SIZE: i64 = 500;

#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    let logging_guard = logging::init_logging()?;
    let result = run().await;
    logging_guard.shutdown();
    result
}

async fn run() -> Result<(), anyhow::Error> {
    let mut args = env::args().skip(1);
    let Some(command) = args.next() else {
        print_usage();
        anyhow::bail!("missing command");
    };

    match command.as_str() {
        "backfill-signatures" => {
            let opts = BackfillOptions::parse(args.collect())?;
            if !execution_feedback_enabled() {
                println!(
                    "EXECUTION_FEEDBACK_ENABLED=false; skipping signature backfill with no changes"
                );
                return Ok(());
            }
            let db_url = env::var("DATABASE_URL").expect("DATABASE_URL must be set");
            let pool = db::create_pool(&db_url)?;
            migrations::run(&pool).await?;
            let report = backfill_signatures(&pool, opts).await?;
            println!(
                "backfill-signatures: pending={} updated={} dry_run={} batch_size={}",
                report.pending, report.updated, report.dry_run, report.batch_size
            );
            Ok(())
        }
        "extract-features" => {
            let opts = ExtractFeaturesOptions::parse(args.collect())?;
            let db_url = env::var("DATABASE_URL").expect("DATABASE_URL must be set");
            let pool = db::create_pool(&db_url)?;
            migrations::run(&pool).await?;
            let report = feature_extraction::run_backfill(
                &pool,
                &feature_extraction::BackfillOptions {
                    repo: opts.repo,
                    session_id: opts.session_id,
                    trajectory_id: opts.trajectory_id,
                    since: opts.since,
                    dry_run: opts.dry_run,
                    batch_size: opts.batch_size,
                    skip_bootstrap_tagging: opts.skip_bootstrap_tagging,
                },
                &feature_extraction::ExtractionConfig {
                    feature_window_sec: feature_extraction::feature_window_sec_from_env(),
                    constraint_freshness_window_sec:
                        feature_extraction::constraint_freshness_window_sec_from_env(),
                    max_operational_constraints:
                        feature_extraction::max_operational_constraints_from_env(),
                    evaluation_time: chrono::Utc::now(),
                },
            )
            .await?;
            println!(
                "extract-features: events_scanned={} bootstrap_updated={} feature_records={} persisted_records={} dry_run={} batch_size={} skip_bootstrap_tagging={}",
                report.events_scanned,
                report.bootstrap_updated,
                report.feature_records,
                report.persisted_records,
                report.dry_run,
                opts.batch_size,
                opts.skip_bootstrap_tagging
            );
            Ok(())
        }
        "classify-harness-feedback" => {
            let opts = HarnessFeedbackOptions::parse(args.collect())?;
            let db_url = env::var("DATABASE_URL").expect("DATABASE_URL must be set");
            let pool = db::create_pool(&db_url)?;
            migrations::run(&pool).await?;
            let report = harness_feedback::run_backfill(
                &pool,
                &harness_feedback::BackfillOptions {
                    repo: opts.repo,
                    session_id: opts.session_id,
                    since: opts.since,
                    dry_run: opts.dry_run,
                    batch_size: opts.batch_size,
                },
            )
            .await?;
            println!(
                "classify-harness-feedback: events_scanned={} updated={} quarantined={} dry_run={} batch_size={}",
                report.events_scanned,
                report.updated,
                report.quarantined,
                report.dry_run,
                report.batch_size
            );
            Ok(())
        }
        "classify-requests" => {
            let opts = RequestClassificationOptions::parse(args.collect())?;
            let db_url = env::var("DATABASE_URL").expect("DATABASE_URL must be set");
            let pool = db::create_pool(&db_url)?;
            migrations::run(&pool).await?;
            let report = request_classification::run_backfill(
                &pool,
                &request_classification::BackfillOptions {
                    repo: opts.repo,
                    session_id: opts.session_id,
                    since: opts.since,
                    dry_run: opts.dry_run,
                    repair: opts.repair,
                    repair_stale: opts.repair,
                    batch_size: opts.batch_size,
                },
            )
            .await?;
            println!(
                "classify-requests: events_scanned={} inserted={} updated={} skipped={} dry_run={} batch_size={}",
                report.events_scanned,
                report.inserted,
                report.updated,
                report.skipped,
                report.dry_run,
                report.batch_size
            );
            Ok(())
        }
        "request-classification-report" => {
            let opts = RequestClassificationReportOptions::parse(args.collect())?;
            let db_url = env::var("DATABASE_URL").expect("DATABASE_URL must be set");
            let pool = db::create_pool(&db_url)?;
            migrations::run(&pool).await?;
            let report = request_classification::request_classification_report(
                &pool,
                &request_classification::ReportOptions {
                    repo: opts.repo,
                    since: opts.since,
                },
            )
            .await?;
            print_request_classification_report(&report);
            Ok(())
        }
        _ => {
            print_usage();
            anyhow::bail!("unknown command: {command}");
        }
    }
}

fn execution_feedback_enabled() -> bool {
    env::var("EXECUTION_FEEDBACK_ENABLED")
        .map(|v| {
            !matches!(
                v.to_ascii_lowercase().as_str(),
                "0" | "false" | "no" | "off"
            )
        })
        .unwrap_or(true)
}

fn print_usage() {
    eprintln!(
        "usage: orchestrator-maint backfill-signatures [--dry-run] [--batch-size N]\n       orchestrator-maint extract-features [--repo REPO] [--session SESSION] [--trajectory TRAJECTORY] [--since TIMESTAMP] [--dry-run] [--batch-size N] [--skip-bootstrap-tagging]\n       orchestrator-maint classify-harness-feedback [--repo REPO] [--session SESSION] [--since TIMESTAMP] [--dry-run] [--batch-size N]\n       orchestrator-maint classify-requests [--repo REPO] [--session SESSION] [--since TIMESTAMP] [--dry-run] [--repair] [--batch-size N]\n       orchestrator-maint request-classification-report [--repo REPO] [--since TIMESTAMP]"
    );
}

struct BackfillOptions {
    dry_run: bool,
    batch_size: i64,
}

struct ExtractFeaturesOptions {
    repo: Option<String>,
    session_id: Option<String>,
    trajectory_id: Option<Uuid>,
    since: Option<chrono::DateTime<chrono::Utc>>,
    dry_run: bool,
    batch_size: i64,
    skip_bootstrap_tagging: bool,
}

struct HarnessFeedbackOptions {
    repo: Option<String>,
    session_id: Option<String>,
    since: Option<chrono::DateTime<chrono::Utc>>,
    dry_run: bool,
    batch_size: i64,
}

struct RequestClassificationOptions {
    repo: Option<String>,
    session_id: Option<String>,
    since: Option<chrono::DateTime<chrono::Utc>>,
    dry_run: bool,
    repair: bool,
    batch_size: i64,
}

struct RequestClassificationReportOptions {
    repo: Option<String>,
    since: Option<chrono::DateTime<chrono::Utc>>,
}

impl HarnessFeedbackOptions {
    fn parse(args: Vec<String>) -> Result<Self, anyhow::Error> {
        let mut repo = None;
        let mut session_id = None;
        let mut since = None;
        let mut dry_run = false;
        let mut batch_size = DEFAULT_BATCH_SIZE;
        let mut idx = 0usize;
        while idx < args.len() {
            match args[idx].as_str() {
                "--repo" | "–repo" => {
                    let Some(value) = args.get(idx + 1) else {
                        anyhow::bail!("--repo requires a value");
                    };
                    repo = Some(value.clone());
                    idx += 2;
                }
                "--session" | "–session" => {
                    let Some(value) = args.get(idx + 1) else {
                        anyhow::bail!("--session requires a value");
                    };
                    session_id = Some(value.clone());
                    idx += 2;
                }
                "--since" | "–since" => {
                    let Some(value) = args.get(idx + 1) else {
                        anyhow::bail!("--since requires an RFC3339 timestamp");
                    };
                    since = Some(
                        chrono::DateTime::parse_from_rfc3339(value)?.with_timezone(&chrono::Utc),
                    );
                    idx += 2;
                }
                "--dry-run" | "–dry-run" => {
                    dry_run = true;
                    idx += 1;
                }
                "--batch-size" | "–batch-size" => {
                    let Some(value) = args.get(idx + 1) else {
                        anyhow::bail!("--batch-size requires a positive integer");
                    };
                    batch_size = value.parse::<i64>()?;
                    if batch_size <= 0 {
                        anyhow::bail!("--batch-size must be positive");
                    }
                    idx += 2;
                }
                other => anyhow::bail!("unknown option: {other}"),
            }
        }

        Ok(Self {
            repo,
            session_id,
            since,
            dry_run,
            batch_size,
        })
    }
}

impl RequestClassificationOptions {
    fn parse(args: Vec<String>) -> Result<Self, anyhow::Error> {
        let mut repo = None;
        let mut session_id = None;
        let mut since = None;
        let mut dry_run = false;
        let mut repair = false;
        let mut batch_size = DEFAULT_BATCH_SIZE;
        let mut idx = 0usize;
        while idx < args.len() {
            match args[idx].as_str() {
                "--repo" | "–repo" => {
                    let Some(value) = args.get(idx + 1) else {
                        anyhow::bail!("--repo requires a value");
                    };
                    repo = Some(value.clone());
                    idx += 2;
                }
                "--session" | "–session" => {
                    let Some(value) = args.get(idx + 1) else {
                        anyhow::bail!("--session requires a value");
                    };
                    session_id = Some(value.clone());
                    idx += 2;
                }
                "--since" | "–since" => {
                    let Some(value) = args.get(idx + 1) else {
                        anyhow::bail!("--since requires an RFC3339 timestamp");
                    };
                    since = Some(
                        chrono::DateTime::parse_from_rfc3339(value)?.with_timezone(&chrono::Utc),
                    );
                    idx += 2;
                }
                "--dry-run" | "–dry-run" => {
                    dry_run = true;
                    idx += 1;
                }
                "--repair" | "–repair" => {
                    repair = true;
                    idx += 1;
                }
                "--batch-size" | "–batch-size" => {
                    let Some(value) = args.get(idx + 1) else {
                        anyhow::bail!("--batch-size requires a positive integer");
                    };
                    batch_size = value.parse::<i64>()?;
                    if batch_size <= 0 {
                        anyhow::bail!("--batch-size must be positive");
                    }
                    idx += 2;
                }
                other => anyhow::bail!("unknown option: {other}"),
            }
        }

        Ok(Self {
            repo,
            session_id,
            since,
            dry_run,
            repair,
            batch_size,
        })
    }
}

impl RequestClassificationReportOptions {
    fn parse(args: Vec<String>) -> Result<Self, anyhow::Error> {
        let mut repo = None;
        let mut since = None;
        let mut idx = 0usize;
        while idx < args.len() {
            match args[idx].as_str() {
                "--repo" | "–repo" => {
                    let Some(value) = args.get(idx + 1) else {
                        anyhow::bail!("--repo requires a value");
                    };
                    repo = Some(value.clone());
                    idx += 2;
                }
                "--since" | "–since" => {
                    let Some(value) = args.get(idx + 1) else {
                        anyhow::bail!("--since requires an RFC3339 timestamp");
                    };
                    since = Some(
                        chrono::DateTime::parse_from_rfc3339(value)?.with_timezone(&chrono::Utc),
                    );
                    idx += 2;
                }
                other => anyhow::bail!("unknown option: {other}"),
            }
        }

        Ok(Self { repo, since })
    }
}

fn print_request_classification_report(
    report: &request_classification::RequestClassificationReport,
) {
    println!("request-classification-report:");
    println!("by_route:");
    for row in &report.by_route {
        println!("  {} {}", row.label, row.count);
    }
    println!("top_risk_flags:");
    for row in &report.top_risk_flags {
        println!("  {} {}", row.label, row.count);
    }
    println!("unknown_label_counts:");
    for row in &report.unknown_label_counts {
        println!("  {} {}", row.label, row.count);
    }
    println!("repeated_guardrail_sessions:");
    for row in &report.repeated_guardrail_sessions {
        println!("  {} {}", row.session_id, row.count);
    }
}

impl ExtractFeaturesOptions {
    fn parse(args: Vec<String>) -> Result<Self, anyhow::Error> {
        let mut repo = None;
        let mut session_id = None;
        let mut trajectory_id = None;
        let mut since = None;
        let mut dry_run = false;
        let mut batch_size = DEFAULT_BATCH_SIZE;
        let mut skip_bootstrap_tagging = false;
        let mut idx = 0usize;
        while idx < args.len() {
            match args[idx].as_str() {
                "--repo" | "–repo" => {
                    let Some(value) = args.get(idx + 1) else {
                        anyhow::bail!("--repo requires a value");
                    };
                    repo = Some(value.clone());
                    idx += 2;
                }
                "--session" | "–session" => {
                    let Some(value) = args.get(idx + 1) else {
                        anyhow::bail!("--session requires a value");
                    };
                    session_id = Some(value.clone());
                    idx += 2;
                }
                "--trajectory" | "–trajectory" => {
                    let Some(value) = args.get(idx + 1) else {
                        anyhow::bail!("--trajectory requires a UUID");
                    };
                    trajectory_id = Some(value.parse::<Uuid>()?);
                    idx += 2;
                }
                "--since" | "–since" => {
                    let Some(value) = args.get(idx + 1) else {
                        anyhow::bail!("--since requires an RFC3339 timestamp");
                    };
                    since = Some(
                        chrono::DateTime::parse_from_rfc3339(value)?.with_timezone(&chrono::Utc),
                    );
                    idx += 2;
                }
                "--dry-run" | "–dry-run" => {
                    dry_run = true;
                    idx += 1;
                }
                "--batch-size" | "–batch-size" => {
                    let Some(value) = args.get(idx + 1) else {
                        anyhow::bail!("--batch-size requires a positive integer");
                    };
                    batch_size = value.parse::<i64>()?;
                    if batch_size <= 0 {
                        anyhow::bail!("--batch-size must be positive");
                    }
                    idx += 2;
                }
                "--skip-bootstrap-tagging" | "–skip-bootstrap-tagging" => {
                    skip_bootstrap_tagging = true;
                    idx += 1;
                }
                other => anyhow::bail!("unknown option: {other}"),
            }
        }

        Ok(Self {
            repo,
            session_id,
            trajectory_id,
            since,
            dry_run,
            batch_size,
            skip_bootstrap_tagging,
        })
    }
}

impl BackfillOptions {
    fn parse(args: Vec<String>) -> Result<Self, anyhow::Error> {
        let mut dry_run = false;
        let mut batch_size = DEFAULT_BATCH_SIZE;
        let mut idx = 0usize;
        while idx < args.len() {
            match args[idx].as_str() {
                "--dry-run" | "–dry-run" => {
                    dry_run = true;
                    idx += 1;
                }
                "--batch-size" | "–batch-size" => {
                    let Some(value) = args.get(idx + 1) else {
                        anyhow::bail!("--batch-size requires a positive integer");
                    };
                    batch_size = value.parse::<i64>()?;
                    if batch_size <= 0 {
                        anyhow::bail!("--batch-size must be positive");
                    }
                    idx += 2;
                }
                other => anyhow::bail!("unknown option: {other}"),
            }
        }

        Ok(Self {
            dry_run,
            batch_size,
        })
    }
}

struct BackfillReport {
    pending: i64,
    updated: i64,
    dry_run: bool,
    batch_size: i64,
}

async fn backfill_signatures(
    pool: &deadpool_postgres::Pool,
    opts: BackfillOptions,
) -> Result<BackfillReport, anyhow::Error> {
    let pending = count_pending_backfill(pool).await?;
    if opts.dry_run || pending == 0 {
        return Ok(BackfillReport {
            pending,
            updated: 0,
            dry_run: opts.dry_run,
            batch_size: opts.batch_size,
        });
    }

    let mut updated = 0i64;
    loop {
        let batch = load_backfill_batch(pool, opts.batch_size).await?;
        if batch.is_empty() {
            break;
        }

        let mut conn = pool.get().await?;
        let tx = conn.deref_mut().deref_mut().transaction().await?;
        for row in batch {
            let Some(updated_metadata) = execution_feedback::backfill_inline_signature_metadata(
                &row.metadata,
                row.legacy_raw_excerpt.as_deref(),
            ) else {
                continue;
            };
            let changed = tx
                .execute(
                    "UPDATE agent_events
                     SET metadata = $2
                     WHERE id = $1
                       AND (
                           metadata->'payload'->>'signature' IS NULL
                           OR metadata->'payload'->>'signature_category' IS NULL
                           OR metadata->'payload'->>'fingerprint_version' IS NULL
                       )",
                    &[&row.id, &updated_metadata],
                )
                .await?;
            updated += changed as i64;
        }
        tx.commit().await?;
    }

    Ok(BackfillReport {
        pending,
        updated,
        dry_run: opts.dry_run,
        batch_size: opts.batch_size,
    })
}

async fn count_pending_backfill(pool: &deadpool_postgres::Pool) -> Result<i64, anyhow::Error> {
    let conn = pool.get().await?;
    let outcome_event_types = execution_feedback::FAILURE_OUTCOME_EVENT_TYPES.to_vec();
    let row = conn
        .query_one(
            "SELECT count(*)::BIGINT AS count
             FROM agent_events
             WHERE event_type = ANY($1)
               AND metadata->>'success' = 'false'
               AND (
                   metadata->'payload'->>'signature' IS NULL
                   OR metadata->'payload'->>'signature_category' IS NULL
                   OR metadata->'payload'->>'fingerprint_version' IS NULL
               )",
            &[&outcome_event_types],
        )
        .await?;
    Ok(row.get("count"))
}

struct BackfillRow {
    id: String,
    metadata: serde_json::Value,
    legacy_raw_excerpt: Option<String>,
}

async fn load_backfill_batch(
    pool: &deadpool_postgres::Pool,
    batch_size: i64,
) -> Result<Vec<BackfillRow>, anyhow::Error> {
    let conn = pool.get().await?;
    let outcome_event_types = execution_feedback::FAILURE_OUTCOME_EVENT_TYPES.to_vec();
    let rows = conn
        .query(
            "SELECT e.id,
                    e.metadata,
                    fs.metadata->'payload'->>'raw_excerpt' AS legacy_raw_excerpt
             FROM agent_events e
             LEFT JOIN LATERAL (
                 SELECT metadata
                 FROM agent_events fs
                 WHERE fs.event_type = 'failure_signature'
                   AND fs.parent_event_id::text = e.id
                 ORDER BY fs.created_at DESC
                 LIMIT 1
             ) fs ON true
             WHERE e.event_type = ANY($1)
               AND e.metadata->>'success' = 'false'
               AND (
                   e.metadata->'payload'->>'signature' IS NULL
                   OR e.metadata->'payload'->>'signature_category' IS NULL
                   OR e.metadata->'payload'->>'fingerprint_version' IS NULL
               )
             ORDER BY e.created_at ASC, e.id ASC
             LIMIT $2",
            &[&outcome_event_types, &batch_size],
        )
        .await?;

    Ok(rows
        .into_iter()
        .map(|row| BackfillRow {
            id: row.get("id"),
            metadata: row.get("metadata"),
            legacy_raw_excerpt: row.get("legacy_raw_excerpt"),
        })
        .collect())
}
