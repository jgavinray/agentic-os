use orchestrator::{
    db, feature_extraction, harness_feedback, logging, migrations, request_classification,
};
use std::env;
use uuid::Uuid;

#[path = "orchestrator-maint/signature_backfill.rs"]
mod signature_backfill;

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
            let report = signature_backfill::backfill_signatures(&pool, opts).await?;
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
