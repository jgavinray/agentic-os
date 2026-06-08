use orchestrator::{feature_extraction, harness_feedback, request_classification};

use super::commands_report::print_request_classification_report;
use super::commands_runtime::{execution_feedback_enabled, open_migrated_pool};
use super::options::{
    BackfillOptions, ExtractFeaturesOptions, HarnessFeedbackOptions, RequestClassificationOptions,
    RequestClassificationReportOptions,
};
use super::signature_backfill;

pub(super) async fn run_command(command: &str, args: Vec<String>) -> Result<(), anyhow::Error> {
    match command {
        "backfill-signatures" => backfill_signatures(args).await,
        "extract-features" => extract_features(args).await,
        "classify-harness-feedback" => classify_harness_feedback(args).await,
        "classify-requests" => classify_requests(args).await,
        "request-classification-report" => request_classification_report(args).await,
        _ => {
            print_usage();
            anyhow::bail!("unknown command: {command}");
        }
    }
}

pub(super) fn print_usage() {
    eprintln!(
        "usage: orchestrator-maint backfill-signatures [--dry-run] [--batch-size N]\n       orchestrator-maint extract-features [--repo REPO] [--session SESSION] [--trajectory TRAJECTORY] [--since TIMESTAMP] [--dry-run] [--batch-size N] [--skip-bootstrap-tagging]\n       orchestrator-maint classify-harness-feedback [--repo REPO] [--session SESSION] [--since TIMESTAMP] [--dry-run] [--batch-size N]\n       orchestrator-maint classify-requests [--repo REPO] [--session SESSION] [--since TIMESTAMP] [--dry-run] [--repair] [--batch-size N]\n       orchestrator-maint request-classification-report [--repo REPO] [--since TIMESTAMP]"
    );
}

async fn backfill_signatures(args: Vec<String>) -> Result<(), anyhow::Error> {
    let opts = BackfillOptions::parse(args)?;
    if !execution_feedback_enabled() {
        println!("EXECUTION_FEEDBACK_ENABLED=false; skipping signature backfill with no changes");
        return Ok(());
    }
    let pool = open_migrated_pool().await?;
    let report = signature_backfill::backfill_signatures(&pool, opts).await?;
    println!(
        "backfill-signatures: pending={} updated={} dry_run={} batch_size={}",
        report.pending, report.updated, report.dry_run, report.batch_size
    );
    Ok(())
}

async fn extract_features(args: Vec<String>) -> Result<(), anyhow::Error> {
    let opts = ExtractFeaturesOptions::parse(args)?;
    let pool = open_migrated_pool().await?;
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
            max_operational_constraints: feature_extraction::max_operational_constraints_from_env(),
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

async fn classify_harness_feedback(args: Vec<String>) -> Result<(), anyhow::Error> {
    let opts = HarnessFeedbackOptions::parse(args)?;
    let pool = open_migrated_pool().await?;
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

async fn classify_requests(args: Vec<String>) -> Result<(), anyhow::Error> {
    let opts = RequestClassificationOptions::parse(args)?;
    let pool = open_migrated_pool().await?;
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

async fn request_classification_report(args: Vec<String>) -> Result<(), anyhow::Error> {
    let opts = RequestClassificationReportOptions::parse(args)?;
    let pool = open_migrated_pool().await?;
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
