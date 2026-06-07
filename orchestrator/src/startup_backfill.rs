use std::env;
use std::sync::Arc;
use std::time::Duration;

use crate::state::AppState;
use crate::{feature_extraction, harness_feedback, request_classification, telemetry};

pub async fn run_startup_feature_backfill(state: &AppState) -> Result<(), anyhow::Error> {
    if !state.feature_extraction_enabled {
        tracing::info!(
            target: "feature_extraction",
            "startup feature backfill skipped because FEATURE_EXTRACTION_ENABLED=false"
        );
        return Ok(());
    }
    if !env_flag("FEATURE_STARTUP_BACKFILL_ENABLED", true) {
        tracing::info!(
            target: "feature_extraction",
            "startup feature backfill disabled by FEATURE_STARTUP_BACKFILL_ENABLED=false"
        );
        return Ok(());
    }

    let batch_size = positive_i64_env("FEATURE_STARTUP_BACKFILL_BATCH_SIZE", 500);
    let skip_bootstrap_tagging = env_flag("FEATURE_STARTUP_SKIP_BOOTSTRAP_TAGGING", false);
    let started = std::time::Instant::now();
    tracing::info!(
        target: "feature_extraction",
        batch_size,
        skip_bootstrap_tagging,
        "startup feature backfill started"
    );
    let report = feature_extraction::run_backfill(
        &state.pool,
        &feature_extraction::BackfillOptions {
            repo: None,
            session_id: None,
            trajectory_id: None,
            since: None,
            dry_run: false,
            batch_size,
            skip_bootstrap_tagging,
        },
        &feature_extraction::ExtractionConfig {
            feature_window_sec: feature_extraction::feature_window_sec_from_env(),
            constraint_freshness_window_sec:
                feature_extraction::constraint_freshness_window_sec_from_env(),
            max_operational_constraints: feature_extraction::max_operational_constraints_from_env(),
            evaluation_time: chrono::Utc::now(),
        },
    )
    .await
    .map_err(|e| {
        telemetry::record_feature_extraction_failure("extraction");
        tracing::error!(
            target: "feature_extraction",
            "startup feature backfill failed; refusing to serve traffic: {e}"
        );
        e
    })?;

    for _ in 0..report.persisted_records {
        telemetry::record_trajectory_feature_written(&state.metrics);
    }
    telemetry::record_feature_extraction_duration(started.elapsed());
    tracing::info!(
        target: "feature_extraction",
        events_scanned = report.events_scanned,
        bootstrap_updated = report.bootstrap_updated,
        feature_records = report.feature_records,
        persisted_records = report.persisted_records,
        elapsed_ms = started.elapsed().as_millis(),
        "startup feature backfill completed"
    );
    Ok(())
}

pub async fn run_startup_request_classification_backfill(
    state: &AppState,
) -> Result<(), anyhow::Error> {
    if !request_classification::request_classification_startup_backfill_enabled() {
        tracing::info!(
            target: "request_classification",
            "startup request classification backfill disabled by REQUEST_CLASSIFICATION_STARTUP_BACKFILL_ENABLED=false"
        );
        return Ok(());
    }

    let batch_size = request_classification::request_classification_startup_batch_size();
    let started = std::time::Instant::now();
    tracing::info!(
        target: "request_classification",
        batch_size,
        "startup request classification backfill started"
    );
    let report = request_classification::run_backfill(
        &state.pool,
        &request_classification::BackfillOptions {
            repo: None,
            session_id: None,
            since: None,
            dry_run: false,
            repair: false,
            repair_stale: true,
            batch_size,
        },
    )
    .await
    .map_err(|e| {
        tracing::error!(
            target: "request_classification",
            "startup request classification backfill failed; refusing to serve traffic: {e}"
        );
        e
    })?;

    tracing::info!(
        target: "request_classification",
        events_scanned = report.events_scanned,
        inserted = report.inserted,
        updated = report.updated,
        skipped = report.skipped,
        elapsed_ms = started.elapsed().as_millis(),
        "startup request classification backfill completed"
    );
    Ok(())
}

pub async fn run_startup_harness_feedback_backfill(state: &AppState) {
    if !env_flag("HARNESS_FEEDBACK_STARTUP_BACKFILL_ENABLED", true) {
        tracing::info!(
            target: "harness_feedback",
            "startup harness feedback repair disabled by HARNESS_FEEDBACK_STARTUP_BACKFILL_ENABLED=false"
        );
        return;
    }

    let batch_size = positive_i64_env("HARNESS_FEEDBACK_STARTUP_BACKFILL_BATCH_SIZE", 500);
    let started = std::time::Instant::now();
    tracing::info!(
        target: "harness_feedback",
        batch_size,
        "startup harness feedback repair started"
    );
    match harness_feedback::run_backfill(
        &state.pool,
        &harness_feedback::BackfillOptions {
            repo: None,
            session_id: None,
            since: None,
            dry_run: false,
            batch_size,
        },
    )
    .await
    {
        Ok(report) => {
            telemetry::record_harness_feedback_repair_run("success");
            tracing::info!(
                target: "harness_feedback",
                events_scanned = report.events_scanned,
                updated = report.updated,
                quarantined = report.quarantined,
                elapsed_ms = started.elapsed().as_millis(),
                "startup harness feedback repair completed"
            );
        }
        Err(e) => {
            telemetry::record_harness_feedback_repair_run("failure");
            tracing::warn!(
                target: "harness_feedback",
                "startup harness feedback repair failed; continuing because live classification remains active: {e}"
            );
        }
    }
}

pub fn spawn_harness_feedback_background_repair(state: Arc<AppState>) {
    if !env_flag("HARNESS_FEEDBACK_BACKGROUND_REPAIR_ENABLED", true) {
        tracing::info!(
            target: "harness_feedback",
            "background harness feedback repair disabled by HARNESS_FEEDBACK_BACKGROUND_REPAIR_ENABLED=false"
        );
        return;
    }

    let interval_sec = positive_i64_env("HARNESS_FEEDBACK_REPAIR_INTERVAL_SEC", 300) as u64;
    let lookback_sec = positive_i64_env(
        "HARNESS_FEEDBACK_REPAIR_LOOKBACK_SEC",
        (interval_sec.saturating_mul(2)).max(60) as i64,
    );
    let batch_size = positive_i64_env("HARNESS_FEEDBACK_REPAIR_BATCH_SIZE", 500);
    tracing::info!(
        target: "harness_feedback",
        interval_sec,
        lookback_sec,
        batch_size,
        "background harness feedback repair enabled"
    );

    tokio::spawn(async move {
        loop {
            tokio::time::sleep(Duration::from_secs(interval_sec)).await;
            let since = chrono::Utc::now() - chrono::Duration::seconds(lookback_sec);
            let started = std::time::Instant::now();
            match harness_feedback::run_backfill(
                &state.pool,
                &harness_feedback::BackfillOptions {
                    repo: None,
                    session_id: None,
                    since: Some(since),
                    dry_run: false,
                    batch_size,
                },
            )
            .await
            {
                Ok(report) => {
                    telemetry::record_harness_feedback_repair_run("success");
                    tracing::info!(
                        target: "harness_feedback",
                        events_scanned = report.events_scanned,
                        updated = report.updated,
                        quarantined = report.quarantined,
                        elapsed_ms = started.elapsed().as_millis(),
                        "background harness feedback repair completed"
                    );
                }
                Err(e) => {
                    telemetry::record_harness_feedback_repair_run("failure");
                    tracing::warn!(
                        target: "harness_feedback",
                        "background harness feedback repair failed: {e}"
                    );
                }
            }
        }
    });
}

fn env_flag(name: &str, default: bool) -> bool {
    env::var(name)
        .map(|v| {
            !matches!(
                v.to_ascii_lowercase().as_str(),
                "0" | "false" | "no" | "off"
            )
        })
        .unwrap_or(default)
}

fn positive_i64_env(name: &str, default: i64) -> i64 {
    env::var(name)
        .ok()
        .and_then(|value| value.parse::<i64>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(default)
}
