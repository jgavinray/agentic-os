use std::env;

use crate::state::AppState;
use crate::{feature_extraction, request_classification, telemetry};

pub use crate::startup::harness_feedback::{
    run_startup_harness_feedback_backfill, spawn_harness_feedback_background_repair,
};

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

pub(crate) fn env_flag(name: &str, default: bool) -> bool {
    env::var(name)
        .map(|v| {
            !matches!(
                v.to_ascii_lowercase().as_str(),
                "0" | "false" | "no" | "off"
            )
        })
        .unwrap_or(default)
}

pub(crate) fn positive_i64_env(name: &str, default: i64) -> i64 {
    env::var(name)
        .ok()
        .and_then(|value| value.parse::<i64>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(default)
}
