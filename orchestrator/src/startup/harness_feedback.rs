use std::sync::Arc;
use std::time::Duration;

use crate::startup::backfill::{env_flag, positive_i64_env};
use crate::state::AppState;
use crate::{harness_feedback, telemetry};

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
