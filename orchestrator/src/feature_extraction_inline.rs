use chrono::Utc;
use deadpool_postgres::Pool;
use uuid::Uuid;

use crate::feature_extraction::extract_records;
use crate::feature_extraction_persistence::persist_feature_record;
use crate::feature_extraction_store::load_events_for_scope;
use crate::feature_extraction_types::{
    constraint_freshness_window_sec_from_env, feature_window_sec_from_env,
    max_operational_constraints_from_env, ExtractionConfig, ExtractionScope,
};

pub async fn run_inline_extraction_best_effort(
    pool: Pool,
    metrics: crate::telemetry::MetricsRegistry,
    repo: String,
    session_id: String,
    trajectory_id: Option<Uuid>,
) {
    let started = std::time::Instant::now();
    let config = ExtractionConfig {
        feature_window_sec: feature_window_sec_from_env(),
        constraint_freshness_window_sec: constraint_freshness_window_sec_from_env(),
        max_operational_constraints: max_operational_constraints_from_env(),
        evaluation_time: Utc::now(),
    };
    let scope = ExtractionScope {
        repo: Some(repo.clone()),
        session_id: Some(session_id),
        trajectory_id,
        since: None,
        until: None,
    };

    let events = match load_events_for_scope(&pool, &scope, None).await {
        Ok(events) => events,
        Err(e) => {
            tracing::warn!(
                target: "feature_extraction",
                repo = %repo,
                stage = "extraction",
                "inline feature extraction failed to load events: {e}"
            );
            crate::telemetry::record_feature_extraction_failure("extraction");
            return;
        }
    };
    let report = extract_records(&events, &config);
    for _ in 0..report.unknown_tag_schema_versions {
        crate::telemetry::record_unknown_tag_schema_version();
    }
    for record in &report.records {
        for class in &record.failure_classes {
            crate::telemetry::record_feature_failure_class(class);
        }
        if let Err(e) = persist_feature_record(&pool, record).await {
            tracing::warn!(
                target: "feature_extraction",
                repo = %repo,
                stage = "persistence",
                "inline feature extraction failed to persist record: {e}"
            );
            crate::telemetry::record_feature_extraction_failure("persistence");
            continue;
        }
        crate::telemetry::record_trajectory_feature_written(&metrics);
    }
    crate::telemetry::record_feature_extraction_duration(started.elapsed());
    drop(metrics);
}
