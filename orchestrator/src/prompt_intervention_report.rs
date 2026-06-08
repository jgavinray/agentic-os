use chrono::{DateTime, Utc};
use deadpool_postgres::Pool;
use uuid::Uuid;

const HEADLINE_CONFIDENCE_THRESHOLD: f64 = 0.8;
const HEADLINE_ELIGIBLE_PREDICATE: &str = "pi.confidence >= $4
               AND NOT EXISTS (
                   SELECT 1
                   FROM prompt_interventions newer_pi
                   WHERE newer_pi.supersedes_record_id = pi.id
               )";

#[derive(Debug, Clone)]
pub struct ReportOptions {
    pub repo: Option<String>,
    pub since: Option<DateTime<Utc>>,
    pub until: Option<DateTime<Utc>>,
    pub limit: i64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CountRow {
    pub label: String,
    pub count: i64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct OutcomeCorrelation {
    pub accepted_results: i64,
    pub accepted_results_with_intervention: i64,
    pub accepted_results_without_intervention: i64,
    pub interventions_on_accepted_results: i64,
    pub interventions_per_accepted_result: Option<f64>,
    pub accepted_without_intervention_rate: Option<f64>,
    pub unavailable_reason: Option<String>,
}

impl OutcomeCorrelation {
    pub fn unavailable(reason: impl Into<String>) -> Self {
        Self {
            accepted_results: 0,
            accepted_results_with_intervention: 0,
            accepted_results_without_intervention: 0,
            interventions_on_accepted_results: 0,
            interventions_per_accepted_result: None,
            accepted_without_intervention_rate: None,
            unavailable_reason: Some(reason.into()),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct PromptInterventionReport {
    pub interventions_per_day: Vec<CountRow>,
    pub by_type: Vec<CountRow>,
    pub by_signal_family: Vec<CountRow>,
    pub by_burden_type: Vec<CountRow>,
    pub by_failure_relation: Vec<CountRow>,
    pub by_requested_model: Vec<CountRow>,
    pub by_routed_model: Vec<CountRow>,
    pub by_baseline_arm: Vec<CountRow>,
    pub top_exact_prompt_hashes: Vec<CountRow>,
    pub top_normalized_prompt_hashes: Vec<CountRow>,
    pub per_trajectory: Vec<CountRow>,
    pub outcome_correlation: Option<OutcomeCorrelation>,
}

pub async fn prompt_intervention_report(
    capture_pool: &Pool,
    outcome_pool: Option<&Pool>,
    opts: &ReportOptions,
) -> Result<PromptInterventionReport, anyhow::Error> {
    let capture_conn = capture_pool.get().await?;
    let interventions_per_day = count_per_day(&capture_conn, opts).await?;
    let by_type = count_grouped(&capture_conn, "intervention_type", opts).await?;
    let by_signal_family = count_grouped(&capture_conn, "signal_family", opts).await?;
    let by_burden_type = count_grouped(&capture_conn, "burden_type", opts).await?;
    let by_failure_relation = count_grouped(&capture_conn, "failure_relation", opts).await?;
    let by_requested_model = count_grouped(&capture_conn, "requested_model", opts).await?;
    let by_routed_model = count_grouped(&capture_conn, "routed_model", opts).await?;
    let by_baseline_arm = count_grouped(&capture_conn, "baseline_arm", opts).await?;
    let top_exact_prompt_hashes = count_grouped(&capture_conn, "exact_prompt_hash", opts).await?;
    let top_normalized_prompt_hashes =
        count_grouped(&capture_conn, "normalized_prompt_hash", opts).await?;
    let per_trajectory = count_grouped(&capture_conn, "trajectory_id", opts).await?;
    let outcome_correlation = match outcome_pool {
        Some(pool) => Some(outcome_correlation(&capture_conn, pool, opts).await?),
        None => None,
    };

    Ok(PromptInterventionReport {
        interventions_per_day,
        by_type,
        by_signal_family,
        by_burden_type,
        by_failure_relation,
        by_requested_model,
        by_routed_model,
        by_baseline_arm,
        top_exact_prompt_hashes,
        top_normalized_prompt_hashes,
        per_trajectory,
        outcome_correlation,
    })
}

async fn count_per_day(
    conn: &deadpool_postgres::Object,
    opts: &ReportOptions,
) -> Result<Vec<CountRow>, anyhow::Error> {
    let rows = conn
        .query(
            "SELECT to_char(date_trunc('day', pi.created_at), 'YYYY-MM-DD') AS label,
                    count(*)::BIGINT AS count
             FROM prompt_interventions pi
             WHERE ($1::TIMESTAMPTZ IS NULL OR pi.created_at >= $1)
               AND ($2::TIMESTAMPTZ IS NULL OR pi.created_at <= $2)
               AND ($3::TEXT IS NULL OR pi.exchange_id IN (
                   SELECT exchange_id FROM raw_http_exchanges WHERE repo = $3
               ))
               AND pi.confidence >= $4
               AND NOT EXISTS (
                   SELECT 1
                   FROM prompt_interventions newer_pi
                   WHERE newer_pi.supersedes_record_id = pi.id
               )
             GROUP BY date_trunc('day', pi.created_at)
             ORDER BY date_trunc('day', pi.created_at) DESC
             LIMIT $5",
            &[
                &opts.since,
                &opts.until,
                &opts.repo,
                &HEADLINE_CONFIDENCE_THRESHOLD,
                &opts.limit,
            ],
        )
        .await?;
    Ok(rows
        .into_iter()
        .map(|row| CountRow {
            label: row.get("label"),
            count: row.get("count"),
        })
        .collect())
}

async fn count_grouped(
    conn: &deadpool_postgres::Object,
    column: &str,
    opts: &ReportOptions,
) -> Result<Vec<CountRow>, anyhow::Error> {
    let qualified_expr = report_expression_for_alias(column, "pi")?;
    let sql = format!(
        "SELECT {qualified_expr} AS label, count(*)::BIGINT AS count
         FROM prompt_interventions pi
         WHERE ($1::TIMESTAMPTZ IS NULL OR pi.created_at >= $1)
           AND ($2::TIMESTAMPTZ IS NULL OR pi.created_at <= $2)
           AND ($3::TEXT IS NULL OR pi.exchange_id IN (
               SELECT exchange_id FROM raw_http_exchanges WHERE repo = $3
           ))
           AND {qualified_expr} IS NOT NULL
           AND {HEADLINE_ELIGIBLE_PREDICATE}
         GROUP BY {qualified_expr}
         ORDER BY count DESC, {qualified_expr} ASC
         LIMIT $5"
    );
    let rows = conn
        .query(
            &sql,
            &[
                &opts.since,
                &opts.until,
                &opts.repo,
                &HEADLINE_CONFIDENCE_THRESHOLD,
                &opts.limit,
            ],
        )
        .await?;
    Ok(rows
        .into_iter()
        .map(|row| CountRow {
            label: row.get("label"),
            count: row.get("count"),
        })
        .collect())
}

fn report_expression_for_alias(column: &str, alias: &str) -> Result<String, anyhow::Error> {
    let expr = match column {
        "intervention_type" => format!("{alias}.intervention_type"),
        "signal_family" => format!("{alias}.signal_family"),
        "burden_type" => format!("{alias}.burden_type"),
        "failure_relation" => format!("{alias}.failure_relation"),
        "requested_model" => format!("{alias}.requested_model"),
        "routed_model" => format!("{alias}.routed_model"),
        "baseline_arm" => format!("{alias}.baseline_arm"),
        "exact_prompt_hash" => format!("{alias}.exact_prompt_hash"),
        "normalized_prompt_hash" => format!("{alias}.normalized_prompt_hash"),
        "trajectory_id" => format!("{alias}.trajectory_id::TEXT"),
        _ => anyhow::bail!("unsupported prompt intervention report column"),
    };
    Ok(expr)
}

async fn outcome_correlation(
    capture_conn: &deadpool_postgres::Object,
    outcome_pool: &Pool,
    opts: &ReportOptions,
) -> Result<OutcomeCorrelation, anyhow::Error> {
    if opts.repo.is_some() {
        return Ok(OutcomeCorrelation::unavailable(
            "repo-scoped accepted-without-intervention rate requires outcome repo attribution",
        ));
    }

    let outcome_conn = outcome_pool.get().await?;
    let accepted_rows = outcome_conn
        .query(
            "SELECT DISTINCT trajectory_id
             FROM harness_outcome_events
             WHERE accepted = true
               AND ($1::TIMESTAMPTZ IS NULL OR created_at >= $1)
               AND ($2::TIMESTAMPTZ IS NULL OR created_at <= $2)",
            &[&opts.since, &opts.until],
        )
        .await?;
    let accepted_trajectory_ids: Vec<Uuid> = accepted_rows
        .into_iter()
        .map(|row| row.get("trajectory_id"))
        .collect();
    if accepted_trajectory_ids.is_empty() {
        return Ok(OutcomeCorrelation {
            accepted_results: 0,
            accepted_results_with_intervention: 0,
            accepted_results_without_intervention: 0,
            interventions_on_accepted_results: 0,
            interventions_per_accepted_result: None,
            accepted_without_intervention_rate: None,
            unavailable_reason: None,
        });
    }

    let row = capture_conn
        .query_one(
            "SELECT count(*)::BIGINT AS intervention_count,
                    count(DISTINCT trajectory_id)::BIGINT AS trajectory_count
             FROM prompt_interventions pi
             WHERE pi.trajectory_id = ANY($1::UUID[])
               AND ($2::TIMESTAMPTZ IS NULL OR pi.created_at >= $2)
               AND ($3::TIMESTAMPTZ IS NULL OR pi.created_at <= $3)
               AND pi.confidence >= $4
               AND NOT EXISTS (
                   SELECT 1
                   FROM prompt_interventions newer_pi
                   WHERE newer_pi.supersedes_record_id = pi.id
               )",
            &[
                &accepted_trajectory_ids,
                &opts.since,
                &opts.until,
                &HEADLINE_CONFIDENCE_THRESHOLD,
            ],
        )
        .await?;
    Ok(correlation_from_counts(
        i64::try_from(accepted_trajectory_ids.len())?,
        row.get("trajectory_count"),
        row.get("intervention_count"),
    ))
}

fn correlation_from_counts(
    accepted_results: i64,
    accepted_results_with_intervention: i64,
    interventions_on_accepted_results: i64,
) -> OutcomeCorrelation {
    let accepted_results_without_intervention =
        accepted_results.saturating_sub(accepted_results_with_intervention);
    let interventions_per_accepted_result = if accepted_results > 0 {
        Some(interventions_on_accepted_results as f64 / accepted_results as f64)
    } else {
        None
    };
    let accepted_without_intervention_rate = if accepted_results > 0 {
        Some(accepted_results_without_intervention as f64 / accepted_results as f64)
    } else {
        None
    };
    OutcomeCorrelation {
        accepted_results,
        accepted_results_with_intervention,
        accepted_results_without_intervention,
        interventions_on_accepted_results,
        interventions_per_accepted_result,
        accepted_without_intervention_rate,
        unavailable_reason: None,
    }
}

pub fn report_lines(report: &PromptInterventionReport) -> Vec<String> {
    let mut lines = vec!["prompt-intervention-report:".to_string()];
    push_section(
        &mut lines,
        "interventions_per_day",
        &report.interventions_per_day,
    );
    push_section(&mut lines, "by_type", &report.by_type);
    push_section(&mut lines, "by_signal_family", &report.by_signal_family);
    push_section(&mut lines, "by_burden_type", &report.by_burden_type);
    push_section(
        &mut lines,
        "by_failure_relation",
        &report.by_failure_relation,
    );
    push_section(&mut lines, "by_requested_model", &report.by_requested_model);
    push_section(&mut lines, "by_routed_model", &report.by_routed_model);
    push_section(&mut lines, "by_baseline_arm", &report.by_baseline_arm);
    push_section(
        &mut lines,
        "top_exact_prompt_hashes",
        &report.top_exact_prompt_hashes,
    );
    push_section(
        &mut lines,
        "top_normalized_prompt_hashes",
        &report.top_normalized_prompt_hashes,
    );
    push_section(&mut lines, "per_trajectory", &report.per_trajectory);
    if let Some(correlation) = &report.outcome_correlation {
        lines.push("outcome_correlation:".to_string());
        if let Some(reason) = &correlation.unavailable_reason {
            lines.push(format!("  unavailable_reason {reason}"));
        } else {
            lines.push(format!(
                "  accepted_results {}",
                correlation.accepted_results
            ));
            lines.push(format!(
                "  accepted_results_with_intervention {}",
                correlation.accepted_results_with_intervention
            ));
            lines.push(format!(
                "  accepted_results_without_intervention {}",
                correlation.accepted_results_without_intervention
            ));
            lines.push(format!(
                "  interventions_on_accepted_results {}",
                correlation.interventions_on_accepted_results
            ));
            if let Some(rate) = correlation.accepted_without_intervention_rate {
                lines.push(format!("  accepted_without_intervention_rate {rate:.4}"));
            }
            if let Some(value) = correlation.interventions_per_accepted_result {
                lines.push(format!("  interventions_per_accepted_result {value:.4}"));
            }
        }
    }
    lines
}

fn push_section(lines: &mut Vec<String>, name: &str, rows: &[CountRow]) {
    lines.push(format!("{name}:"));
    for row in rows {
        lines.push(format!("  {} {}", row.label, row.count));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn report_expression_rejects_unknown_columns() {
        assert!(report_expression_for_alias("evidence_excerpt", "pi").is_err());
        assert!(report_expression_for_alias("raw_prompt", "pi").is_err());
    }

    #[test]
    fn report_expression_allows_only_safe_derived_dimensions() {
        assert_eq!(
            report_expression_for_alias("normalized_prompt_hash", "pi").unwrap(),
            "pi.normalized_prompt_hash"
        );
        assert_eq!(
            report_expression_for_alias("trajectory_id", "pi").unwrap(),
            "pi.trajectory_id::TEXT"
        );
    }

    #[test]
    fn headline_eligibility_uses_confidence_and_newer_supersession() {
        assert_eq!(HEADLINE_CONFIDENCE_THRESHOLD, 0.8);
        assert!(HEADLINE_ELIGIBLE_PREDICATE.contains("pi.confidence >= $4"));
        assert!(
            HEADLINE_ELIGIBLE_PREDICATE.contains("newer_pi.supersedes_record_id = pi.id"),
            "default reports must exclude an original row when a newer row points at it"
        );
        assert!(
            !HEADLINE_ELIGIBLE_PREDICATE.contains("pi.supersedes_record_id IS NULL"),
            "replacement rows are still headline-eligible when they meet confidence threshold"
        );
    }

    #[test]
    fn correlation_computes_accepted_without_intervention_rate() {
        let correlation = correlation_from_counts(4, 1, 3);
        assert_eq!(correlation.accepted_results_without_intervention, 3);
        assert_eq!(correlation.accepted_without_intervention_rate, Some(0.75));
        assert_eq!(correlation.interventions_per_accepted_result, Some(0.75));
    }

    #[test]
    fn report_lines_do_not_include_evidence_sections() {
        let report = PromptInterventionReport {
            interventions_per_day: vec![CountRow {
                label: "2026-06-08".to_string(),
                count: 2,
            }],
            by_type: vec![],
            by_signal_family: vec![],
            by_burden_type: vec![],
            by_failure_relation: vec![],
            by_requested_model: vec![],
            by_routed_model: vec![],
            by_baseline_arm: vec![],
            top_exact_prompt_hashes: vec![],
            top_normalized_prompt_hashes: vec![],
            per_trajectory: vec![],
            outcome_correlation: None,
        };
        let output = report_lines(&report).join("\n");
        assert!(output.contains("interventions_per_day"));
        assert!(!output.contains("evidence_excerpt"));
        assert!(!output.contains("raw_prompt"));
    }
}
