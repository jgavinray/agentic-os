use super::parse_common::{
    parse_dry_run_batch_options, parse_prompt_intervention_backfill_options, parse_report_options,
    parse_scoped_backfill_options, ScopedBackfillFlags,
};
use super::types::{
    BackfillOptions, ExtractFeaturesOptions, HarnessFeedbackOptions,
    PromptInterventionBackfillOptions, RequestClassificationOptions,
    RequestClassificationReportOptions,
};

impl HarnessFeedbackOptions {
    pub(crate) fn parse(args: Vec<String>) -> Result<Self, anyhow::Error> {
        let parsed = parse_scoped_backfill_options(
            args,
            ScopedBackfillFlags {
                allow_trajectory: false,
                allow_repair: false,
                allow_skip_bootstrap_tagging: false,
            },
        )?;

        Ok(Self {
            repo: parsed.repo,
            session_id: parsed.session_id,
            since: parsed.since,
            dry_run: parsed.dry_run,
            batch_size: parsed.batch_size,
        })
    }
}

impl RequestClassificationOptions {
    pub(crate) fn parse(args: Vec<String>) -> Result<Self, anyhow::Error> {
        let parsed = parse_scoped_backfill_options(
            args,
            ScopedBackfillFlags {
                allow_trajectory: false,
                allow_repair: true,
                allow_skip_bootstrap_tagging: false,
            },
        )?;

        Ok(Self {
            repo: parsed.repo,
            session_id: parsed.session_id,
            since: parsed.since,
            dry_run: parsed.dry_run,
            repair: parsed.repair,
            batch_size: parsed.batch_size,
        })
    }
}

impl RequestClassificationReportOptions {
    pub(crate) fn parse(args: Vec<String>) -> Result<Self, anyhow::Error> {
        let parsed = parse_report_options(args)?;

        Ok(Self {
            repo: parsed.repo,
            since: parsed.since,
        })
    }
}

impl ExtractFeaturesOptions {
    pub(crate) fn parse(args: Vec<String>) -> Result<Self, anyhow::Error> {
        let parsed = parse_scoped_backfill_options(
            args,
            ScopedBackfillFlags {
                allow_trajectory: true,
                allow_repair: false,
                allow_skip_bootstrap_tagging: true,
            },
        )?;

        Ok(Self {
            repo: parsed.repo,
            session_id: parsed.session_id,
            trajectory_id: parsed.trajectory_id,
            since: parsed.since,
            dry_run: parsed.dry_run,
            batch_size: parsed.batch_size,
            skip_bootstrap_tagging: parsed.skip_bootstrap_tagging,
        })
    }
}

impl BackfillOptions {
    pub(crate) fn parse(args: Vec<String>) -> Result<Self, anyhow::Error> {
        let parsed = parse_dry_run_batch_options(args)?;

        Ok(Self {
            dry_run: parsed.dry_run,
            batch_size: parsed.batch_size,
        })
    }
}

impl PromptInterventionBackfillOptions {
    pub(crate) fn parse(args: Vec<String>) -> Result<Self, anyhow::Error> {
        let parsed = parse_prompt_intervention_backfill_options(args)?;

        Ok(Self {
            since: parsed.since,
            until: parsed.until,
            requested_model: parsed.requested_model,
            response_model: parsed.response_model,
            repo: parsed.repo,
            namespace: parsed.namespace,
            dry_run: parsed.dry_run,
            batch_size: parsed.batch_size,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prompt_intervention_backfill_options_parse_all_filters() {
        let opts = PromptInterventionBackfillOptions::parse(vec![
            "--since".to_string(),
            "2026-06-01T00:00:00Z".to_string(),
            "--until".to_string(),
            "2026-06-02T00:00:00Z".to_string(),
            "--requested-model".to_string(),
            "claude-opus-4-8".to_string(),
            "--response-model".to_string(),
            "qwen3.6-27b".to_string(),
            "--repo".to_string(),
            "agentic-os".to_string(),
            "--namespace".to_string(),
            "default".to_string(),
            "--dry-run".to_string(),
            "--batch-size".to_string(),
            "25".to_string(),
        ])
        .expect("valid prompt intervention backfill options");

        assert!(opts.since.is_some());
        assert!(opts.until.is_some());
        assert_eq!(opts.requested_model.as_deref(), Some("claude-opus-4-8"));
        assert_eq!(opts.response_model.as_deref(), Some("qwen3.6-27b"));
        assert_eq!(opts.repo.as_deref(), Some("agentic-os"));
        assert_eq!(opts.namespace.as_deref(), Some("default"));
        assert!(opts.dry_run);
        assert_eq!(opts.batch_size, 25);
    }

    #[test]
    fn prompt_intervention_backfill_rejects_nonpositive_batch_size() {
        let err = PromptInterventionBackfillOptions::parse(vec![
            "--batch-size".to_string(),
            "0".to_string(),
        ])
        .expect_err("zero batch size should be rejected");

        assert!(err.to_string().contains("--batch-size must be positive"));
    }
}
