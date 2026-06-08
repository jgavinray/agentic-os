use super::parse_common::{
    parse_dry_run_batch_options, parse_report_options, parse_scoped_backfill_options,
    ScopedBackfillFlags,
};
use super::types::{
    BackfillOptions, ExtractFeaturesOptions, HarnessFeedbackOptions, RequestClassificationOptions,
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
