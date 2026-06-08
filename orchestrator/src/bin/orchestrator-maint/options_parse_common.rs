use uuid::Uuid;

pub(super) const DEFAULT_BATCH_SIZE: i64 = 500;

#[derive(Default)]
pub(super) struct ScopedBackfillOptions {
    pub(super) repo: Option<String>,
    pub(super) session_id: Option<String>,
    pub(super) trajectory_id: Option<Uuid>,
    pub(super) since: Option<chrono::DateTime<chrono::Utc>>,
    pub(super) dry_run: bool,
    pub(super) repair: bool,
    pub(super) batch_size: i64,
    pub(super) skip_bootstrap_tagging: bool,
}

pub(super) struct ScopedBackfillFlags {
    pub(super) allow_trajectory: bool,
    pub(super) allow_repair: bool,
    pub(super) allow_skip_bootstrap_tagging: bool,
}

pub(super) struct ReportOptions {
    pub(super) repo: Option<String>,
    pub(super) since: Option<chrono::DateTime<chrono::Utc>>,
}

pub(super) struct DryRunBatchOptions {
    pub(super) dry_run: bool,
    pub(super) batch_size: i64,
}

pub(super) fn parse_scoped_backfill_options(
    args: Vec<String>,
    flags: ScopedBackfillFlags,
) -> Result<ScopedBackfillOptions, anyhow::Error> {
    let mut parsed = ScopedBackfillOptions {
        batch_size: DEFAULT_BATCH_SIZE,
        ..Default::default()
    };
    let mut idx = 0usize;
    while idx < args.len() {
        match args[idx].as_str() {
            "--repo" | "–repo" => {
                parsed.repo = Some(option_value(&args, idx, "--repo requires a value")?);
                idx += 2;
            }
            "--session" | "–session" => {
                parsed.session_id = Some(option_value(&args, idx, "--session requires a value")?);
                idx += 2;
            }
            "--trajectory" | "–trajectory" if flags.allow_trajectory => {
                let value = option_value(&args, idx, "--trajectory requires a UUID")?;
                parsed.trajectory_id = Some(value.parse::<Uuid>()?);
                idx += 2;
            }
            "--since" | "–since" => {
                parsed.since = Some(parse_since(&args, idx)?);
                idx += 2;
            }
            "--dry-run" | "–dry-run" => {
                parsed.dry_run = true;
                idx += 1;
            }
            "--repair" | "–repair" if flags.allow_repair => {
                parsed.repair = true;
                idx += 1;
            }
            "--batch-size" | "–batch-size" => {
                parsed.batch_size = parse_positive_batch_size(&args, idx)?;
                idx += 2;
            }
            "--skip-bootstrap-tagging" | "–skip-bootstrap-tagging"
                if flags.allow_skip_bootstrap_tagging =>
            {
                parsed.skip_bootstrap_tagging = true;
                idx += 1;
            }
            other => anyhow::bail!("unknown option: {other}"),
        }
    }
    Ok(parsed)
}

pub(super) fn parse_report_options(args: Vec<String>) -> Result<ReportOptions, anyhow::Error> {
    let mut repo = None;
    let mut since = None;
    let mut idx = 0usize;
    while idx < args.len() {
        match args[idx].as_str() {
            "--repo" | "–repo" => {
                repo = Some(option_value(&args, idx, "--repo requires a value")?);
                idx += 2;
            }
            "--since" | "–since" => {
                since = Some(parse_since(&args, idx)?);
                idx += 2;
            }
            other => anyhow::bail!("unknown option: {other}"),
        }
    }

    Ok(ReportOptions { repo, since })
}

pub(super) fn parse_dry_run_batch_options(
    args: Vec<String>,
) -> Result<DryRunBatchOptions, anyhow::Error> {
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
                batch_size = parse_positive_batch_size(&args, idx)?;
                idx += 2;
            }
            other => anyhow::bail!("unknown option: {other}"),
        }
    }

    Ok(DryRunBatchOptions {
        dry_run,
        batch_size,
    })
}

fn option_value(
    args: &[String],
    idx: usize,
    message: &'static str,
) -> Result<String, anyhow::Error> {
    let Some(value) = args.get(idx + 1) else {
        anyhow::bail!(message);
    };
    Ok(value.clone())
}

fn parse_since(
    args: &[String],
    idx: usize,
) -> Result<chrono::DateTime<chrono::Utc>, anyhow::Error> {
    let value = option_value(args, idx, "--since requires an RFC3339 timestamp")?;
    Ok(chrono::DateTime::parse_from_rfc3339(&value)?.with_timezone(&chrono::Utc))
}

fn parse_positive_batch_size(args: &[String], idx: usize) -> Result<i64, anyhow::Error> {
    let value = option_value(args, idx, "--batch-size requires a positive integer")?;
    let batch_size = value.parse::<i64>()?;
    if batch_size <= 0 {
        anyhow::bail!("--batch-size must be positive");
    }
    Ok(batch_size)
}
