use uuid::Uuid;

const DEFAULT_BATCH_SIZE: i64 = 500;

pub(super) struct BackfillOptions {
    pub(super) dry_run: bool,
    pub(super) batch_size: i64,
}

pub(super) struct ExtractFeaturesOptions {
    pub(super) repo: Option<String>,
    pub(super) session_id: Option<String>,
    pub(super) trajectory_id: Option<Uuid>,
    pub(super) since: Option<chrono::DateTime<chrono::Utc>>,
    pub(super) dry_run: bool,
    pub(super) batch_size: i64,
    pub(super) skip_bootstrap_tagging: bool,
}

pub(super) struct HarnessFeedbackOptions {
    pub(super) repo: Option<String>,
    pub(super) session_id: Option<String>,
    pub(super) since: Option<chrono::DateTime<chrono::Utc>>,
    pub(super) dry_run: bool,
    pub(super) batch_size: i64,
}

pub(super) struct RequestClassificationOptions {
    pub(super) repo: Option<String>,
    pub(super) session_id: Option<String>,
    pub(super) since: Option<chrono::DateTime<chrono::Utc>>,
    pub(super) dry_run: bool,
    pub(super) repair: bool,
    pub(super) batch_size: i64,
}

pub(super) struct RequestClassificationReportOptions {
    pub(super) repo: Option<String>,
    pub(super) since: Option<chrono::DateTime<chrono::Utc>>,
}

impl HarnessFeedbackOptions {
    pub(super) fn parse(args: Vec<String>) -> Result<Self, anyhow::Error> {
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
    pub(super) fn parse(args: Vec<String>) -> Result<Self, anyhow::Error> {
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
    pub(super) fn parse(args: Vec<String>) -> Result<Self, anyhow::Error> {
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

impl ExtractFeaturesOptions {
    pub(super) fn parse(args: Vec<String>) -> Result<Self, anyhow::Error> {
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
    pub(super) fn parse(args: Vec<String>) -> Result<Self, anyhow::Error> {
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
