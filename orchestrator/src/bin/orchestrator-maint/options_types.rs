use uuid::Uuid;

pub(crate) struct BackfillOptions {
    pub(crate) dry_run: bool,
    pub(crate) batch_size: i64,
}

#[derive(Debug)]
pub(crate) struct PromptInterventionBackfillOptions {
    pub(crate) since: Option<chrono::DateTime<chrono::Utc>>,
    pub(crate) until: Option<chrono::DateTime<chrono::Utc>>,
    pub(crate) requested_model: Option<String>,
    pub(crate) response_model: Option<String>,
    pub(crate) repo: Option<String>,
    pub(crate) namespace: Option<String>,
    pub(crate) dry_run: bool,
    pub(crate) batch_size: i64,
}

#[derive(Debug)]
pub(crate) struct PromptInterventionReportOptions {
    pub(crate) repo: Option<String>,
    pub(crate) since: Option<chrono::DateTime<chrono::Utc>>,
    pub(crate) until: Option<chrono::DateTime<chrono::Utc>>,
    pub(crate) limit: i64,
}

pub(crate) struct ExtractFeaturesOptions {
    pub(crate) repo: Option<String>,
    pub(crate) session_id: Option<String>,
    pub(crate) trajectory_id: Option<Uuid>,
    pub(crate) since: Option<chrono::DateTime<chrono::Utc>>,
    pub(crate) dry_run: bool,
    pub(crate) batch_size: i64,
    pub(crate) skip_bootstrap_tagging: bool,
}

pub(crate) struct HarnessFeedbackOptions {
    pub(crate) repo: Option<String>,
    pub(crate) session_id: Option<String>,
    pub(crate) since: Option<chrono::DateTime<chrono::Utc>>,
    pub(crate) dry_run: bool,
    pub(crate) batch_size: i64,
}

pub(crate) struct RequestClassificationOptions {
    pub(crate) repo: Option<String>,
    pub(crate) session_id: Option<String>,
    pub(crate) since: Option<chrono::DateTime<chrono::Utc>>,
    pub(crate) dry_run: bool,
    pub(crate) repair: bool,
    pub(crate) batch_size: i64,
}

pub(crate) struct RequestClassificationReportOptions {
    pub(crate) repo: Option<String>,
    pub(crate) since: Option<chrono::DateTime<chrono::Utc>>,
}
