//! Deterministic request classification taxonomy.
//!
//! This module defines the bounded labels and table-facing structs for the
//! pre-LLM request classification layer. Feature extraction, backfill, and live
//! routing are later phases.

use chrono::{DateTime, Utc};
use deadpool_postgres::Pool;
use serde::{Deserialize, Serialize};
use serde_json::{json, Map, Value};
use uuid::Uuid;

pub const CLASSIFICATION_SCHEMA_VERSION: i32 = 1;
pub const ROUTING_POLICY_VERSION: &str = "deterministic-v1";
pub const CLASSIFIER_SOURCE_DETERMINISTIC_RULES: &str = "deterministic_rules";

pub const FEATURE_KEYS: &[&str] = &[
    "char_count",
    "line_count",
    "estimated_tokens",
    "has_code_block",
    "has_yaml",
    "has_json",
    "has_stack_trace",
    "has_logs",
    "has_shell_command",
    "has_url",
    "has_file_path",
    "has_secret_candidate",
    "contains_error_words",
    "contains_destructive_verbs",
    "asks_for_latest",
    "asks_for_file_generation",
    "detected_domain_terms",
    "has_kubernetes_terms",
    "has_docker_terms",
    "has_llm_terms",
    "has_networking_terms",
    "has_security_terms",
    "has_config_shape",
    "has_diff_or_patch",
    "has_test_failure",
    "is_composite",
    "decomposition_candidate",
    "decomposition_reason",
    "sub_intent_count",
    "sub_intents",
];

macro_rules! request_classification_enums {
    (
        $(
            $(#[$enum_meta:meta])*
            pub enum $name:ident {
                $($variant:ident => $label:literal),* $(,)?
            }
        )*
    ) => {
        $(
            $(#[$enum_meta])*
            #[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
            pub enum $name {
                $($variant,)*
                #[default]
                Unknown,
            }

            impl $name {
                pub const fn as_str(self) -> &'static str {
                    match self {
                        $(Self::$variant => $label,)*
                        Self::Unknown => "unknown",
                    }
                }

                pub fn from_label(value: &str) -> Self {
                    match value {
                        $($label => Self::$variant,)*
                        "unknown" => Self::Unknown,
                        _ => Self::Unknown,
                    }
                }
            }
        )*

        pub fn enum_inventory() -> &'static [(&'static str, &'static [&'static str])] {
            &[
                $(
                    (
                        stringify!($name),
                        &[$($label,)* "unknown"],
                    ),
                )*
            ]
        }
    };
}

request_classification_enums! {
    pub enum RequestIntent {
        Explain => "explain",
        Debug => "debug",
        Implement => "implement",
        GenerateConfig => "generate_config",
        ModifyConfig => "modify_config",
        Summarize => "summarize",
        Classify => "classify",
        Search => "search",
        Plan => "plan",
        OperateTool => "operate_tool",
    }

    pub enum RequestDomain {
        Shell => "shell",
        Kubernetes => "kubernetes",
        LlmInference => "llm_inference",
        Docker => "docker",
        Networking => "networking",
        Security => "security",
        Medical => "medical",
        Legal => "legal",
        Finance => "finance",
        Generic => "generic",
    }

    pub enum RequestArtifactType {
        PlainText => "plain_text",
        Code => "code",
        Logs => "logs",
        Yaml => "yaml",
        Json => "json",
        Sql => "sql",
        Markdown => "markdown",
        Image => "image",
        File => "file",
    }

    pub enum RequestComplexity {
        L0Trivial => "l0_trivial",
        L1Simple => "l1_simple",
        L2Moderate => "l2_moderate",
        L3Complex => "l3_complex",
        L4ToolRequired => "l4_tool_required",
        L5HighRisk => "l5_high_risk",
    }

    pub enum RequestRisk {
        None => "none",
        SecretPresent => "secret_present",
        DestructiveCommand => "destructive_command",
        ExternalCurrentInfoRequired => "external_current_info_required",
        HighStakes => "high_stakes",
        PromptInjection => "prompt_injection",
        UnsafeSecurity => "unsafe_security",
    }

    pub enum RecommendedRoute {
        DeterministicTemplate => "deterministic_template",
        SmallLocalModel => "small_local_model",
        StrongLocalModel => "strong_local_model",
        WebRequired => "web_required",
        ToolRequired => "tool_required",
        AskClarification => "ask_clarification",
        RefuseOrGuardrail => "refuse_or_guardrail",
    }

    pub enum ResponseContract {
        DirectAnswer => "direct_answer",
        StructuredJson => "structured_json",
        MarkdownSummary => "markdown_summary",
        PatchRequired => "patch_required",
        ValidationRequired => "validation_required",
        ClarificationQuestion => "clarification_question",
        Refusal => "refusal",
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RequestClassification {
    pub event_id: String,
    pub repo: String,
    pub session_id: String,
    pub trajectory_id: Option<Uuid>,
    pub event_created_at: DateTime<Utc>,
    pub classified_at: DateTime<Utc>,
    pub classification_schema_version: i32,
    pub routing_policy_version: String,
    pub classifier_source: String,
    pub intent: RequestIntent,
    pub domain: RequestDomain,
    pub secondary_domains: Vec<RequestDomain>,
    pub artifact_type: RequestArtifactType,
    pub risk: Vec<RequestRisk>,
    pub complexity: RequestComplexity,
    pub recommended_route: RecommendedRoute,
    pub response_contract: ResponseContract,
    pub features: Value,
}

impl RequestClassification {
    pub fn deterministic(
        event_id: String,
        repo: String,
        session_id: String,
        event_created_at: DateTime<Utc>,
    ) -> Self {
        Self {
            event_id,
            repo,
            session_id,
            trajectory_id: None,
            event_created_at,
            classified_at: event_created_at,
            classification_schema_version: CLASSIFICATION_SCHEMA_VERSION,
            routing_policy_version: ROUTING_POLICY_VERSION.to_string(),
            classifier_source: CLASSIFIER_SOURCE_DETERMINISTIC_RULES.to_string(),
            intent: RequestIntent::Unknown,
            domain: RequestDomain::Unknown,
            secondary_domains: Vec::new(),
            artifact_type: RequestArtifactType::Unknown,
            risk: vec![RequestRisk::Unknown],
            complexity: RequestComplexity::Unknown,
            recommended_route: RecommendedRoute::Unknown,
            response_contract: ResponseContract::Unknown,
            features: Value::Object(Default::default()),
        }
    }
}

#[derive(Debug, Clone)]
pub struct BackfillOptions {
    pub repo: Option<String>,
    pub session_id: Option<String>,
    pub since: Option<DateTime<Utc>>,
    pub dry_run: bool,
    pub repair: bool,
    pub repair_stale: bool,
    pub batch_size: i64,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct BackfillReport {
    pub events_scanned: usize,
    pub inserted: usize,
    pub updated: usize,
    pub skipped: usize,
    pub dry_run: bool,
    pub batch_size: i64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PersistOutcome {
    Inserted,
    Updated,
    Skipped,
}

impl PersistOutcome {
    pub const fn as_str(&self) -> &'static str {
        match self {
            Self::Inserted => "inserted",
            Self::Updated => "updated",
            Self::Skipped => "skipped",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RequestClassificationReport {
    pub by_route: Vec<LabelCount>,
    pub top_risk_flags: Vec<LabelCount>,
    pub unknown_label_counts: Vec<LabelCount>,
    pub repeated_guardrail_sessions: Vec<SessionRouteCount>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LabelCount {
    pub label: String,
    pub count: i64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SessionRouteCount {
    pub session_id: String,
    pub count: i64,
}

#[derive(Debug, Clone)]
pub struct ReportOptions {
    pub repo: Option<String>,
    pub since: Option<DateTime<Utc>>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LivePolicyConfig {
    pub enabled: bool,
    pub policy_version: String,
}

impl Default for LivePolicyConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            policy_version: "v1".to_string(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LivePolicyDecision {
    pub action: &'static str,
    pub reason: &'static str,
    pub route: RecommendedRoute,
    pub response_contract: ResponseContract,
}

pub fn live_policy_config_from_env() -> LivePolicyConfig {
    LivePolicyConfig {
        enabled: std::env::var("REQUEST_CLASSIFICATION_LIVE_POLICY_ENABLED")
            .map(|value| {
                !matches!(
                    value.to_ascii_lowercase().as_str(),
                    "0" | "false" | "no" | "off"
                )
            })
            .unwrap_or(false),
        policy_version: std::env::var("REQUEST_CLASSIFICATION_POLICY_VERSION")
            .unwrap_or_else(|_| "v1".to_string()),
    }
}

pub fn request_classification_startup_backfill_enabled() -> bool {
    std::env::var("REQUEST_CLASSIFICATION_STARTUP_BACKFILL_ENABLED")
        .map(|value| {
            !matches!(
                value.to_ascii_lowercase().as_str(),
                "0" | "false" | "no" | "off"
            )
        })
        .unwrap_or(true)
}

pub fn request_classification_startup_batch_size() -> i64 {
    std::env::var("REQUEST_CLASSIFICATION_STARTUP_BACKFILL_BATCH_SIZE")
        .ok()
        .and_then(|value| value.parse::<i64>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(500)
}

#[derive(Debug, Clone, Copy)]
struct RequestFeatures {
    char_count: usize,
    line_count: usize,
    estimated_tokens: usize,
    has_code_block: bool,
    has_yaml: bool,
    has_json: bool,
    has_stack_trace: bool,
    has_logs: bool,
    has_shell_command: bool,
    has_url: bool,
    has_file_path: bool,
    has_secret_candidate: bool,
    contains_error_words: bool,
    contains_destructive_verbs: bool,
    asks_for_latest: bool,
    asks_for_file_generation: bool,
    has_kubernetes_terms: bool,
    has_docker_terms: bool,
    has_llm_terms: bool,
    has_networking_terms: bool,
    has_security_terms: bool,
    has_config_shape: bool,
    has_diff_or_patch: bool,
    has_test_failure: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct CompositeAnalysis {
    is_composite: bool,
    decomposition_candidate: bool,
    reason: &'static str,
    sub_intents: Vec<RequestIntent>,
}

/// Classify a loaded event into deterministic request-level features and labels.
///
/// This is a pure Phase 2 entry point: it does not query storage, call models,
/// touch the filesystem, emit metrics, or change live routing behavior.
pub fn classify_request_event(event: &crate::db::AgentEvent) -> RequestClassification {
    let text = event_text(event);
    let lower = text.to_ascii_lowercase();
    let metadata_keys = metadata_key_text(&event.metadata);
    let metadata_keys_lower = metadata_keys.to_ascii_lowercase();
    let features = extract_features(&text, &lower, &metadata_keys_lower);
    let detected_domains = detected_domains(&features, &lower);
    let composite = analyze_composition(&text, &lower, &event.event_type);

    let mut row = RequestClassification::deterministic(
        event.id.clone(),
        event.repo.clone(),
        event.session_id.clone(),
        event.created_at,
    );
    row.trajectory_id = event.trajectory_id;
    row.features = features_to_json(&features, &detected_domains, &composite);
    row.intent = classify_intent(&features, &lower, &event.event_type);
    row.domain = classify_domain(&features, &lower, &detected_domains);
    row.secondary_domains = detected_domains
        .iter()
        .copied()
        .filter(|domain| *domain != row.domain)
        .collect();
    row.artifact_type = classify_artifact(&features, &lower);
    row.risk = classify_risk(&features, &lower, row.domain);
    row.complexity = classify_complexity(&features, row.intent, &row.risk, detected_domains.len());
    row.recommended_route = recommend_route(row.intent, row.complexity, &row.risk, &features);
    row.response_contract = response_contract(row.intent, row.artifact_type, row.recommended_route);
    row
}

pub fn classify_request_text(
    repo: &str,
    session_id: &str,
    summary: &str,
    evidence: Option<&str>,
    event_type: &str,
) -> RequestClassification {
    let event = crate::db::AgentEvent {
        id: "live-request".to_string(),
        session_id: session_id.to_string(),
        repo: repo.to_string(),
        actor: "user".to_string(),
        event_type: event_type.to_string(),
        summary: summary.to_string(),
        evidence: evidence.map(str::to_string),
        metadata: Value::Object(Default::default()),
        correlation_id: None,
        parent_event_id: None,
        trajectory_id: None,
        attempt_index: None,
        event_role: Some("request".to_string()),
        created_at: Utc::now(),
        summary_level: 0,
    };
    classify_request_event(&event)
}

pub fn is_classifiable_request_event(event: &crate::db::AgentEvent) -> bool {
    if !has_request_text(event) {
        return false;
    }

    event.event_type == "user_message"
        || event.event_role.as_deref() == Some("request")
        || (event.event_type == "context_pack"
            && event.event_role.as_deref() == Some("context_pack")
            && event.metadata.get("request").is_some())
}

pub async fn classify_and_persist_event(
    pool: &Pool,
    event: &crate::db::AgentEvent,
) -> Result<Option<PersistOutcome>, anyhow::Error> {
    if !is_classifiable_request_event(event) {
        return Ok(None);
    }
    let classification = classify_request_event(event);
    let outcome = persist_classification(pool, &classification).await?;
    Ok(Some(outcome))
}

pub async fn persist_classification(
    pool: &Pool,
    classification: &RequestClassification,
) -> Result<PersistOutcome, anyhow::Error> {
    let result = async {
        let conn = pool.get().await?;
        let secondary_domains = classification
            .secondary_domains
            .iter()
            .map(|domain| domain.as_str().to_string())
            .collect::<Vec<_>>();
        let risk = classification
            .risk
            .iter()
            .map(|risk| risk.as_str().to_string())
            .collect::<Vec<_>>();
        let affected = conn
            .execute(
                "INSERT INTO agent_request_classifications (
                    event_id,
                    repo,
                    session_id,
                    trajectory_id,
                    event_created_at,
                    classified_at,
                    classification_schema_version,
                    routing_policy_version,
                    classifier_source,
                    intent,
                    domain,
                    secondary_domains,
                    artifact_type,
                    risk,
                    complexity,
                    recommended_route,
                    response_contract,
                    features
                )
                VALUES (
                    $1, $2, $3, $4, $5, $6, $7, $8, $9,
                    $10, $11, $12, $13, $14, $15, $16, $17, $18
                )
                ON CONFLICT (
                    event_id,
                    classification_schema_version,
                    routing_policy_version
                )
                DO NOTHING",
                &[
                    &classification.event_id,
                    &classification.repo,
                    &classification.session_id,
                    &classification.trajectory_id,
                    &classification.event_created_at,
                    &classification.classified_at,
                    &classification.classification_schema_version,
                    &classification.routing_policy_version,
                    &classification.classifier_source,
                    &classification.intent.as_str(),
                    &classification.domain.as_str(),
                    &secondary_domains,
                    &classification.artifact_type.as_str(),
                    &risk,
                    &classification.complexity.as_str(),
                    &classification.recommended_route.as_str(),
                    &classification.response_contract.as_str(),
                    &classification.features,
                ],
            )
            .await?;
        Ok::<PersistOutcome, anyhow::Error>(if affected == 1 {
            PersistOutcome::Inserted
        } else {
            PersistOutcome::Skipped
        })
    }
    .await;

    match &result {
        Ok(outcome) => {
            crate::telemetry::record_request_classification_write(outcome.as_str());
            if matches!(outcome, PersistOutcome::Inserted) {
                record_classification_metrics(classification);
            }
        }
        Err(_) => crate::telemetry::record_request_classification_write("error"),
    }
    result
}

pub async fn update_classification_if_changed(
    pool: &Pool,
    classification: &RequestClassification,
) -> Result<PersistOutcome, anyhow::Error> {
    let result = async {
        let conn = pool.get().await?;
        let secondary_domains = classification
            .secondary_domains
            .iter()
            .map(|domain| domain.as_str().to_string())
            .collect::<Vec<_>>();
        let risk = classification
            .risk
            .iter()
            .map(|risk| risk.as_str().to_string())
            .collect::<Vec<_>>();
        let affected = conn
            .execute(
                "UPDATE agent_request_classifications
                 SET
                    repo = $2,
                    session_id = $3,
                    trajectory_id = $4,
                    event_created_at = $5,
                    classified_at = $6,
                    classifier_source = $9,
                    intent = $10,
                    domain = $11,
                    secondary_domains = $12,
                    artifact_type = $13,
                    risk = $14,
                    complexity = $15,
                    recommended_route = $16,
                    response_contract = $17,
                    features = $18
                 WHERE event_id = $1
                   AND classification_schema_version = $7
                   AND routing_policy_version = $8
                   AND (
                       repo IS DISTINCT FROM $2
                       OR session_id IS DISTINCT FROM $3
                       OR trajectory_id IS DISTINCT FROM $4
                       OR event_created_at IS DISTINCT FROM $5
                       OR classified_at IS DISTINCT FROM $6
                       OR classifier_source IS DISTINCT FROM $9
                       OR intent IS DISTINCT FROM $10
                       OR domain IS DISTINCT FROM $11
                       OR secondary_domains IS DISTINCT FROM $12
                       OR artifact_type IS DISTINCT FROM $13
                       OR risk IS DISTINCT FROM $14
                       OR complexity IS DISTINCT FROM $15
                       OR recommended_route IS DISTINCT FROM $16
                       OR response_contract IS DISTINCT FROM $17
                       OR features IS DISTINCT FROM $18
                   )",
                &[
                    &classification.event_id,
                    &classification.repo,
                    &classification.session_id,
                    &classification.trajectory_id,
                    &classification.event_created_at,
                    &classification.classified_at,
                    &classification.classification_schema_version,
                    &classification.routing_policy_version,
                    &classification.classifier_source,
                    &classification.intent.as_str(),
                    &classification.domain.as_str(),
                    &secondary_domains,
                    &classification.artifact_type.as_str(),
                    &risk,
                    &classification.complexity.as_str(),
                    &classification.recommended_route.as_str(),
                    &classification.response_contract.as_str(),
                    &classification.features,
                ],
            )
            .await?;
        Ok::<PersistOutcome, anyhow::Error>(if affected == 1 {
            PersistOutcome::Updated
        } else {
            PersistOutcome::Skipped
        })
    }
    .await;

    match &result {
        Ok(outcome) => {
            crate::telemetry::record_request_classification_write(outcome.as_str());
            if matches!(outcome, PersistOutcome::Updated) {
                record_classification_metrics(classification);
            }
        }
        Err(_) => crate::telemetry::record_request_classification_write("error"),
    }
    result
}

pub async fn run_backfill(
    pool: &Pool,
    opts: &BackfillOptions,
) -> Result<BackfillReport, anyhow::Error> {
    let result = run_backfill_inner(pool, opts).await;
    crate::telemetry::record_request_classification_backfill_run(if result.is_ok() {
        "success"
    } else {
        "failure"
    });
    result
}

async fn run_backfill_inner(
    pool: &Pool,
    opts: &BackfillOptions,
) -> Result<BackfillReport, anyhow::Error> {
    let batch_size = opts.batch_size.max(1);
    let mut report = BackfillReport {
        dry_run: opts.dry_run,
        batch_size,
        ..BackfillReport::default()
    };
    let mut last_created_at: Option<DateTime<Utc>> = None;
    let mut last_id: Option<String> = None;

    loop {
        let rows =
            load_classification_batch(pool, opts, batch_size, last_created_at, last_id.as_deref())
                .await?;
        if rows.is_empty() {
            break;
        }

        for row in rows {
            report.events_scanned += 1;
            last_created_at = Some(row.event.created_at);
            last_id = Some(row.event.id.clone());

            let should_repair = opts.repair || (opts.repair_stale && row.needs_stale_repair);
            if row.already_classified && !should_repair {
                report.skipped += 1;
                continue;
            }

            let classification = classify_request_event(&row.event);
            if opts.dry_run {
                if row.already_classified {
                    report.updated += 1;
                } else {
                    report.inserted += 1;
                }
                crate::telemetry::record_request_classification_write("dry_run");
                continue;
            }

            let outcome = if row.already_classified {
                update_classification_if_changed(pool, &classification).await?
            } else {
                persist_classification(pool, &classification).await?
            };
            match outcome {
                PersistOutcome::Inserted => report.inserted += 1,
                PersistOutcome::Updated => report.updated += 1,
                PersistOutcome::Skipped => report.skipped += 1,
            }
        }
    }

    Ok(report)
}

async fn load_classification_batch(
    pool: &Pool,
    opts: &BackfillOptions,
    batch_size: i64,
    last_created_at: Option<DateTime<Utc>>,
    last_id: Option<&str>,
) -> Result<Vec<ClassificationBatchRow>, anyhow::Error> {
    let conn = pool.get().await?;
    let last_id = last_id.map(str::to_string);
    let rows = conn
        .query(
            "SELECT
                e.id,
                e.session_id,
                e.repo,
                e.actor,
                e.event_type,
                e.summary,
                e.evidence,
                e.metadata,
                e.correlation_id,
                e.parent_event_id,
                e.trajectory_id,
                e.attempt_index,
                e.event_role,
                e.created_at,
                e.summary_level,
                c.event_id IS NOT NULL AS already_classified,
                (
                    c.event_id IS NOT NULL
                    AND coalesce(
                        CASE
                            WHEN (c.features->>'char_count') ~ '^[0-9]+$'
                            THEN (c.features->>'char_count')::INTEGER
                            ELSE NULL
                        END,
                        -1
                    ) = 0
                ) AS needs_stale_repair
             FROM agent_events e
             LEFT JOIN agent_request_classifications c
               ON c.event_id = e.id
              AND c.classification_schema_version = $1
              AND c.routing_policy_version = $2
             WHERE ($3::TEXT IS NULL OR e.repo = $3)
               AND ($4::TEXT IS NULL OR e.session_id = $4)
               AND ($5::TIMESTAMPTZ IS NULL OR e.created_at >= $5)
               AND length(btrim(coalesce(e.summary, '') || coalesce(e.evidence, ''), E' \t\n\r')) > 0
               AND (
                   e.event_type = 'user_message'
                   OR e.event_role = 'request'
                   OR (
                       e.event_type = 'context_pack'
                       AND e.event_role = 'context_pack'
                       AND e.metadata ? 'request'
                   )
               )
               AND (
                   $6::TIMESTAMPTZ IS NULL
                   OR e.created_at > $6
                   OR (e.created_at = $6 AND e.id > $7)
               )
             ORDER BY e.created_at ASC, e.id ASC
             LIMIT $8",
            &[
                &CLASSIFICATION_SCHEMA_VERSION,
                &ROUTING_POLICY_VERSION,
                &opts.repo,
                &opts.session_id,
                &opts.since,
                &last_created_at,
                &last_id,
                &batch_size,
            ],
        )
        .await?;

    Ok(rows
        .into_iter()
        .map(|row| {
            let event = crate::db::AgentEvent {
                id: row.get("id"),
                session_id: row.get("session_id"),
                repo: row.get("repo"),
                actor: row.get("actor"),
                event_type: row.get("event_type"),
                summary: row.get("summary"),
                evidence: row.get("evidence"),
                metadata: row.get("metadata"),
                correlation_id: row.get("correlation_id"),
                parent_event_id: row.get("parent_event_id"),
                trajectory_id: row.get("trajectory_id"),
                attempt_index: row.get("attempt_index"),
                event_role: row.get("event_role"),
                created_at: row.get("created_at"),
                summary_level: row.get("summary_level"),
            };
            ClassificationBatchRow {
                event,
                already_classified: row.get("already_classified"),
                needs_stale_repair: row.get("needs_stale_repair"),
            }
        })
        .collect())
}

struct ClassificationBatchRow {
    event: crate::db::AgentEvent,
    already_classified: bool,
    needs_stale_repair: bool,
}

pub async fn request_classification_report(
    pool: &Pool,
    opts: &ReportOptions,
) -> Result<RequestClassificationReport, anyhow::Error> {
    let conn = pool.get().await?;
    let by_route = count_grouped(
        &conn,
        "recommended_route",
        opts.repo.as_deref(),
        opts.since,
        20,
    )
    .await?;
    let unknown_label_counts =
        count_unknown_labels(&conn, opts.repo.as_deref(), opts.since).await?;
    let repeated_guardrail_sessions =
        count_repeated_guardrail_sessions(&conn, opts.repo.as_deref(), opts.since).await?;
    let top_risk_flags = count_risk_flags(&conn, opts.repo.as_deref(), opts.since, 20).await?;

    Ok(RequestClassificationReport {
        by_route,
        top_risk_flags,
        unknown_label_counts,
        repeated_guardrail_sessions,
    })
}

async fn count_grouped(
    conn: &deadpool_postgres::Object,
    column: &str,
    repo: Option<&str>,
    since: Option<DateTime<Utc>>,
    limit: i64,
) -> Result<Vec<LabelCount>, anyhow::Error> {
    let column = match column {
        "recommended_route" => "recommended_route",
        "intent" => "intent",
        "domain" => "domain",
        "complexity" => "complexity",
        _ => anyhow::bail!("unsupported report column"),
    };
    let sql = format!(
        "SELECT {column} AS label, count(*)::BIGINT AS count
         FROM agent_request_classifications
         WHERE ($1::TEXT IS NULL OR repo = $1)
           AND ($2::TIMESTAMPTZ IS NULL OR event_created_at >= $2)
         GROUP BY {column}
         ORDER BY count DESC, {column} ASC
         LIMIT $3"
    );
    let rows = conn.query(&sql, &[&repo, &since, &limit]).await?;
    Ok(rows
        .into_iter()
        .map(|row| LabelCount {
            label: row.get("label"),
            count: row.get("count"),
        })
        .collect())
}

async fn count_risk_flags(
    conn: &deadpool_postgres::Object,
    repo: Option<&str>,
    since: Option<DateTime<Utc>>,
    limit: i64,
) -> Result<Vec<LabelCount>, anyhow::Error> {
    let rows = conn
        .query(
            "SELECT risk_label AS label, count(*)::BIGINT AS count
             FROM agent_request_classifications,
                  unnest(risk) AS risk_label
             WHERE ($1::TEXT IS NULL OR repo = $1)
               AND ($2::TIMESTAMPTZ IS NULL OR event_created_at >= $2)
             GROUP BY risk_label
             ORDER BY count DESC, risk_label ASC
             LIMIT $3",
            &[&repo, &since, &limit],
        )
        .await?;
    Ok(rows
        .into_iter()
        .map(|row| LabelCount {
            label: row.get("label"),
            count: row.get("count"),
        })
        .collect())
}

async fn count_unknown_labels(
    conn: &deadpool_postgres::Object,
    repo: Option<&str>,
    since: Option<DateTime<Utc>>,
) -> Result<Vec<LabelCount>, anyhow::Error> {
    let rows = conn
        .query(
            "SELECT field, sum(count)::BIGINT AS count
             FROM (
                 SELECT 'intent' AS field, count(*)::BIGINT AS count
                 FROM agent_request_classifications
                 WHERE intent = 'unknown'
                   AND ($1::TEXT IS NULL OR repo = $1)
                   AND ($2::TIMESTAMPTZ IS NULL OR event_created_at >= $2)
                 UNION ALL
                 SELECT 'domain' AS field, count(*)::BIGINT AS count
                 FROM agent_request_classifications
                 WHERE domain = 'unknown'
                   AND ($1::TEXT IS NULL OR repo = $1)
                   AND ($2::TIMESTAMPTZ IS NULL OR event_created_at >= $2)
                 UNION ALL
                 SELECT 'artifact_type' AS field, count(*)::BIGINT AS count
                 FROM agent_request_classifications
                 WHERE artifact_type = 'unknown'
                   AND ($1::TEXT IS NULL OR repo = $1)
                   AND ($2::TIMESTAMPTZ IS NULL OR event_created_at >= $2)
                 UNION ALL
                 SELECT 'complexity' AS field, count(*)::BIGINT AS count
                 FROM agent_request_classifications
                 WHERE complexity = 'unknown'
                   AND ($1::TEXT IS NULL OR repo = $1)
                   AND ($2::TIMESTAMPTZ IS NULL OR event_created_at >= $2)
                 UNION ALL
                 SELECT 'recommended_route' AS field, count(*)::BIGINT AS count
                 FROM agent_request_classifications
                 WHERE recommended_route = 'unknown'
                   AND ($1::TEXT IS NULL OR repo = $1)
                   AND ($2::TIMESTAMPTZ IS NULL OR event_created_at >= $2)
                 UNION ALL
                 SELECT 'response_contract' AS field, count(*)::BIGINT AS count
                 FROM agent_request_classifications
                 WHERE response_contract = 'unknown'
                   AND ($1::TEXT IS NULL OR repo = $1)
                   AND ($2::TIMESTAMPTZ IS NULL OR event_created_at >= $2)
             ) unknowns
             GROUP BY field
             ORDER BY field ASC",
            &[&repo, &since],
        )
        .await?;
    Ok(rows
        .into_iter()
        .map(|row| LabelCount {
            label: row.get("field"),
            count: row.get("count"),
        })
        .collect())
}

async fn count_repeated_guardrail_sessions(
    conn: &deadpool_postgres::Object,
    repo: Option<&str>,
    since: Option<DateTime<Utc>>,
) -> Result<Vec<SessionRouteCount>, anyhow::Error> {
    let rows = conn
        .query(
            "SELECT session_id, count(*)::BIGINT AS count
             FROM agent_request_classifications
             WHERE recommended_route = 'refuse_or_guardrail'
               AND ($1::TEXT IS NULL OR repo = $1)
               AND ($2::TIMESTAMPTZ IS NULL OR event_created_at >= $2)
             GROUP BY session_id
             HAVING count(*) > 1
             ORDER BY count DESC, session_id ASC
             LIMIT 20",
            &[&repo, &since],
        )
        .await?;
    Ok(rows
        .into_iter()
        .map(|row| SessionRouteCount {
            session_id: row.get("session_id"),
            count: row.get("count"),
        })
        .collect())
}

pub fn evaluate_live_policy(
    classification: &RequestClassification,
    config: &LivePolicyConfig,
) -> Option<LivePolicyDecision> {
    if !config.enabled {
        return None;
    }
    if config.policy_version != "v1" {
        return None;
    }

    if classification
        .risk
        .iter()
        .any(|risk| matches!(risk, RequestRisk::UnsafeSecurity))
    {
        return Some(LivePolicyDecision {
            action: "refuse_or_guardrail",
            reason: "unsafe_security",
            route: RecommendedRoute::RefuseOrGuardrail,
            response_contract: ResponseContract::Refusal,
        });
    }
    if classification.risk.iter().any(|risk| {
        matches!(
            risk,
            RequestRisk::HighStakes | RequestRisk::DestructiveCommand | RequestRisk::SecretPresent
        )
    }) {
        return Some(LivePolicyDecision {
            action: "refuse_or_guardrail",
            reason: "objective_risk",
            route: RecommendedRoute::RefuseOrGuardrail,
            response_contract: ResponseContract::Refusal,
        });
    }
    if classification
        .risk
        .iter()
        .any(|risk| matches!(risk, RequestRisk::ExternalCurrentInfoRequired))
    {
        return Some(LivePolicyDecision {
            action: "web_required",
            reason: "external_current_info_required",
            route: RecommendedRoute::WebRequired,
            response_contract: ResponseContract::DirectAnswer,
        });
    }
    if classification.recommended_route == RecommendedRoute::AskClarification {
        return Some(LivePolicyDecision {
            action: "ask_clarification",
            reason: "missing_target_context",
            route: RecommendedRoute::AskClarification,
            response_contract: ResponseContract::ClarificationQuestion,
        });
    }
    if classification.recommended_route == RecommendedRoute::DeterministicTemplate {
        return Some(LivePolicyDecision {
            action: "deterministic_template",
            reason: "l0_trivial",
            route: RecommendedRoute::DeterministicTemplate,
            response_contract: ResponseContract::DirectAnswer,
        });
    }

    None
}

pub fn record_classification_metrics(classification: &RequestClassification) {
    crate::telemetry::record_request_classification(
        classification.intent.as_str(),
        classification.domain.as_str(),
        classification.recommended_route.as_str(),
    );
    crate::telemetry::record_request_route_recommendation(
        classification.recommended_route.as_str(),
    );
    crate::telemetry::record_request_complexity(classification.complexity.as_str());
    for risk in &classification.risk {
        crate::telemetry::record_request_risk_flag(risk.as_str());
    }
    for (field, is_unknown) in [
        ("intent", classification.intent == RequestIntent::Unknown),
        ("domain", classification.domain == RequestDomain::Unknown),
        (
            "artifact_type",
            classification.artifact_type == RequestArtifactType::Unknown,
        ),
        (
            "complexity",
            classification.complexity == RequestComplexity::Unknown,
        ),
        (
            "recommended_route",
            classification.recommended_route == RecommendedRoute::Unknown,
        ),
        (
            "response_contract",
            classification.response_contract == ResponseContract::Unknown,
        ),
    ] {
        if is_unknown {
            crate::telemetry::record_request_classification_unknown_label(field);
        }
    }
}

pub fn bounded_intent(value: &str) -> &'static str {
    RequestIntent::from_label(value).as_str()
}

pub fn bounded_domain(value: &str) -> &'static str {
    RequestDomain::from_label(value).as_str()
}

pub fn bounded_route(value: &str) -> &'static str {
    RecommendedRoute::from_label(value).as_str()
}

pub fn bounded_risk(value: &str) -> &'static str {
    RequestRisk::from_label(value).as_str()
}

pub fn bounded_complexity(value: &str) -> &'static str {
    RequestComplexity::from_label(value).as_str()
}

pub fn bounded_live_policy_action(value: &str) -> &'static str {
    match value {
        "ask_clarification" => "ask_clarification",
        "refuse_or_guardrail" => "refuse_or_guardrail",
        "web_required" => "web_required",
        "deterministic_template" => "deterministic_template",
        _ => "refuse_or_guardrail",
    }
}

pub fn bounded_live_policy_reason(value: &str) -> &'static str {
    match value {
        "unsafe_security" => "unsafe_security",
        "objective_risk" => "objective_risk",
        "external_current_info_required" => "external_current_info_required",
        "missing_target_context" => "missing_target_context",
        "l0_trivial" => "l0_trivial",
        _ => "objective_risk",
    }
}

pub fn bounded_live_policy_bypass(value: &str) -> &'static str {
    match value {
        "disabled" => "disabled",
        "shadow_only" => "shadow_only",
        "unsupported_policy_version" => "unsupported_policy_version",
        _ => "shadow_only",
    }
}

fn event_text(event: &crate::db::AgentEvent) -> String {
    match event.evidence.as_deref().filter(|value| !value.is_empty()) {
        Some(evidence) => format!("{}\n{}", event.summary, evidence),
        None => event.summary.clone(),
    }
}

fn has_request_text(event: &crate::db::AgentEvent) -> bool {
    !event_text(event).trim().is_empty()
}

fn metadata_key_text(value: &Value) -> String {
    fn collect(value: &Value, keys: &mut Vec<String>) {
        match value {
            Value::Object(map) => {
                for (key, nested) in map {
                    keys.push(key.clone());
                    collect(nested, keys);
                }
            }
            Value::Array(items) => {
                for item in items {
                    collect(item, keys);
                }
            }
            _ => {}
        }
    }

    let mut keys = Vec::new();
    collect(value, &mut keys);
    keys.join(" ")
}

fn extract_features(text: &str, lower: &str, metadata_keys_lower: &str) -> RequestFeatures {
    let trimmed = text.trim();
    let char_count = text.chars().count();
    let line_count = if trimmed.is_empty() {
        0
    } else {
        text.lines().count()
    };

    let has_code_block = text.contains("```");
    let has_json = has_code_block && lower.contains("```json")
        || trimmed.starts_with('{') && trimmed.ends_with('}')
        || contains_any(lower, &[" json ", ".json", "application/json"]);
    let has_yaml = has_code_block && (lower.contains("```yaml") || lower.contains("```yml"))
        || contains_any(lower, &[" yaml", ".yaml", ".yml", "apiVersion:"])
        || looks_like_yaml(text);
    let has_stack_trace = contains_any(
        lower,
        &[
            "traceback",
            "stack trace",
            "stack backtrace",
            "panicked at",
            "caused by:",
        ],
    );
    let has_logs = contains_any(
        lower,
        &[
            "\"level\":\"error\"",
            "\"level\":\"warn\"",
            " level=error",
            " level=warn",
            "[error]",
            "[warn]",
            "timestamp",
        ],
    );
    let has_shell_command = contains_any(
        lower,
        &[
            "docker compose",
            "kubectl ",
            "curl ",
            "cargo test",
            "git ",
            "psql ",
            "sudo ",
            "npm ",
            "yarn ",
            "rm -",
            "$ ",
        ],
    );
    let has_url = contains_any(lower, &["http://", "https://", "localhost:", "127.0.0.1"]);
    let has_file_path = has_url
        || contains_any(
            lower,
            &[
                "/users/",
                "/tmp/",
                "src/",
                "docs/",
                ".rs",
                ".py",
                ".ts",
                ".tsx",
                ".js",
                ".json",
                ".yaml",
                ".yml",
                ".md",
                "cargo.toml",
                "dockerfile",
            ],
        );
    let has_secret_candidate = contains_any(
        lower,
        &[
            "authorization: bearer",
            "api_key",
            "apikey",
            "access_token",
            "secret=",
            "password=",
            "sk-",
            "aws_secret_access_key",
        ],
    ) || contains_any(
        metadata_keys_lower,
        &[
            "authorization",
            "api_key",
            "apikey",
            "access_token",
            "secret",
            "password",
        ],
    );
    let contains_error_words = contains_any(
        lower,
        &[
            "error",
            "failed",
            "failure",
            "panic",
            "exception",
            "timeout",
            "refused",
            "invalid",
        ],
    );
    let contains_destructive_verbs = contains_any(
        lower,
        &[
            "rm -rf",
            "drop table",
            "delete from",
            "truncate table",
            "kubectl delete",
            "docker rm",
            "destroy",
            "wipe",
            "format disk",
            "git reset --hard",
        ],
    );
    let asks_for_latest = contains_any(
        lower,
        &[
            "latest",
            "current",
            "today",
            "right now",
            "up to date",
            "up-to-date",
            "look up",
            "search the web",
        ],
    );
    let asks_for_file_generation = contains_any(
        lower,
        &[
            "create a file",
            "write a file",
            "generate a file",
            "save this",
            "draft a commit",
            "produce a patch",
        ],
    );
    let has_kubernetes_terms = contains_any(
        lower,
        &[
            "kubernetes",
            "kubectl",
            "k8s",
            "pod",
            "deployment",
            "namespace",
            "helm",
        ],
    );
    let has_docker_terms = contains_any(
        lower,
        &[
            "docker",
            "dockerfile",
            "compose.yaml",
            "compose.yml",
            "container",
            "image",
        ],
    );
    let has_llm_terms = contains_any(
        lower,
        &[
            "llm",
            "prompt",
            "tokens",
            "inference",
            "vllm",
            "qwen",
            "claude",
            "openai",
            "litellm",
            "model",
        ],
    );
    let has_networking_terms = contains_any(
        lower,
        &[
            "dns",
            "tcp",
            "http",
            "https",
            "ingress",
            "proxy",
            "tls",
            "port ",
            "localhost",
        ],
    );
    let has_security_terms = contains_any(
        lower,
        &[
            "auth",
            "token",
            "secret",
            "password",
            "jwt",
            "exploit",
            "vulnerability",
            "csrf",
            "xss",
        ],
    );
    let has_config_shape = has_yaml
        || has_json
        || contains_any(
            lower,
            &[
                "config",
                "configuration",
                "env var",
                "environment variable",
                ".env",
                "compose.yaml",
                "values.yaml",
            ],
        );
    let has_diff_or_patch = contains_any(
        lower,
        &[
            "diff --git",
            "--- a/",
            "+++ b/",
            "@@",
            "apply_patch",
            "patch",
        ],
    );
    let has_test_failure = contains_any(
        lower,
        &[
            "test failed",
            "tests failed",
            "failures:",
            "assertion failed",
            "cargo test",
            "pytest",
            "expected:",
        ],
    );

    RequestFeatures {
        char_count,
        line_count,
        estimated_tokens: estimate_tokens(char_count),
        has_code_block,
        has_yaml,
        has_json,
        has_stack_trace,
        has_logs,
        has_shell_command,
        has_url,
        has_file_path,
        has_secret_candidate,
        contains_error_words,
        contains_destructive_verbs,
        asks_for_latest,
        asks_for_file_generation,
        has_kubernetes_terms,
        has_docker_terms,
        has_llm_terms,
        has_networking_terms,
        has_security_terms,
        has_config_shape,
        has_diff_or_patch,
        has_test_failure,
    }
}

fn features_to_json(
    features: &RequestFeatures,
    detected_domains: &[RequestDomain],
    composite: &CompositeAnalysis,
) -> Value {
    let mut object = Map::new();
    object.insert("char_count".to_string(), json!(features.char_count));
    object.insert("line_count".to_string(), json!(features.line_count));
    object.insert(
        "estimated_tokens".to_string(),
        json!(features.estimated_tokens),
    );
    object.insert("has_code_block".to_string(), json!(features.has_code_block));
    object.insert("has_yaml".to_string(), json!(features.has_yaml));
    object.insert("has_json".to_string(), json!(features.has_json));
    object.insert(
        "has_stack_trace".to_string(),
        json!(features.has_stack_trace),
    );
    object.insert("has_logs".to_string(), json!(features.has_logs));
    object.insert(
        "has_shell_command".to_string(),
        json!(features.has_shell_command),
    );
    object.insert("has_url".to_string(), json!(features.has_url));
    object.insert("has_file_path".to_string(), json!(features.has_file_path));
    object.insert(
        "has_secret_candidate".to_string(),
        json!(features.has_secret_candidate),
    );
    object.insert(
        "contains_error_words".to_string(),
        json!(features.contains_error_words),
    );
    object.insert(
        "contains_destructive_verbs".to_string(),
        json!(features.contains_destructive_verbs),
    );
    object.insert(
        "asks_for_latest".to_string(),
        json!(features.asks_for_latest),
    );
    object.insert(
        "asks_for_file_generation".to_string(),
        json!(features.asks_for_file_generation),
    );
    object.insert(
        "detected_domain_terms".to_string(),
        json!(detected_domains
            .iter()
            .map(|domain| domain.as_str())
            .collect::<Vec<_>>()),
    );
    object.insert(
        "has_kubernetes_terms".to_string(),
        json!(features.has_kubernetes_terms),
    );
    object.insert(
        "has_docker_terms".to_string(),
        json!(features.has_docker_terms),
    );
    object.insert("has_llm_terms".to_string(), json!(features.has_llm_terms));
    object.insert(
        "has_networking_terms".to_string(),
        json!(features.has_networking_terms),
    );
    object.insert(
        "has_security_terms".to_string(),
        json!(features.has_security_terms),
    );
    object.insert(
        "has_config_shape".to_string(),
        json!(features.has_config_shape),
    );
    object.insert(
        "has_diff_or_patch".to_string(),
        json!(features.has_diff_or_patch),
    );
    object.insert(
        "has_test_failure".to_string(),
        json!(features.has_test_failure),
    );
    object.insert("is_composite".to_string(), json!(composite.is_composite));
    object.insert(
        "decomposition_candidate".to_string(),
        json!(composite.decomposition_candidate),
    );
    object.insert("decomposition_reason".to_string(), json!(composite.reason));
    object.insert(
        "sub_intent_count".to_string(),
        json!(composite.sub_intents.len()),
    );
    object.insert(
        "sub_intents".to_string(),
        json!(composite
            .sub_intents
            .iter()
            .map(|intent| intent.as_str())
            .collect::<Vec<_>>()),
    );
    Value::Object(object)
}

fn detected_domains(features: &RequestFeatures, lower: &str) -> Vec<RequestDomain> {
    let mut domains = Vec::new();
    push_if(
        &mut domains,
        features.has_kubernetes_terms,
        RequestDomain::Kubernetes,
    );
    push_if(
        &mut domains,
        features.has_docker_terms,
        RequestDomain::Docker,
    );
    push_if(
        &mut domains,
        features.has_llm_terms,
        RequestDomain::LlmInference,
    );
    push_if(
        &mut domains,
        features.has_networking_terms,
        RequestDomain::Networking,
    );
    push_if(
        &mut domains,
        features.has_security_terms,
        RequestDomain::Security,
    );
    push_if(
        &mut domains,
        contains_any(lower, &["medical", "doctor", "diagnosis", "medicine"]),
        RequestDomain::Medical,
    );
    push_if(
        &mut domains,
        contains_any(lower, &["legal", "lawyer", "lawsuit", "contract"]),
        RequestDomain::Legal,
    );
    push_if(
        &mut domains,
        contains_any(lower, &["finance", "tax", "investment", "stock", "loan"]),
        RequestDomain::Finance,
    );
    push_if(
        &mut domains,
        features.has_shell_command,
        RequestDomain::Shell,
    );
    domains
}

fn classify_intent(features: &RequestFeatures, lower: &str, event_type: &str) -> RequestIntent {
    if features.char_count == 0 {
        return RequestIntent::Unknown;
    }
    if contains_any(lower, &["summarize", "summary", "recap"]) {
        RequestIntent::Summarize
    } else if contains_any(lower, &["classify", "categorize", "label this"]) {
        RequestIntent::Classify
    } else if contains_any(lower, &["search", "look up", "find current", "latest"]) {
        RequestIntent::Search
    } else if contains_any(lower, &["plan", "proposal", "approach", "design"]) {
        RequestIntent::Plan
    } else if contains_any(
        lower,
        &[
            "implement",
            "implementation",
            "build this",
            "build the",
            "add feature",
            "add support",
            "add functionality",
            "wire up",
            "integrate",
        ],
    ) {
        RequestIntent::Implement
    } else if contains_any(
        lower,
        &["generate config", "create yaml", "write yaml", "manifest"],
    ) {
        RequestIntent::GenerateConfig
    } else if features.has_config_shape
        && contains_any(lower, &["edit", "modify", "change", "fix", "update"])
    {
        RequestIntent::ModifyConfig
    } else if contains_any(lower, &["run ", "execute ", "deploy", "restart"])
        || event_type == "tool_call"
    {
        RequestIntent::OperateTool
    } else if features.contains_error_words || features.has_stack_trace || features.has_test_failure
    {
        RequestIntent::Debug
    } else {
        RequestIntent::Explain
    }
}

fn classify_domain(
    features: &RequestFeatures,
    lower: &str,
    detected_domains: &[RequestDomain],
) -> RequestDomain {
    for high_stakes in [
        RequestDomain::Medical,
        RequestDomain::Legal,
        RequestDomain::Finance,
    ] {
        if detected_domains.contains(&high_stakes) {
            return high_stakes;
        }
    }
    if features.char_count == 0 {
        RequestDomain::Unknown
    } else if let Some(domain) = detected_domains.first() {
        *domain
    } else if contains_any(lower, &["shell", "terminal", "bash", "zsh"]) {
        RequestDomain::Shell
    } else {
        RequestDomain::Generic
    }
}

fn classify_artifact(features: &RequestFeatures, lower: &str) -> RequestArtifactType {
    if features.char_count == 0 {
        RequestArtifactType::Unknown
    } else if features.has_logs || features.has_stack_trace || features.has_test_failure {
        RequestArtifactType::Logs
    } else if features.has_json {
        RequestArtifactType::Json
    } else if features.has_yaml {
        RequestArtifactType::Yaml
    } else if contains_any(
        lower,
        &["sql", "select ", "insert into", "update ", "delete from"],
    ) {
        RequestArtifactType::Sql
    } else if contains_any(lower, &["markdown", ".md", "# "]) {
        RequestArtifactType::Markdown
    } else if features.has_code_block || features.has_diff_or_patch {
        RequestArtifactType::Code
    } else if contains_any(lower, &["image", "screenshot", ".png", ".jpg", ".jpeg"]) {
        RequestArtifactType::Image
    } else if features.has_file_path || features.asks_for_file_generation {
        RequestArtifactType::File
    } else {
        RequestArtifactType::PlainText
    }
}

fn classify_risk(
    features: &RequestFeatures,
    lower: &str,
    domain: RequestDomain,
) -> Vec<RequestRisk> {
    if features.char_count == 0 {
        return vec![RequestRisk::Unknown];
    }

    let mut risks = Vec::new();
    push_if(
        &mut risks,
        features.has_secret_candidate,
        RequestRisk::SecretPresent,
    );
    push_if(
        &mut risks,
        features.contains_destructive_verbs,
        RequestRisk::DestructiveCommand,
    );
    push_if(
        &mut risks,
        features.asks_for_latest,
        RequestRisk::ExternalCurrentInfoRequired,
    );
    push_if(
        &mut risks,
        matches!(
            domain,
            RequestDomain::Medical | RequestDomain::Legal | RequestDomain::Finance
        ),
        RequestRisk::HighStakes,
    );
    push_if(
        &mut risks,
        contains_any(
            lower,
            &[
                "ignore previous",
                "ignore all previous",
                "system prompt",
                "developer message",
                "jailbreak",
            ],
        ),
        RequestRisk::PromptInjection,
    );
    push_if(
        &mut risks,
        contains_any(
            lower,
            &[
                "steal credentials",
                "credential dump",
                "phishing",
                "malware",
                "exploit this",
                "bypass auth",
            ],
        ),
        RequestRisk::UnsafeSecurity,
    );

    if risks.is_empty() {
        risks.push(RequestRisk::None);
    }
    risks
}

fn classify_complexity(
    features: &RequestFeatures,
    intent: RequestIntent,
    risks: &[RequestRisk],
    domain_count: usize,
) -> RequestComplexity {
    if features.char_count == 0 {
        RequestComplexity::Unknown
    } else if risks.iter().any(|risk| {
        matches!(
            risk,
            RequestRisk::HighStakes
                | RequestRisk::UnsafeSecurity
                | RequestRisk::DestructiveCommand
                | RequestRisk::SecretPresent
        )
    }) {
        RequestComplexity::L5HighRisk
    } else if matches!(intent, RequestIntent::OperateTool)
        || features.has_shell_command
        || features.asks_for_file_generation
    {
        RequestComplexity::L4ToolRequired
    } else if features.char_count > 2_000
        || features.line_count > 60
        || features.has_stack_trace
        || features.has_diff_or_patch
        || features.has_test_failure
    {
        RequestComplexity::L3Complex
    } else if domain_count > 1
        || matches!(
            intent,
            RequestIntent::Debug
                | RequestIntent::Implement
                | RequestIntent::GenerateConfig
                | RequestIntent::ModifyConfig
        )
        || features.has_config_shape
    {
        RequestComplexity::L2Moderate
    } else if features.char_count <= 40 && !features.contains_error_words {
        RequestComplexity::L0Trivial
    } else {
        RequestComplexity::L1Simple
    }
}

fn recommend_route(
    intent: RequestIntent,
    complexity: RequestComplexity,
    risks: &[RequestRisk],
    features: &RequestFeatures,
) -> RecommendedRoute {
    if risks.iter().any(|risk| {
        matches!(
            risk,
            RequestRisk::UnsafeSecurity
                | RequestRisk::HighStakes
                | RequestRisk::DestructiveCommand
                | RequestRisk::SecretPresent
        )
    }) {
        RecommendedRoute::RefuseOrGuardrail
    } else if risks.contains(&RequestRisk::ExternalCurrentInfoRequired) {
        RecommendedRoute::WebRequired
    } else if matches!(
        intent,
        RequestIntent::OperateTool | RequestIntent::Implement | RequestIntent::ModifyConfig
    ) && !features.has_file_path
        && !features.has_config_shape
        && !features.has_shell_command
    {
        RecommendedRoute::AskClarification
    } else if matches!(intent, RequestIntent::OperateTool) || features.has_shell_command {
        RecommendedRoute::ToolRequired
    } else if matches!(complexity, RequestComplexity::L0Trivial) {
        RecommendedRoute::DeterministicTemplate
    } else if matches!(
        (intent, complexity),
        (
            RequestIntent::Explain | RequestIntent::Summarize | RequestIntent::Classify,
            RequestComplexity::L1Simple | RequestComplexity::L2Moderate
        )
    ) {
        RecommendedRoute::SmallLocalModel
    } else if matches!(complexity, RequestComplexity::L3Complex) {
        RecommendedRoute::StrongLocalModel
    } else if matches!(complexity, RequestComplexity::Unknown) {
        RecommendedRoute::Unknown
    } else {
        RecommendedRoute::SmallLocalModel
    }
}

fn response_contract(
    intent: RequestIntent,
    artifact_type: RequestArtifactType,
    route: RecommendedRoute,
) -> ResponseContract {
    match route {
        RecommendedRoute::RefuseOrGuardrail => ResponseContract::Refusal,
        RecommendedRoute::AskClarification => ResponseContract::ClarificationQuestion,
        RecommendedRoute::ToolRequired => ResponseContract::ValidationRequired,
        _ if matches!(intent, RequestIntent::Classify) => ResponseContract::StructuredJson,
        _ if matches!(intent, RequestIntent::Summarize | RequestIntent::Plan) => {
            ResponseContract::MarkdownSummary
        }
        _ if matches!(intent, RequestIntent::Implement) => ResponseContract::ValidationRequired,
        _ if matches!(
            artifact_type,
            RequestArtifactType::Code | RequestArtifactType::Yaml | RequestArtifactType::Json
        ) =>
        {
            ResponseContract::ValidationRequired
        }
        _ if matches!(route, RecommendedRoute::Unknown) => ResponseContract::Unknown,
        _ => ResponseContract::DirectAnswer,
    }
}

fn analyze_composition(text: &str, lower: &str, event_type: &str) -> CompositeAnalysis {
    if text.trim().is_empty() {
        return CompositeAnalysis {
            is_composite: false,
            decomposition_candidate: false,
            reason: "none",
            sub_intents: Vec::new(),
        };
    }

    let (fragments, reason) = decomposition_fragments(text, lower);
    let mut sub_intents = Vec::new();
    for fragment in fragments.iter().take(5) {
        let fragment = fragment.trim();
        if fragment.len() < 3 {
            continue;
        }
        let fragment_lower = fragment.to_ascii_lowercase();
        if !has_subtask_action_signal(&fragment_lower) {
            continue;
        }
        let features = extract_features(fragment, &fragment_lower, "");
        sub_intents.push(classify_intent(&features, &fragment_lower, event_type));
    }

    let decomposition_candidate = sub_intents.len() >= 2;
    CompositeAnalysis {
        is_composite: decomposition_candidate,
        decomposition_candidate,
        reason: if decomposition_candidate {
            reason
        } else {
            "none"
        },
        sub_intents: if decomposition_candidate {
            sub_intents
        } else {
            Vec::new()
        },
    }
}

fn decomposition_fragments(text: &str, lower: &str) -> (Vec<String>, &'static str) {
    let structured = structured_list_fragments(text);
    if structured.len() >= 2 {
        return (structured, "structured_list");
    }

    for (separator, reason) in [
        ("\n", "line_separated"),
        (";", "sequence_separator"),
        (", then ", "sequence_separator"),
        (" then ", "sequence_separator"),
    ] {
        if lower.contains(separator) {
            let fragments = split_nonempty(text, separator);
            if fragments.len() >= 2 {
                return (fragments, reason);
            }
        }
    }

    if action_signal_count(lower) >= 2 {
        let coordinated = split_nonempty(text, " and ");
        if coordinated.len() >= 2 {
            return (coordinated, "coordinated_actions");
        }

        let comma_actions = split_on_action_commas(text);
        if comma_actions.len() >= 2 {
            return (comma_actions, "coordinated_actions");
        }
    }

    (Vec::new(), "none")
}

fn structured_list_fragments(text: &str) -> Vec<String> {
    text.lines()
        .filter_map(|line| {
            let trimmed = line.trim();
            let item = trimmed
                .strip_prefix("- ")
                .or_else(|| trimmed.strip_prefix("* "))
                .or_else(|| numbered_item_text(trimmed))?;
            let item = item.trim();
            (!item.is_empty()).then(|| item.to_string())
        })
        .take(5)
        .collect()
}

fn numbered_item_text(value: &str) -> Option<&str> {
    let split_at = value
        .char_indices()
        .take_while(|(_, ch)| ch.is_ascii_digit())
        .last()
        .map(|(idx, ch)| idx + ch.len_utf8())?;
    let rest = value.get(split_at..)?;
    rest.strip_prefix(". ").or_else(|| rest.strip_prefix(") "))
}

fn split_nonempty(text: &str, separator: &str) -> Vec<String> {
    text.split(separator)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(str::to_string)
        .take(5)
        .collect()
}

fn split_on_action_commas(text: &str) -> Vec<String> {
    let mut fragments = Vec::new();
    for part in text.split(',') {
        let part = part.trim();
        if part.is_empty() {
            continue;
        }
        if fragments.is_empty() || has_subtask_action_signal(&part.to_ascii_lowercase()) {
            fragments.push(part.to_string());
        } else if let Some(last) = fragments.last_mut() {
            last.push_str(", ");
            last.push_str(part);
        }
    }
    fragments.into_iter().take(5).collect()
}

fn action_signal_count(lower: &str) -> usize {
    SUBTASK_ACTION_SIGNALS
        .iter()
        .filter(|signal| lower.contains(**signal))
        .count()
}

fn has_subtask_action_signal(lower: &str) -> bool {
    contains_any(lower, SUBTASK_ACTION_SIGNALS)
}

const SUBTASK_ACTION_SIGNALS: &[&str] = &[
    "investigate",
    "inspect",
    "look at",
    "look up",
    "search",
    "find",
    "read",
    "open",
    "review",
    "explain",
    "summarize",
    "classify",
    "plan",
    "design",
    "debug",
    "fix",
    "patch",
    "edit",
    "modify",
    "change",
    "update",
    "add",
    "remove",
    "create",
    "write",
    "generate",
    "run",
    "execute",
    "validate",
    "test",
    "deploy",
    "restart",
];

fn contains_any(value: &str, needles: &[&str]) -> bool {
    needles.iter().any(|needle| value.contains(needle))
}

fn looks_like_yaml(text: &str) -> bool {
    let mut yamlish_lines = 0;
    for line in text.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with('#') || trimmed.is_empty() {
            continue;
        }
        if trimmed.contains(": ") || trimmed.starts_with("- ") {
            yamlish_lines += 1;
        }
    }
    yamlish_lines >= 2
}

fn estimate_tokens(char_count: usize) -> usize {
    if char_count == 0 {
        0
    } else {
        char_count.div_ceil(4)
    }
}

fn push_if<T: PartialEq + Copy>(items: &mut Vec<T>, condition: bool, item: T) {
    if condition && !items.contains(&item) {
        items.push(item);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::db::AgentEvent;
    use std::collections::HashSet;

    const MIGRATION: &str = include_str!("../migrations/V9__request_classifications.sql");

    #[test]
    fn enum_inventory_contains_unknown_for_every_enum() {
        for (name, variants) in enum_inventory() {
            assert!(
                variants.contains(&"unknown"),
                "{name} must include unknown variant"
            );
        }
    }

    #[test]
    fn unknown_labels_map_to_unknown() {
        assert_eq!(
            RequestIntent::from_label("not-a-real-intent"),
            RequestIntent::Unknown
        );
        assert_eq!(
            RequestDomain::from_label("not-a-real-domain"),
            RequestDomain::Unknown
        );
        assert_eq!(
            RequestArtifactType::from_label("not-a-real-artifact"),
            RequestArtifactType::Unknown
        );
        assert_eq!(
            RequestComplexity::from_label("not-a-real-complexity"),
            RequestComplexity::Unknown
        );
        assert_eq!(
            RequestRisk::from_label("not-a-real-risk"),
            RequestRisk::Unknown
        );
        assert_eq!(
            RecommendedRoute::from_label("not-a-real-route"),
            RecommendedRoute::Unknown
        );
        assert_eq!(
            ResponseContract::from_label("not-a-real-contract"),
            ResponseContract::Unknown
        );
    }

    #[test]
    fn migration_has_checks_for_all_closed_labels() {
        for (_name, variants) in enum_inventory() {
            for variant in *variants {
                assert!(
                    MIGRATION.contains(&format!("'{variant}'")),
                    "migration is missing enum label {variant}"
                );
            }
        }
        assert!(MIGRATION.contains("CHECK (intent IN"));
        assert!(MIGRATION.contains("CHECK (domain IN"));
        assert!(MIGRATION.contains("CHECK (artifact_type IN"));
        assert!(MIGRATION.contains("CHECK (complexity IN"));
        assert!(MIGRATION.contains("CHECK (recommended_route IN"));
        assert!(MIGRATION.contains("CHECK (response_contract IN"));
        assert!(MIGRATION.contains("CHECK (risk <@ ARRAY"));
        assert!(MIGRATION.contains("CHECK (secondary_domains <@ ARRAY"));
    }

    #[test]
    fn schema_privacy_allows_only_documented_unbounded_text_columns() {
        let allowed_unbounded_strings: HashSet<&str> = [
            "event_id",
            "repo",
            "session_id",
            "routing_policy_version",
            "classifier_source",
        ]
        .into_iter()
        .collect();
        let text_columns = [
            "event_id",
            "repo",
            "session_id",
            "routing_policy_version",
            "classifier_source",
        ];

        for column in text_columns {
            assert!(
                allowed_unbounded_strings.contains(column),
                "{column} must be explicitly privacy-allowlisted"
            );
        }
        assert!(!MIGRATION.contains("summary TEXT"));
        assert!(!MIGRATION.contains("evidence TEXT"));
        assert!(!MIGRATION.contains("request TEXT"));
        assert!(!MIGRATION.contains("prompt TEXT"));
    }

    #[test]
    fn deterministic_default_row_uses_current_versions() {
        let now = Utc::now();
        let row = RequestClassification::deterministic(
            "event-1".to_string(),
            "repo".to_string(),
            "session".to_string(),
            now,
        );

        assert_eq!(
            row.classification_schema_version,
            CLASSIFICATION_SCHEMA_VERSION
        );
        assert_eq!(row.routing_policy_version, ROUTING_POLICY_VERSION);
        assert_eq!(row.classifier_source, CLASSIFIER_SOURCE_DETERMINISTIC_RULES);
        assert_eq!(row.intent, RequestIntent::Unknown);
        assert_eq!(row.recommended_route, RecommendedRoute::Unknown);
    }

    #[test]
    fn feature_extraction_is_deterministic_for_identical_input() {
        let event = event(
            "e-1",
            "Please explain the Docker compose error",
            Some("ERROR failed to connect to http://localhost:8088"),
        );

        let first = classify_request_event(&event);
        let second = classify_request_event(&event);

        assert_eq!(first, second);
        assert_eq!(first.domain, RequestDomain::Docker);
        assert_eq!(first.intent, RequestIntent::Debug);
    }

    #[test]
    fn feature_keys_are_closed_and_stable() {
        let row = classify_request_event(&event(
            "e-keys",
            "Summarize this Kubernetes log",
            Some("[ERROR] pod failed"),
        ));
        let object = row.features.as_object().expect("features must be object");
        let actual: HashSet<&str> = object.keys().map(String::as_str).collect();
        let expected: HashSet<&str> = FEATURE_KEYS.iter().copied().collect();

        assert_eq!(actual, expected);
    }

    #[test]
    fn composite_requests_emit_bounded_sub_intents() {
        let row = classify_request_event(&event(
            "e-composite",
            "Search the repo for context injection; implement the fix in src/main.rs; run cargo test; summarize the result",
            None,
        ));

        assert_eq!(row.features["is_composite"], true);
        assert_eq!(row.features["decomposition_candidate"], true);
        assert_eq!(row.features["decomposition_reason"], "sequence_separator");
        assert_eq!(row.features["sub_intent_count"], 4);
        assert_eq!(
            row.features["sub_intents"],
            json!(["search", "implement", "operate_tool", "summarize"])
        );
    }

    #[test]
    fn implementation_language_maps_to_implement_intent() {
        let row = classify_request_event(&event(
            "e-implement",
            "Implement the classifier change in src/request_classification.rs",
            None,
        ));

        assert_eq!(row.intent, RequestIntent::Implement);
        assert_eq!(row.response_contract, ResponseContract::ValidationRequired);
    }

    #[test]
    fn single_intent_with_conjunction_is_not_decomposed() {
        let row = classify_request_event(&event(
            "e-not-composite",
            "Explain Docker and Kubernetes networking",
            None,
        ));

        assert_eq!(row.features["is_composite"], false);
        assert_eq!(row.features["decomposition_candidate"], false);
        assert_eq!(row.features["decomposition_reason"], "none");
        assert_eq!(row.features["sub_intent_count"], 0);
        assert_eq!(row.features["sub_intents"], json!([]));
    }

    #[test]
    fn secret_candidates_set_secret_present() {
        let row = classify_request_event(&event(
            "e-secret",
            "The request included Authorization: Bearer sk-secret-value",
            None,
        ));

        assert!(row.risk.contains(&RequestRisk::SecretPresent));
        assert_eq!(row.recommended_route, RecommendedRoute::RefuseOrGuardrail);
        assert_eq!(row.features["has_secret_candidate"], true);
    }

    #[test]
    fn destructive_commands_set_destructive_command() {
        let row = classify_request_event(&event(
            "e-destructive",
            "Run rm -rf /tmp/agentic-os-cache to clean everything",
            None,
        ));

        assert!(row.risk.contains(&RequestRisk::DestructiveCommand));
        assert_eq!(row.complexity, RequestComplexity::L5HighRisk);
        assert_eq!(row.recommended_route, RecommendedRoute::RefuseOrGuardrail);
    }

    #[test]
    fn latest_current_phrasing_sets_external_current_info_required() {
        let row = classify_request_event(&event(
            "e-latest",
            "What is the latest vLLM release today?",
            None,
        ));

        assert!(row.risk.contains(&RequestRisk::ExternalCurrentInfoRequired));
        assert_eq!(row.domain, RequestDomain::LlmInference);
        assert_eq!(row.recommended_route, RecommendedRoute::WebRequired);
    }

    #[test]
    fn high_stakes_domains_set_high_stakes_risk() {
        for (id, summary, domain) in [
            (
                "e-medical",
                "Can you diagnose this medical symptom?",
                RequestDomain::Medical,
            ),
            (
                "e-legal",
                "Is this legal contract enforceable?",
                RequestDomain::Legal,
            ),
            (
                "e-finance",
                "Should I make this tax and investment move?",
                RequestDomain::Finance,
            ),
        ] {
            let row = classify_request_event(&event(id, summary, None));
            assert_eq!(row.domain, domain);
            assert!(row.risk.contains(&RequestRisk::HighStakes));
            assert_eq!(row.recommended_route, RecommendedRoute::RefuseOrGuardrail);
        }
    }

    #[test]
    fn infrastructure_terms_map_to_expected_domains() {
        let row = classify_request_event(&event(
            "e-domains",
            "kubectl deployment, Docker container, vLLM model, DNS proxy, and JWT auth",
            None,
        ));

        assert_eq!(row.domain, RequestDomain::Kubernetes);
        assert!(row.secondary_domains.contains(&RequestDomain::Docker));
        assert!(row.secondary_domains.contains(&RequestDomain::LlmInference));
        assert!(row.secondary_domains.contains(&RequestDomain::Networking));
        assert!(row.secondary_domains.contains(&RequestDomain::Security));
        assert!(row.secondary_domains.contains(&RequestDomain::Shell));
        assert_eq!(
            row.features["detected_domain_terms"],
            json!([
                "kubernetes",
                "docker",
                "llm_inference",
                "networking",
                "security",
                "shell"
            ])
        );
    }

    #[test]
    fn unknown_or_empty_events_produce_bounded_safe_defaults() {
        let row = classify_request_event(&event("e-empty", "", None));

        assert_eq!(row.intent, RequestIntent::Unknown);
        assert_eq!(row.domain, RequestDomain::Unknown);
        assert_eq!(row.artifact_type, RequestArtifactType::Unknown);
        assert_eq!(row.risk, vec![RequestRisk::Unknown]);
        assert_eq!(row.complexity, RequestComplexity::Unknown);
        assert_eq!(row.recommended_route, RecommendedRoute::Unknown);
        assert_eq!(row.response_contract, ResponseContract::Unknown);
        assert_eq!(row.features["char_count"], 0);
        assert_eq!(row.features["estimated_tokens"], 0);
    }

    #[test]
    fn non_empty_generic_requests_use_safe_fallback_labels() {
        let row = classify_request_event(&event("e-generic", "Can you help with this?", None));

        assert_eq!(row.intent, RequestIntent::Explain);
        assert_eq!(row.domain, RequestDomain::Generic);
        assert_eq!(row.artifact_type, RequestArtifactType::PlainText);
        assert_eq!(row.risk, vec![RequestRisk::None]);
        assert_ne!(row.complexity, RequestComplexity::Unknown);
        assert_ne!(row.recommended_route, RecommendedRoute::Unknown);
        assert_ne!(row.response_contract, ResponseContract::Unknown);
    }

    #[test]
    fn features_do_not_copy_raw_text() {
        let raw_secret = "sk-raw-secret-value";
        let row = classify_request_event(&event(
            "e-privacy",
            &format!("Please classify this Authorization: Bearer {raw_secret}"),
            Some("Raw evidence body should not be copied into features"),
        ));
        let serialized = row.features.to_string();

        assert!(!serialized.contains(raw_secret));
        assert!(!serialized.contains("Raw evidence body"));
        assert!(feature_string_values_are_bounded(&row.features));
    }

    #[test]
    fn classifiable_request_event_selection_is_bounded() {
        let user_event = event("e-user", "hello", None);
        assert!(is_classifiable_request_event(&user_event));

        let empty_user_event = event("e-empty-user", "", None);
        assert!(!is_classifiable_request_event(&empty_user_event));

        let whitespace_user_event = event("e-whitespace-user", " \n\t ", None);
        assert!(!is_classifiable_request_event(&whitespace_user_event));

        let evidence_only_event = event("e-evidence-only", "", Some("hello from evidence"));
        assert!(is_classifiable_request_event(&evidence_only_event));

        let mut request_role = event("e-role", "hello", None);
        request_role.event_type = "checkpoint".to_string();
        request_role.event_role = Some("request".to_string());
        assert!(is_classifiable_request_event(&request_role));

        let mut maintenance = event("e-maint", "summary maintenance", None);
        maintenance.event_type = "summary".to_string();
        maintenance.event_role = None;
        assert!(!is_classifiable_request_event(&maintenance));
    }

    #[test]
    fn backfill_sql_does_not_treat_separator_newline_as_request_text() {
        let source = include_str!("request_classification.rs");

        assert!(source.contains("btrim(coalesce(e.summary, '') || coalesce(e.evidence, '')"));
        assert!(!source.contains("btrim(coalesce(e.summary, '') || E'\\n'"));
    }

    #[test]
    fn live_policy_is_disabled_by_default() {
        let classification = classify_request_event(&event(
            "e-disabled",
            "Can you diagnose this medical issue?",
            None,
        ));

        assert_eq!(
            evaluate_live_policy(&classification, &LivePolicyConfig::default()),
            None
        );
    }

    #[test]
    fn live_policy_handles_high_stakes_current_info_and_destructive_commands() {
        let enabled = LivePolicyConfig {
            enabled: true,
            policy_version: "v1".to_string(),
        };

        let medical = classify_request_event(&event(
            "e-live-medical",
            "Can you diagnose this medical issue?",
            None,
        ));
        let decision = evaluate_live_policy(&medical, &enabled).expect("medical should be stopped");
        assert_eq!(decision.action, "refuse_or_guardrail");
        assert_eq!(decision.reason, "objective_risk");
        assert_eq!(decision.response_contract, ResponseContract::Refusal);

        let latest = classify_request_event(&event(
            "e-live-latest",
            "What is the latest Docker release today?",
            None,
        ));
        let decision = evaluate_live_policy(&latest, &enabled).expect("latest should need web");
        assert_eq!(decision.action, "web_required");
        assert_eq!(decision.reason, "external_current_info_required");

        let destructive = classify_request_event(&event(
            "e-live-destructive",
            "Please run kubectl delete namespace production",
            None,
        ));
        let decision =
            evaluate_live_policy(&destructive, &enabled).expect("destructive should be stopped");
        assert_eq!(decision.action, "refuse_or_guardrail");
        assert_eq!(decision.reason, "objective_risk");
    }

    #[test]
    fn clarification_route_returns_bounded_contract() {
        let enabled = LivePolicyConfig {
            enabled: true,
            policy_version: "v1".to_string(),
        };
        let classification = classify_request_event(&event("e-clarify", "Please restart it", None));

        assert_eq!(
            classification.recommended_route,
            RecommendedRoute::AskClarification
        );
        let decision =
            evaluate_live_policy(&classification, &enabled).expect("clarification should apply");
        assert_eq!(decision.action, "ask_clarification");
        assert_eq!(
            decision.response_contract,
            ResponseContract::ClarificationQuestion
        );
    }

    #[test]
    fn small_and_strong_model_routes_remain_shadow_only() {
        let enabled = LivePolicyConfig {
            enabled: true,
            policy_version: "v1".to_string(),
        };
        let small = classify_request_event(&event(
            "e-small",
            "Please explain how Docker container networking works at a high level",
            None,
        ));
        assert_eq!(small.recommended_route, RecommendedRoute::SmallLocalModel);
        assert_eq!(evaluate_live_policy(&small, &enabled), None);

        let long_text = format!("Explain this architecture. {}", "detail ".repeat(400));
        let strong = classify_request_event(&event("e-strong", &long_text, None));
        assert_eq!(strong.recommended_route, RecommendedRoute::StrongLocalModel);
        assert_eq!(evaluate_live_policy(&strong, &enabled), None);
    }

    #[test]
    fn live_policy_bounding_helpers_reject_unbounded_labels() {
        assert_eq!(
            bounded_live_policy_action("something else"),
            "refuse_or_guardrail"
        );
        assert_eq!(
            bounded_live_policy_reason("raw path /tmp/foo"),
            "objective_risk"
        );
        assert_eq!(bounded_live_policy_bypass("raw unknown"), "shadow_only");
        assert_eq!(bounded_route("bad route"), "unknown");
        assert_eq!(bounded_risk("bad risk"), "unknown");
        assert_eq!(bounded_complexity("bad complexity"), "unknown");
    }

    fn event(id: &str, summary: &str, evidence: Option<&str>) -> AgentEvent {
        AgentEvent {
            id: id.to_string(),
            session_id: "session-1".to_string(),
            repo: "agent-os".to_string(),
            actor: "user".to_string(),
            event_type: "user_message".to_string(),
            summary: summary.to_string(),
            evidence: evidence.map(str::to_string),
            metadata: json!({
                "payload": {
                    "kind": "test"
                }
            }),
            correlation_id: None,
            parent_event_id: None,
            trajectory_id: None,
            attempt_index: None,
            event_role: Some("request".to_string()),
            created_at: DateTime::parse_from_rfc3339("2026-05-23T12:00:00Z")
                .unwrap()
                .with_timezone(&Utc),
            summary_level: 0,
        }
    }

    fn feature_string_values_are_bounded(value: &Value) -> bool {
        match value {
            Value::String(value) => enum_inventory()
                .iter()
                .any(|(_name, variants)| variants.contains(&value.as_str())),
            Value::Array(values) => values.iter().all(feature_string_values_are_bounded),
            Value::Object(values) => values.values().all(feature_string_values_are_bounded),
            _ => true,
        }
    }
}
