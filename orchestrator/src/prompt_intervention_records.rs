/// Prompt intervention record types, validation, and capture-side persistence.
///
/// This module provides:
/// - `PromptInterventionRecord` — the canonical append-only intervention record
/// - `PromptInterventionBackfillSummary` — lightweight backfill summary
/// - Validation helpers
/// - Table init and insert APIs for the capture database
///
/// Per the record contract (docs/PromptInterventions/02-record-contract.md),
/// records are append-only and never updated or deleted.
use deadpool_postgres::Pool;
use sha2::{Digest, Sha256};
use uuid::Uuid;

use crate::prompt_intervention_taxonomy::{
    BurdenType, FailureRelation, InterventionType, LabelerType, SignalFamily, SignalStrength,
    SourceKind,
};

/// Maximum length for evidence excerpts.
pub const EVIDENCE_EXCERPT_MAX: usize = 500;

// ── Record ──────────────────────────────────────────────────────

/// A single append-only prompt intervention record.
///
/// This is a derived interpretation — the raw exchange remains the source of truth.
#[derive(Clone, Debug)]
pub struct PromptInterventionRecord {
    /// Unique intervention record ID.
    pub id: Uuid,
    /// Exact raw exchange ID that produced the evidence.
    pub exchange_id: Uuid,
    /// Optional trajectory ID.
    pub trajectory_id: Option<Uuid>,
    /// Optional request event ID.
    pub request_event_id: Option<Uuid>,
    /// Optional attempt ID.
    pub attempt_id: Option<Uuid>,
    /// Optional requested model.
    pub requested_model: Option<String>,
    /// Optional routed model.
    pub routed_model: Option<String>,
    /// Optional baseline arm.
    pub baseline_arm: Option<String>,
    /// Optional selected route.
    pub selected_route: Option<String>,
    /// Optional routing policy version.
    pub routing_policy_version: Option<String>,
    /// Exact prompt hash (SHA-256 hex).
    pub exact_prompt_hash: String,
    /// Normalized prompt hash (SHA-256 hex after volatile-value removal).
    pub normalized_prompt_hash: String,
    /// Prompt fingerprint version.
    pub prompt_fingerprint_version: u32,
    /// Where the evidence came from.
    pub source_kind: SourceKind,
    /// The kind of intervention observed.
    pub intervention_type: InterventionType,
    /// The broad category of steering or control.
    pub signal_family: SignalFamily,
    /// The specific signal type (free-text label within the family).
    pub signal_type: String,
    /// How directly the prompt states the steering.
    pub signal_strength: SignalStrength,
    /// What kind of burden or failure the intervention represents.
    pub burden_type: BurdenType,
    /// How the steering relates to failures.
    pub failure_relation: FailureRelation,
    /// Optional target behavior.
    pub target_behavior: Option<String>,
    /// Optional blocked behavior.
    pub blocked_behavior: Option<String>,
    /// Optional replacement behavior.
    pub replacement_behavior: Option<String>,
    /// Bounded evidence excerpt (redacted, short).
    pub evidence_excerpt: String,
    /// Evidence hash (SHA-256 hex of the evidence excerpt).
    pub evidence_hash: String,
    /// What produced the label.
    pub labeler_type: LabelerType,
    /// Confidence score in \[0.0, 1.0\].
    pub confidence: f64,
    /// Taxonomy version string.
    pub taxonomy_version: String,
    /// Optional pointer to a superseded record.
    pub supersedes_record_id: Option<Uuid>,
    /// Creation timestamp.
    pub created_at: chrono::DateTime<chrono::Utc>,
}

/// Compute the SHA-256 hex digest of a string.
pub fn sha256_hex(s: &str) -> String {
    format!("{:x}", Sha256::digest(s))
}

/// Check whether a string contains obvious unredacted secrets.
fn contains_unredacted_secrets(s: &str) -> bool {
    let lower = s.to_lowercase();
    // Bearer tokens
    if lower.contains("bearer ") {
        return true;
    }
    // Authorization headers
    if lower.contains("authorization:") || lower.contains("authorization: ") {
        return true;
    }
    // Cookies
    if lower.contains("cookie:") || lower.contains("cookie: ") || lower.contains("set-cookie") {
        return true;
    }
    // password=
    if lower.contains("password=") {
        return true;
    }
    // api_key=
    if lower.contains("api_key=") {
        return true;
    }
    // sk- style keys (OpenAI, Anthropic, etc.)
    if lower.contains("sk-") {
        return true;
    }
    false
}

impl PromptInterventionRecord {
    /// Validates the record invariants before persistence.
    pub fn validate(&self) -> Result<(), anyhow::Error> {
        if self.exact_prompt_hash.is_empty() {
            anyhow::bail!("exact_prompt_hash must not be empty");
        }
        if self.normalized_prompt_hash.is_empty() {
            anyhow::bail!("normalized_prompt_hash must not be empty");
        }
        if self.prompt_fingerprint_version == 0 {
            anyhow::bail!("prompt_fingerprint_version must be \u{2265} 1");
        }
        if self.signal_type.is_empty() {
            anyhow::bail!("signal_type must not be empty");
        }
        if self.evidence_hash.is_empty() {
            anyhow::bail!("evidence_hash must not be empty");
        }
        if self.taxonomy_version.is_empty() {
            anyhow::bail!("taxonomy_version must not be empty");
        }
        if self.confidence < 0.0 || self.confidence > 1.0 {
            anyhow::bail!("confidence must be in [0.0, 1.0]");
        }
        // Evidence excerpt must be bounded.
        if self.evidence_excerpt.len() > EVIDENCE_EXCERPT_MAX {
            anyhow::bail!(
                "evidence_excerpt exceeds {} character limit",
                EVIDENCE_EXCERPT_MAX
            );
        }
        // evidence_hash must equal sha256(evidence_excerpt).
        let expected_hash = sha256_hex(&self.evidence_excerpt);
        if self.evidence_hash != expected_hash {
            anyhow::bail!("evidence_hash mismatch: expected sha256(evidence_excerpt)");
        }
        // Reject obvious unredacted secrets in evidence_excerpt.
        if contains_unredacted_secrets(&self.evidence_excerpt) {
            anyhow::bail!("evidence_excerpt contains unredacted secrets");
        }
        // signal_family must match intervention_type primary_signal_family.
        // For Other: allow any non-NoSignal family only when evidence_excerpt is non-empty;
        // do not map Other to NoSignal by default for persisted records.
        let primary = self.intervention_type.primary_signal_family();
        if self.intervention_type == InterventionType::Other {
            // Other may pair with any non-NoSignal family when evidence is present.
            if self.signal_family == SignalFamily::NoSignal {
                anyhow::bail!(
                    "Other intervention_type must not map to NoSignal for persisted records"
                );
            }
            if self.evidence_excerpt.is_empty() {
                anyhow::bail!("Other intervention_type requires non-empty evidence_excerpt");
            }
        } else {
            if self.signal_family != primary {
                anyhow::bail!(
                    "signal_family {:?} does not match intervention_type {:?} primary {:?})",
                    self.signal_family,
                    self.intervention_type,
                    primary,
                );
            }
        }
        Ok(())
    }
}

// ── Backfill Summary ───────────────────────────────────────────

/// Lightweight summary of a backfill run.
#[derive(Clone, Debug)]
pub struct PromptInterventionBackfillSummary {
    /// Backfill run ID.
    pub run_id: Uuid,
    /// Number of records inserted during this backfill.
    pub records_inserted: u64,
    /// Number of exchanges scanned.
    pub exchanges_scanned: u64,
    /// Taxonomy version used.
    pub taxonomy_version: String,
    /// Labeler type for all records in this backfill.
    pub labeler_type: LabelerType,
    /// Timestamp when the backfill completed.
    pub completed_at: chrono::DateTime<chrono::Utc>,
}

// ── Capture-side table init ────────────────────────────────────

/// All valid source_kind string values for CHECK constraint.
const SOURCE_KIND_VALUES: &[&str] = &[
    "raw_prompt",
    "user_message",
    "assistant_message",
    "tool_result",
    "posthoc_review",
];

/// All valid intervention_type string values for CHECK constraint.
const INTERVENTION_TYPE_VALUES: &[&str] = &[
    "stop_and_redirect",
    "scope_narrowing",
    "prompt_repair",
    "quality_gate",
    "risk_warning",
    "clarification_request",
    "implementation_block",
    "validation_requirement",
    "model_failure_correction",
    "other",
];

/// All valid signal_family string values for CHECK constraint.
const SIGNAL_FAMILY_VALUES: &[&str] = &[
    "steering",
    "failure_correction",
    "risk_control",
    "validation_pressure",
    "context_pressure",
    "no_signal",
];

/// All valid signal_strength string values for CHECK constraint.
const SIGNAL_STRENGTH_VALUES: &[&str] = &["explicit", "implicit", "ambiguous"];

/// All valid burden_type string values for CHECK constraint.
const BURDEN_TYPE_VALUES: &[&str] = &[
    "human_prompt_repair",
    "human_scope_control",
    "human_stop_control",
    "human_validation_control",
    "human_risk_control",
    "context_recovery",
    "model_error_recovery",
    "unknown_burden",
];

/// All valid failure_relation string values for CHECK constraint.
const FAILURE_RELATION_VALUES: &[&str] = &[
    "prevention",
    "correction",
    "recovery",
    "quality_control",
    "risk_control",
    "unknown_relation",
];

/// All valid labeler_type string values for CHECK constraint.
const LABELER_TYPE_VALUES: &[&str] = &["rule", "posthoc", "human", "local_model", "frontier_model"];

fn check_constraint(col: &str, values: &[&str]) -> String {
    let list: Vec<String> = values.iter().map(|v| format!("'{}'", v)).collect();
    format!("CHECK ({} IN ({}) )", col, list.join(", "))
}

/// Initialize the prompt_interventions table in the capture database.
///
/// This is called during startup only when CAPTURE_DATABASE_URL exists.
pub async fn init(pool: &Pool) -> Result<(), anyhow::Error> {
    let sk_check = check_constraint("source_kind", SOURCE_KIND_VALUES);
    let it_check = check_constraint("intervention_type", INTERVENTION_TYPE_VALUES);
    let sf_check = check_constraint("signal_family", SIGNAL_FAMILY_VALUES);
    let ss_check = check_constraint("signal_strength", SIGNAL_STRENGTH_VALUES);
    let bt_check = check_constraint("burden_type", BURDEN_TYPE_VALUES);
    let fr_check = check_constraint("failure_relation", FAILURE_RELATION_VALUES);
    let lt_check = check_constraint("labeler_type", LABELER_TYPE_VALUES);

    let conn = pool.get().await?;
    conn.batch_execute(&format!(
        "CREATE TABLE IF NOT EXISTS prompt_interventions (
            id UUID PRIMARY KEY,
            exchange_id UUID NOT NULL,
            trajectory_id UUID,
            request_event_id UUID,
            attempt_id UUID,
            requested_model TEXT,
            routed_model TEXT,
            baseline_arm TEXT,
            selected_route TEXT,
            routing_policy_version TEXT,
            exact_prompt_hash TEXT NOT NULL,
            normalized_prompt_hash TEXT NOT NULL,
            prompt_fingerprint_version INT NOT NULL CHECK (prompt_fingerprint_version >= 1),
            source_kind TEXT NOT NULL {sk_check},
            intervention_type TEXT NOT NULL {it_check},
            signal_family TEXT NOT NULL {sf_check},
            signal_type TEXT NOT NULL,
            signal_strength TEXT NOT NULL {ss_check},
            burden_type TEXT NOT NULL {bt_check},
            failure_relation TEXT NOT NULL {fr_check},
            target_behavior TEXT,
            blocked_behavior TEXT,
            replacement_behavior TEXT,
            evidence_excerpt TEXT NOT NULL CHECK (char_length(evidence_excerpt) <= {EVIDENCE_EXCERPT_MAX}),
            evidence_hash TEXT NOT NULL,
            labeler_type TEXT NOT NULL {lt_check},
            confidence DOUBLE PRECISION NOT NULL CHECK (confidence >= 0.0 AND confidence <= 1.0),
            taxonomy_version TEXT NOT NULL,
            supersedes_record_id UUID,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now()
        );
        CREATE INDEX IF NOT EXISTS prompt_interventions_exchange_id_idx
            ON prompt_interventions(exchange_id);
        CREATE INDEX IF NOT EXISTS prompt_interventions_signal_family_idx
            ON prompt_interventions(signal_family);
        CREATE INDEX IF NOT EXISTS prompt_interventions_created_at_idx
            ON prompt_interventions(created_at DESC);
        CREATE INDEX IF NOT EXISTS prompt_interventions_trajectory_id_idx
            ON prompt_interventions(trajectory_id);
        CREATE INDEX IF NOT EXISTS prompt_interventions_normalized_prompt_hash_idx
            ON prompt_interventions(normalized_prompt_hash);",
    ))
    .await?;
    init_backfill_summaries(pool).await?;
    Ok(())
}

/// Insert a single prompt intervention record into the capture database.
pub async fn insert(pool: &Pool, record: &PromptInterventionRecord) -> Result<(), anyhow::Error> {
    record.validate()?;
    let conn = pool.get().await?;
    conn.execute(
        "INSERT INTO prompt_interventions
         (id, exchange_id, trajectory_id, request_event_id, attempt_id,
          requested_model, routed_model, baseline_arm, selected_route,
          routing_policy_version, exact_prompt_hash, normalized_prompt_hash,
          prompt_fingerprint_version, source_kind, intervention_type,
          signal_family, signal_type, signal_strength, burden_type,
          failure_relation, target_behavior, blocked_behavior,
          replacement_behavior, evidence_excerpt, evidence_hash,
          labeler_type, confidence, taxonomy_version, supersedes_record_id, created_at)
         VALUES
         ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18, $19, $20, $21, $22, $23, $24, $25, $26, $27, $28, $29, $30)",
        &[
            &record.id,
            &record.exchange_id,
            &record.trajectory_id,
            &record.request_event_id,
            &record.attempt_id,
            &record.requested_model,
            &record.routed_model,
            &record.baseline_arm,
            &record.selected_route,
            &record.routing_policy_version,
            &record.exact_prompt_hash,
            &record.normalized_prompt_hash,
            &record.prompt_fingerprint_version,
            &record.source_kind.as_str(),
            &record.intervention_type.as_str(),
            &record.signal_family.as_str(),
            &record.signal_type,
            &record.signal_strength.as_str(),
            &record.burden_type.as_str(),
            &record.failure_relation.as_str(),
            &record.target_behavior,
            &record.blocked_behavior,
            &record.replacement_behavior,
            &record.evidence_excerpt,
            &record.evidence_hash,
            &record.labeler_type.as_str(),
            &record.confidence,
            &record.taxonomy_version,
            &record.supersedes_record_id,
            &record.created_at,
        ],
    )
    .await?;
    Ok(())
}

/// Best-effort insert that logs warnings on failure.
pub async fn insert_best_effort(pool: Option<&Pool>, record: PromptInterventionRecord) {
    let Some(pool) = pool else {
        return;
    };
    if let Err(e) = insert(pool, &record).await {
        tracing::warn!(
            target: "prompt_intervention",
            record_id = %record.id,
            exchange_id = %record.exchange_id,
            "failed to insert prompt intervention record: {e}"
        );
    }
}

// ── Backfill Summary table init & insert ────────────────────────

/// Initialize the prompt_intervention_backfill_summaries table.
pub async fn init_backfill_summaries(pool: &Pool) -> Result<(), anyhow::Error> {
    let lt_check = check_constraint("labeler_type", LABELER_TYPE_VALUES);
    let conn = pool.get().await?;
    conn.batch_execute(&format!(
        "CREATE TABLE IF NOT EXISTS prompt_intervention_backfill_summaries (
            run_id UUID PRIMARY KEY,
            records_inserted BIGINT NOT NULL CHECK (records_inserted >= 0),
            exchanges_scanned BIGINT NOT NULL CHECK (exchanges_scanned >= 0),
            taxonomy_version TEXT NOT NULL,
            labeler_type TEXT NOT NULL {lt_check},
            completed_at TIMESTAMPTZ NOT NULL
        );",
    ))
    .await?;
    Ok(())
}

/// Insert a backfill summary into the capture database.
pub async fn insert_backfill_summary(
    pool: &Pool,
    summary: &PromptInterventionBackfillSummary,
) -> Result<(), anyhow::Error> {
    let conn = pool.get().await?;
    conn.execute(
        "INSERT INTO prompt_intervention_backfill_summaries
         (run_id, records_inserted, exchanges_scanned, taxonomy_version, labeler_type, completed_at)
         VALUES ($1, $2, $3, $4, $5, $6)",
        &[
            &summary.run_id,
            &(summary.records_inserted as i64),
            &(summary.exchanges_scanned as i64),
            &summary.taxonomy_version,
            &summary.labeler_type.as_str(),
            &summary.completed_at,
        ],
    )
    .await?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_record() -> PromptInterventionRecord {
        let evidence = "prompt contained scope narrowing instruction";
        let evidence_hash = sha256_hex(evidence);
        PromptInterventionRecord {
            id: Uuid::new_v4(),
            exchange_id: Uuid::new_v4(),
            trajectory_id: None,
            request_event_id: None,
            attempt_id: None,
            requested_model: None,
            routed_model: None,
            baseline_arm: None,
            selected_route: None,
            routing_policy_version: None,
            exact_prompt_hash: "abc123".to_string(),
            normalized_prompt_hash: "def456".to_string(),
            prompt_fingerprint_version: 1,
            source_kind: SourceKind::UserMessage,
            intervention_type: InterventionType::ScopeNarrowing,
            signal_family: SignalFamily::Steering,
            signal_type: "file_scope_reduction".to_string(),
            signal_strength: SignalStrength::Explicit,
            burden_type: BurdenType::HumanScopeControl,
            failure_relation: FailureRelation::Prevention,
            target_behavior: None,
            blocked_behavior: None,
            replacement_behavior: None,
            evidence_excerpt: evidence.to_string(),
            evidence_hash,
            labeler_type: LabelerType::Rule,
            confidence: 0.95,
            taxonomy_version: "1.0.0".to_string(),
            supersedes_record_id: None,
            created_at: chrono::Utc::now(),
        }
    }

    #[test]
    fn test_sha256_hex() {
        let h = sha256_hex("hello");
        assert_eq!(
            h,
            "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"
        );
    }

    #[test]
    fn test_validate_ok() {
        assert!(sample_record().validate().is_ok());
    }

    #[test]
    fn test_validate_empty_exact_prompt_hash() {
        let mut r = sample_record();
        r.exact_prompt_hash = String::new();
        assert!(r.validate().is_err());
    }

    #[test]
    fn test_validate_empty_normalized_prompt_hash() {
        let mut r = sample_record();
        r.normalized_prompt_hash = String::new();
        assert!(r.validate().is_err());
    }

    #[test]
    fn test_validate_zero_fingerprint_version() {
        let mut r = sample_record();
        r.prompt_fingerprint_version = 0;
        assert!(r.validate().is_err());
    }

    #[test]
    fn test_validate_empty_signal_type() {
        let mut r = sample_record();
        r.signal_type = String::new();
        assert!(r.validate().is_err());
    }

    #[test]
    fn test_validate_empty_evidence_hash() {
        let mut r = sample_record();
        r.evidence_hash = String::new();
        assert!(r.validate().is_err());
    }

    #[test]
    fn test_validate_empty_taxonomy_version() {
        let mut r = sample_record();
        r.taxonomy_version = String::new();
        assert!(r.validate().is_err());
    }

    #[test]
    fn test_validate_confidence_below_zero() {
        let mut r = sample_record();
        r.confidence = -0.1;
        assert!(r.validate().is_err());
    }

    #[test]
    fn test_validate_confidence_above_one() {
        let mut r = sample_record();
        r.confidence = 1.1;
        assert!(r.validate().is_err());
    }

    #[test]
    fn test_validate_confidence_boundary_zero() {
        let mut r = sample_record();
        r.confidence = 0.0;
        assert!(r.validate().is_ok());
    }

    #[test]
    fn test_validate_confidence_boundary_one() {
        let mut r = sample_record();
        r.confidence = 1.0;
        assert!(r.validate().is_ok());
    }

    // ── Evidence hash validation ──────────────────────────────

    #[test]
    fn test_validate_evidence_hash_mismatch() {
        let mut r = sample_record();
        r.evidence_hash = "wrong_hash".to_string();
        assert!(r.validate().is_err());
    }

    #[test]
    fn test_validate_evidence_hash_matches() {
        let r = sample_record();
        assert!(r.validate().is_ok());
    }

    // ── Evidence excerpt length ────────────────────────────────

    #[test]
    fn test_validate_evidence_excerpt_too_long() {
        let mut r = sample_record();
        r.evidence_excerpt = "x".repeat(EVIDENCE_EXCERPT_MAX + 1);
        r.evidence_hash = sha256_hex(&r.evidence_excerpt);
        assert!(r.validate().is_err());
    }

    #[test]
    fn test_validate_evidence_excerpt_max_length() {
        let mut r = sample_record();
        r.evidence_excerpt = "x".repeat(EVIDENCE_EXCERPT_MAX);
        r.evidence_hash = sha256_hex(&r.evidence_excerpt);
        assert!(r.validate().is_ok());
    }

    // ── Secret detection ───────────────────────────────────────

    #[test]
    fn test_validate_rejects_bearer_token() {
        let mut r = sample_record();
        r.evidence_excerpt = "auth: Bearer sk-abc123token".to_string();
        r.evidence_hash = sha256_hex(&r.evidence_excerpt);
        assert!(r.validate().is_err());
    }

    #[test]
    fn test_validate_rejects_authorization_header() {
        let mut r = sample_record();
        r.evidence_excerpt = "Authorization: Bearer xyz".to_string();
        r.evidence_hash = sha256_hex(&r.evidence_excerpt);
        assert!(r.validate().is_err());
    }

    #[test]
    fn test_validate_rejects_cookie() {
        let mut r = sample_record();
        r.evidence_excerpt = "Cookie: session=abc123".to_string();
        r.evidence_hash = sha256_hex(&r.evidence_excerpt);
        assert!(r.validate().is_err());
    }

    #[test]
    fn test_validate_rejects_password() {
        let mut r = sample_record();
        r.evidence_excerpt = "password=supersecret123".to_string();
        r.evidence_hash = sha256_hex(&r.evidence_excerpt);
        assert!(r.validate().is_err());
    }

    #[test]
    fn test_validate_rejects_api_key() {
        let mut r = sample_record();
        r.evidence_excerpt = "api_key=sk-live-abc123".to_string();
        r.evidence_hash = sha256_hex(&r.evidence_excerpt);
        assert!(r.validate().is_err());
    }

    #[test]
    fn test_validate_rejects_sk_key() {
        let mut r = sample_record();
        r.evidence_excerpt = "key is sk-proj-abc123def456".to_string();
        r.evidence_hash = sha256_hex(&r.evidence_excerpt);
        assert!(r.validate().is_err());
    }

    #[test]
    fn test_validate_clean_evidence_passes() {
        let mut r = sample_record();
        r.evidence_excerpt = "prompt asked to narrow scope to config files only".to_string();
        r.evidence_hash = sha256_hex(&r.evidence_excerpt);
        assert!(r.validate().is_ok());
    }

    // ── Signal family consistency ──────────────────────────────

    #[test]
    fn test_validate_signal_family_mismatch() {
        let mut r = sample_record();
        r.intervention_type = InterventionType::ScopeNarrowing;
        r.signal_family = SignalFamily::RiskControl;
        assert!(r.validate().is_err());
    }

    #[test]
    fn test_validate_signal_family_matches() {
        let r = sample_record();
        assert!(r.validate().is_ok());
    }

    #[test]
    fn test_validate_other_with_nosignal_rejected() {
        let mut r = sample_record();
        r.intervention_type = InterventionType::Other;
        r.signal_family = SignalFamily::NoSignal;
        assert!(r.validate().is_err());
    }

    #[test]
    fn test_validate_other_with_empty_evidence_rejected() {
        let mut r = sample_record();
        r.intervention_type = InterventionType::Other;
        r.signal_family = SignalFamily::Steering;
        r.evidence_excerpt = String::new();
        r.evidence_hash = sha256_hex(&r.evidence_excerpt);
        assert!(r.validate().is_err());
    }

    #[test]
    fn test_validate_other_with_valid_family_and_evidence_ok() {
        let mut r = sample_record();
        r.intervention_type = InterventionType::Other;
        r.signal_family = SignalFamily::Steering;
        // evidence_excerpt is already non-empty from sample_record
        assert!(r.validate().is_ok());
    }

    #[test]
    fn test_validate_other_with_risk_control_and_evidence_ok() {
        let mut r = sample_record();
        r.intervention_type = InterventionType::Other;
        r.signal_family = SignalFamily::RiskControl;
        assert!(r.validate().is_ok());
    }

    // ── Record does not store raw prompts ──────────────────────

    #[test]
    fn test_record_does_not_store_raw_prompts() {
        let r = sample_record();
        assert!(!r.exact_prompt_hash.is_empty());
        assert!(!r.normalized_prompt_hash.is_empty());
        assert!(r.evidence_excerpt.len() <= EVIDENCE_EXCERPT_MAX);
    }

    // ── Backfill summary ───────────────────────────────────────

    #[test]
    fn test_backfill_summary_creation() {
        let summary = PromptInterventionBackfillSummary {
            run_id: Uuid::new_v4(),
            records_inserted: 42,
            exchanges_scanned: 100,
            taxonomy_version: "1.0.0".to_string(),
            labeler_type: LabelerType::Posthoc,
            completed_at: chrono::Utc::now(),
        };
        assert_eq!(summary.records_inserted, 42);
        assert_eq!(summary.exchanges_scanned, 100);
        assert_eq!(summary.labeler_type, LabelerType::Posthoc);
    }

    // ── Superseded record pointer ──────────────────────────────

    #[test]
    fn test_superseded_record_pointer() {
        let original_id = Uuid::new_v4();
        let mut r = sample_record();
        r.supersedes_record_id = Some(original_id);
        assert!(r.validate().is_ok());
        assert_eq!(r.supersedes_record_id, Some(original_id));
    }

    // ── contains_unredacted_secrets edge cases ─────────────────

    #[test]
    fn test_contains_unredacted_secrets_case_insensitive() {
        assert!(contains_unredacted_secrets("BEARER token123"));
        assert!(contains_unredacted_secrets("Authorization: xyz"));
        assert!(contains_unredacted_secrets("PASSWORD=secret"));
        assert!(contains_unredacted_secrets("API_KEY=key123"));
        assert!(contains_unredacted_secrets("sk-test-abc"));
    }

    #[test]
    fn test_contains_unredacted_secrets_clean() {
        assert!(!contains_unredacted_secrets("prompt asked to narrow scope"));
        assert!(!contains_unredacted_secrets("fix the build error"));
        assert!(!contains_unredacted_secrets("add tests for the API"));
    }
}
