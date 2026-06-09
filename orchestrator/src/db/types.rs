use uuid::Uuid;

#[derive(Clone, Debug)]
pub struct AgentEvent {
    pub id: String,
    pub session_id: String,
    pub repo: String,
    pub actor: String,
    pub event_type: String,
    pub summary: String,
    pub evidence: Option<String>,
    pub metadata: serde_json::Value,
    pub correlation_id: Option<Uuid>,
    pub parent_event_id: Option<Uuid>,
    pub trajectory_id: Option<Uuid>,
    pub attempt_index: Option<i32>,
    pub event_role: Option<String>,
    pub created_at: chrono::DateTime<chrono::Utc>,
    pub summary_level: i32,
}

impl AgentEvent {
    pub fn to_memory(&self) -> crate::state::EventMemory {
        crate::state::EventMemory {
            event_type: self.event_type.clone(),
            summary: self.summary.clone(),
            evidence: self.evidence.clone(),
            metadata: self.metadata.clone(),
            created_at: self.created_at,
            summary_level: self.summary_level,
        }
    }

    pub fn payload(&self) -> serde_json::Value {
        serde_json::json!({
            "event_id": self.id,
            "session_id": self.session_id,
            "repo": self.repo,
            "actor": self.actor,
            "event_type": self.event_type,
            "summary": self.summary,
            "evidence": self.evidence,
            "metadata": self.metadata,
            "correlation_id": self.correlation_id.map(|id| id.to_string()),
            "parent_event_id": self.parent_event_id.map(|id| id.to_string()),
            "trajectory_id": self.trajectory_id.map(|id| id.to_string()),
            "attempt_index": self.attempt_index,
            "event_role": self.event_role,
            "created_at": self.created_at,
            "summary_level": self.summary_level
        })
    }

    pub fn vector_text(&self) -> String {
        format!(
            "{}\n{}\n{}\n{}\n{}",
            self.repo,
            self.event_type,
            self.summary,
            self.evidence.as_deref().unwrap_or(""),
            self.metadata
        )
    }
}

#[derive(Default)]
pub struct ContextEvidence {
    pub l0_recent: Vec<AgentEvent>,
    pub l1_matching: Vec<AgentEvent>,
    pub l2_repo: Vec<AgentEvent>,
    pub l3_project: Vec<AgentEvent>,
    pub failures: Vec<AgentEvent>,
    pub failure_history: Vec<FailureHistoryItem>,
    pub operational_constraints: Vec<crate::feature_extraction::OperationalConstraint>,
}

impl ContextEvidence {
    pub fn memories(&self) -> Vec<crate::state::EventMemory> {
        self.l0_recent
            .iter()
            .chain(self.l1_matching.iter())
            .chain(self.l2_repo.iter())
            .chain(self.l3_project.iter())
            .chain(self.failures.iter())
            .chain(
                self.failure_history
                    .iter()
                    .flat_map(|item| std::iter::once(&item.failure).chain(item.remediation.iter())),
            )
            .map(AgentEvent::to_memory)
            .collect()
    }

    pub fn stats(&self) -> crate::state::ContextPackStats {
        crate::state::ContextPackStats {
            l0_items_injected: self.l0_recent.len(),
            l1_items_injected: self.l1_matching.len(),
            l2_items_injected: self.l2_repo.len(),
            l3_items_injected: self.l3_project.len(),
            failed_attempts_injected: self
                .failures
                .iter()
                .filter(|e| e.event_type == "failed_attempt")
                .count(),
            remediations_injected: self
                .failures
                .iter()
                .filter(|e| e.event_type == "remediation")
                .count(),
            failure_history_items_injected: self.failure_history.len(),
            operational_constraints_injected: self.operational_constraints.len(),
            failure_history_remediation_signatures: self
                .failure_history
                .iter()
                .filter_map(|item| item.remediation.as_ref().map(|_| item.signature.clone()))
                .collect(),
            ..Default::default()
        }
    }
}

#[derive(Clone)]
pub struct FailureHistoryItem {
    pub signature: String,
    pub category: String,
    pub failure: AgentEvent,
    pub remediation: Option<AgentEvent>,
}

#[derive(Clone, Debug, serde::Serialize)]
pub struct ContextCompilerLedgerEntry {
    pub id: Uuid,
    pub repo: String,
    pub artifact_type: String,
    pub candidate_source: String,
    pub candidate_id: Option<String>,
    pub decision: String,
    pub reason: String,
    pub artifact_id: Option<Uuid>,
    pub metadata: serde_json::Value,
    pub created_at: chrono::DateTime<chrono::Utc>,
}

#[derive(Clone, Debug)]
pub struct VllmCacheObservationInput {
    pub session_id: Option<String>,
    pub namespace: String,
    pub repo: String,
    pub task: String,
    pub endpoint: String,
    pub requested_model: String,
    pub routed_model: String,
    pub request_event_id: Option<Uuid>,
    pub context_pack_id: Option<Uuid>,
    pub attempt_id: Uuid,
    pub metrics_url: String,
    pub delta: crate::vllm_metrics::VllmCacheDelta,
    pub request_input_tokens: i64,
    pub request_output_tokens: i64,
    pub provider_cache: crate::litellm::ProviderCacheCounters,
}

#[derive(Clone, Debug, Default, serde::Serialize)]
pub struct VllmCacheStats {
    pub observations: i64,
    pub prefix_cache_queries: i64,
    pub prefix_cache_hits: i64,
    pub prompt_tokens_total: i64,
    pub prompt_tokens_cached: i64,
    pub prompt_tokens_local_compute: i64,
    pub prompt_tokens_local_cache_hit: i64,
    pub prompt_tokens_external_kv: i64,
    pub request_input_tokens: i64,
    pub request_output_tokens: i64,
    pub provider_cached_tokens: i64,
    pub provider_cache_created_tokens: i64,
    pub provider_cache_read_tokens: i64,
    pub prefix_cache_hit_rate: f64,
    pub prompt_cached_rate: f64,
}
