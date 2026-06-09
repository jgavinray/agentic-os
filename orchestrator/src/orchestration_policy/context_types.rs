use serde::{Deserialize, Serialize};

/// Sources of context that may be packed into the request.
///
/// These are eligibility labels, not a retrieval result. A source appearing in
/// this list means the context compiler may draw from it for this request; the
/// compiler still owns scoring, deduplication, and budget decisions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum ContextSource {
    /// Episodic memory from the Total Recall service.
    TotalRecall,
    /// Durable structured events stored in Postgres.
    PostgresEvents,
    /// Semantic recall from the Qdrant vector index.
    QdrantSemantic,
    /// Stable summaries or artifacts produced by the context compiler.
    CompiledSummaries,
    /// Prior context-selection decisions and candidate features.
    ContextLedger,
    /// Derived features from raw request/response capture, when enabled.
    RawCaptureFeatures,
    /// Fallback label for corrupt, missing, or future values.
    #[default]
    Unknown,
}

impl ContextSource {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::TotalRecall => "total_recall",
            Self::PostgresEvents => "postgres_events",
            Self::QdrantSemantic => "qdrant_semantic",
            Self::CompiledSummaries => "compiled_summaries",
            Self::ContextLedger => "context_ledger",
            Self::RawCaptureFeatures => "raw_capture_features",
            Self::Unknown => "unknown",
        }
    }
}
