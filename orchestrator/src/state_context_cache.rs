use serde::Serialize;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::{Duration, Instant};

use crate::state_context::EventMemory;

#[derive(Debug, Clone, Serialize, Default)]
pub struct ContextPackStats {
    pub build_ms: u64,
    pub context_chars: usize,
    pub context_tokens_estimate: usize,
    pub stable_prefix_hash: Option<String>,
    pub dynamic_tail_hash: Option<String>,
    pub l0_items_injected: usize,
    pub l1_items_injected: usize,
    pub l2_items_injected: usize,
    pub l3_items_injected: usize,
    pub failed_attempts_injected: usize,
    pub remediations_injected: usize,
    pub failure_history_items_injected: usize,
    pub operational_constraints_injected: usize,
    #[serde(skip_serializing)]
    pub failure_history_remediation_signatures: Vec<String>,
    pub retrieval_semantic_hits: usize,
    pub retrieval_fts_hits: usize,
    pub retrieval_deduped_hits: usize,
    pub retrieved_event_ids: Vec<String>,
    pub memory_levels_used: Vec<String>,
    pub injected_failure_signatures: Vec<String>,
    pub token_budget: usize,
    pub truncated: bool,
    pub cache_hit: bool,
}

/// Cached context pack with timestamp.
#[derive(Clone, Debug)]
pub struct CachedContext {
    pub context: String,
    pub memories: Vec<EventMemory>,
    pub cached_at: Instant,
    pub stats: ContextPackStats,
}

/// Context cache keyed by repo:task:version.
#[derive(Clone)]
pub struct ContextCache {
    entries: Arc<std::sync::RwLock<HashMap<String, CachedContext>>>,
    refreshes: Arc<std::sync::Mutex<HashSet<String>>>,
    ttl_ms: u64,
}

impl ContextCache {
    pub fn new(ttl_ms: u64) -> Self {
        Self {
            entries: Arc::new(std::sync::RwLock::new(HashMap::new())),
            refreshes: Arc::new(std::sync::Mutex::new(HashSet::new())),
            ttl_ms,
        }
    }

    pub fn get(&self, key: &str) -> Option<CachedContext> {
        let entries = self.entries.read().unwrap();
        let entry = entries.get(key)?;
        if entry.cached_at.elapsed() < Duration::from_millis(self.ttl_ms) {
            Some(entry.clone())
        } else {
            None
        }
    }

    pub fn latest_by_prefix(&self, prefix: &str) -> Option<CachedContext> {
        let entries = self.entries.read().unwrap();
        entries
            .iter()
            .filter(|(key, _)| key.starts_with(prefix))
            .max_by_key(|(_, entry)| entry.cached_at)
            .map(|(_, entry)| entry.clone())
    }

    pub fn put(&self, key: String, value: CachedContext) -> usize {
        let mut entries = self.entries.write().unwrap();
        let mut replaced = 0;
        if let Some((prefix, _)) = key.rsplit_once(':') {
            let prefix = format!("{prefix}:");
            entries.retain(|existing_key, _| {
                let keep = existing_key == &key || !existing_key.starts_with(&prefix);
                if !keep {
                    replaced += 1;
                }
                keep
            });
        }
        entries.insert(key, value);
        replaced
    }

    pub fn try_begin_refresh(&self, key: String) -> bool {
        let mut refreshes = self.refreshes.lock().unwrap();
        refreshes.insert(key)
    }

    pub fn finish_refresh(&self, key: &str) {
        let mut refreshes = self.refreshes.lock().unwrap();
        refreshes.remove(key);
    }

    pub fn stats(&self) -> CacheStats {
        let entries = self.entries.read().unwrap();
        CacheStats {
            entries: entries.len(),
            ttl_ms: self.ttl_ms,
        }
    }
}

#[derive(Debug, Serialize)]
pub struct CacheStats {
    pub entries: usize,
    pub ttl_ms: u64,
}

pub fn context_cache_key(repo: &str, task: &str, event_count: i64) -> String {
    format!("{repo}:{task}:{event_count}")
}
