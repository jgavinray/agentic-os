use super::*;
use std::time::{Duration, Instant};

#[test]
fn context_cache_key_includes_repo_task_and_event_count() {
    assert_eq!(
        context_cache_key("repo", "task", 42),
        "repo:task:42".to_string()
    );
}

#[test]
fn context_cache_key_separates_repos() {
    let a = context_cache_key("repo-a", "task", 1);
    let b = context_cache_key("repo-b", "task", 1);
    assert_ne!(a, b);
}

#[test]
fn context_cache_put_replaces_prior_versions_for_prefix() {
    let cache = ContextCache::new(CONTEXT_CACHE_TTL_MS);
    cache.put(
        context_cache_key("repo-a", "task", 1),
        CachedContext {
            context: "a".to_string(),
            memories: vec![],
            cached_at: Instant::now(),
            stats: ContextPackStats::default(),
        },
    );
    cache.put(
        context_cache_key("repo-a", "task", 2),
        CachedContext {
            context: "new".to_string(),
            memories: vec![],
            cached_at: Instant::now(),
            stats: ContextPackStats::default(),
        },
    );

    assert!(cache.get(&context_cache_key("repo-a", "task", 1)).is_none());
    assert!(cache.get(&context_cache_key("repo-a", "task", 2)).is_some());
}

#[test]
fn context_cache_returns_latest_entry_by_prefix() {
    let cache = ContextCache::new(CONTEXT_CACHE_TTL_MS);
    cache.put(
        context_cache_key("repo-a", "task", 1),
        CachedContext {
            context: "old".to_string(),
            memories: vec![],
            cached_at: Instant::now() - Duration::from_secs(5),
            stats: ContextPackStats::default(),
        },
    );
    cache.put(
        context_cache_key("repo-a", "task", 2),
        CachedContext {
            context: "new".to_string(),
            memories: vec![],
            cached_at: Instant::now(),
            stats: ContextPackStats::default(),
        },
    );

    let cached = cache.latest_by_prefix("repo-a:task:").unwrap();

    assert_eq!(cached.context, "new");
}

#[test]
fn context_cache_coalesces_refreshes() {
    let cache = ContextCache::new(CONTEXT_CACHE_TTL_MS);

    assert!(cache.try_begin_refresh("repo:task".to_string()));
    assert!(!cache.try_begin_refresh("repo:task".to_string()));
    cache.finish_refresh("repo:task");
    assert!(cache.try_begin_refresh("repo:task".to_string()));
}

#[test]
fn task_category_detects_architecture_before_broad() {
    assert_eq!(
        TaskCategory::from_task("architecture refactor for memory"),
        TaskCategory::Architecture
    );
}

#[test]
fn context_policy_small_task_includes_l0_and_l1() {
    let policy = ContextPolicy::for_category(TaskCategory::Narrow);
    assert!(policy.l0_recent_pct > 0);
    assert!(policy.l1_matching_pct > 0);
    assert!(policy.failure_pct > 0);
}

#[test]
fn context_policy_architecture_task_prioritizes_l2_l3() {
    let policy = ContextPolicy::for_category(TaskCategory::Architecture);
    assert!(policy.l2_repo_pct + policy.l3_project_pct > policy.l0_recent_pct);
    assert!(policy.l3_limit > 0);
}

#[test]
fn token_usage_extracts_processed_cached_and_generated_tokens() {
    let usage = TokenUsage::from_openai_value(&serde_json::json!({
        "usage": {
            "prompt_tokens": 100,
            "completion_tokens": 25,
            "prompt_tokens_details": {"cached_tokens": 40}
        }
    }));
    assert_eq!(usage.processed_tokens, 100);
    assert_eq!(usage.cached_tokens, 40);
    assert_eq!(usage.generated_tokens, 25);
}
