# Metrics

`GET /metrics` serves Prometheus text exposition. `GET /metrics/json` keeps the legacy JSON snapshot for local callers. Time histograms use seconds.

| Metric | Type | Labels | Measures |
| --- | --- | --- | --- |
| `http_requests_total` | counter | `endpoint`, `method`, `status` | Requests completed by route template. |
| `http_request_duration_seconds` | histogram | `endpoint`, `method` | End-to-end HTTP latency. |
| `http_requests_in_flight` | gauge | `endpoint` | Active HTTP requests. |
| `auth_attempts_total` | counter | `result` | Accepted or rejected API auth attempts. |
| `upstream_litellm_requests_total` | counter | `path`, `status` | LiteLLM upstream calls. |
| `upstream_litellm_duration_seconds` | histogram | `path` | LiteLLM upstream latency. |
| `upstream_litellm_errors_total` | counter | `path`, `kind` | Timeout, connection, HTTP, and parse failures. |
| `summarizer_upstream_requests_total` | counter | `path`, `status` | Calls to the configured summarizer endpoint. |
| `summarizer_upstream_duration_seconds` | histogram | `path` | Summarizer endpoint latency. |
| `summarizer_upstream_errors_total` | counter | `path`, `kind` | Timeout, connection, HTTP, and parse failures from the summarizer endpoint. |
| `stream_first_token_seconds` | histogram | `path` | Time to first streamed byte. |
| `stream_duration_seconds` | histogram | `path` | Full stream duration. |
| `stream_disconnects_total` | counter | `path`, `reason` | Interrupted streams. |
| `db_pool_size` | gauge | none | Configured Postgres pool capacity. |
| `db_pool_available` | gauge | none | Idle Postgres pool connections. |
| `db_pool_waiters` | gauge | none | Tasks waiting for a pool connection. |
| `db_query_duration_seconds` | histogram | `op` | Logical Postgres operation latency. |
| `db_query_errors_total` | counter | `op` | Logical Postgres operation errors. |
| `qdrant_requests_total` | counter | `op`, `status` | Qdrant API calls. |
| `qdrant_request_duration_seconds` | histogram | `op` | Qdrant API latency. |
| `embedder_inferences_total` | counter | none | Local embedder inference successes. |
| `embedder_inference_duration_seconds` | histogram | none | Local embedder inference latency. |
| `embedder_input_tokens` | histogram | none | Tokenized embedder input length. |
| `sentiment_inferences_total` | counter | `verdict` | Sentiment classifications when loaded. |
| `sentiment_inference_duration_seconds` | histogram | none | Sentiment inference latency. |
| `summarizer_ticks_total` | counter | none | Background summarizer ticks. |
| `summarizer_candidates_found_total` | counter | `target_level` | Candidate summaries found. |
| `summarizer_summaries_written_total` | counter | `target_level`, `result` | Summary write attempts. |
| `summarizer_duration_seconds` | histogram | `target_level` | Summarizer pass latency. |
| `context_pack_requests_total` | counter | none | Context pack requests. |
| `context_pack_cache_hits_total` | counter | none | Context cache hits. |
| `context_pack_cache_misses_total` | counter | none | Context cache misses. |
| `context_cache_replacements_total` | counter | none | Older cache versions replaced after background refresh. |
| `context_pack_build_duration_seconds` | histogram | none | Context build latency. |
| `context_pack_tokens_estimate` | histogram | none | Estimated injected tokens. |
| `context_pack_items_injected_total` | counter | `layer` | Injected L0/L1/L2/L3/failure/remediation/failure_history/operational_constraints items. |
| `retrieval_hits_total` | counter | `source` | Semantic, FTS, and deduped retrieval hits. |
| `inference_tokens_total` | counter | `kind`, `model` | Processed, cached, and generated tokens. |
| `context_cache_stale_invalidations_total` | counter | none | Deprecated cache invalidation counter retained for compatibility. Stale-while-revalidate no longer deletes cached packs on memory writes. |
| `execution_artifacts_total` | counter | `event_type`, `success` | Structured execution artifact events written. |
| `failure_signatures_total` | counter | `signature`, `category` | Canonical failure fingerprints recorded inline on failed outcome events. |
| `patch_lifecycle_total` | counter | `outcome` | Patch outcomes: applied, rejected, or reverted. |
| `validation_results_total` | counter | `validator`, `result` | Deterministic validator outcomes. |
| `remediation_reuse_total` | counter | `signature` | Prior remediations surfaced in context packs. |
| `task_retries` | gauge | none | Currently active retry chains across tasks. |
| `task_retries_total` | counter | `task_type`, `outcome`, `trigger_category` | Historical retry behavior. `task_type` is one of `coding`, `infra`, `recall`, `general`; `outcome` is one of `succeeded`, `abandoned`, `still_active`; `trigger_category` is one of `borrow_checker`, `import_error`, `type_error`, `parse_error`, `unknown`, `none`. |
| `memory_promotions_total` | counter | `result` | Accepted/rejected promotion decisions. |
| `memory_source_coverage` | gauge | none | Fraction of promoted memory with source IDs. |
| `rate_limited_total` | counter | `key_hash` | Inference requests rejected with HTTP 429. |
| `sampling_param_overrides_total` | counter | `parameter`, `reason` | Sampling override hook executions. `parameter` is one of `temperature`, `top_p`, `max_tokens`, `seed`, or `none`; `reason` is `noop` or `overridden_by_orchestrator`. Raw parameter values are never labels. |
| `trajectory_results_total` | counter | `status` | Completed trajectory results. `status` is one of `succeeded`, `abandoned`, `unresolved`, `reverted`. |
| `trajectory_attempts_total` | counter | none | Attempt indexes observed on completed trajectories. |
| `trajectory_validation_failures_total` | counter | `validator_type` | Failed trajectory validations by bounded type. `validator_type` is one of `compile`, `test`, `lint`, `type_check`, `schema`, `static_analysis`, `other`. |
| `trajectory_tokens_total` | counter | `direction` | Non-null model tokens aggregated from trajectory result events. `direction` is `input` or `output`. |
| `trajectory_features_total` | counter | none | Feature records written or updated. |
| `feature_failure_classes_total` | counter | `failure_class` | Feature failure classes observed. `failure_class` is one of `tool_loop`, `user_interruption`, `missing_auth`, `wrong_endpoint`, `summarization_failure`, `migration_failure`, `context_pack_empty`, `context_pack_truncated`, `high_input_tokens`, `slow_upstream_model`, `empty_tool_use_message`, `abandoned_before_model`, `single_model_abandoned_no_tools`, `summarizer_shared_upstream`, `other`. |
| `operational_constraints_injected_total` | counter | `constraint_type` | Operational constraints injected into context packs. `constraint_type` is one of `use_known_auth`, `use_known_endpoint`, `use_known_migration_fix`, `avoid_tool_loop`, `fix_context_retrieval`, `reduce_context_bloat`, `separate_summarizer_upstream`, `handle_user_interruption`, `handle_summarization_failure`. |
| `operational_constraints_suppressed_total` | counter | `constraint_type`, `reason` | Operational constraints suppressed. `reason` is one of `freshness_window`, `recovery_detected`, `priority_cap`, `token_budget`. |
| `feature_extraction_duration_seconds` | histogram | none | Inline feature extraction latency. |
| `feature_extraction_failures_total` | counter | `stage` | Best-effort feature extraction failures. `stage` is one of `extraction`, `constraint_build`, `persistence`. |
| `feature_tag_schema_version_unknown_total` | counter | none | Detection tags skipped because their schema version is unknown. |
| `harness_feedback_signals_total` | counter | `signal_type` | Deterministic harness failure or recovery signals recorded on persisted events. Labels are bounded and never include raw paths, prompts, commands, or user text. |
| `harness_feedback_quarantined_total` | counter | `reason` | Events marked as excluded from future context memory because their harness trace is likely poisoned or benchmark-only. |
| `harness_feedback_learning_records_total` | counter | `status` | Structured harness feedback records available for later offline learning analysis. `status` is one of `recorded`, `quarantined`, or `ignored`. |
| `harness_feedback_repair_runs_total` | counter | `result` | Startup and periodic harness feedback repair runs. `result` is `success` or `failure`. |
| `process_cpu_seconds_total` | counter | none | Process CPU seconds. |
| `process_resident_memory_bytes` | gauge | none | Resident memory bytes. |
| `process_start_time_seconds` | gauge | none | Process start time. |
| `process_open_fds` | gauge | none | Open file descriptors. |

Histogram buckets are explicit. HTTP/context buckets span 1 ms to 60 s, LiteLLM, summarizer upstream, and streaming buckets span 100 ms to 10 min, database/Qdrant/feature extraction buckets span 1 ms to 5 s, embedder/sentiment buckets span 1 ms to 5 s, and token buckets span 16 to 32768 tokens.
