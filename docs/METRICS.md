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
| `context_pack_build_duration_seconds` | histogram | none | Context build latency. |
| `context_pack_tokens_estimate` | histogram | none | Estimated injected tokens. |
| `context_pack_items_injected_total` | counter | `layer` | Injected L0/L1/L2/L3/failure/remediation items. |
| `retrieval_hits_total` | counter | `source` | Semantic, FTS, and deduped retrieval hits. |
| `inference_tokens_total` | counter | `kind`, `model` | Processed, cached, and generated tokens. |
| `context_cache_stale_invalidations_total` | counter | none | Cache invalidations caused by memory writes. |
| `memory_promotions_total` | counter | `result` | Accepted/rejected promotion decisions. |
| `memory_source_coverage` | gauge | none | Fraction of promoted memory with source IDs. |
| `rate_limited_total` | counter | `key_hash` | Inference requests rejected with HTTP 429. |
| `process_cpu_seconds_total` | counter | none | Process CPU seconds. |
| `process_resident_memory_bytes` | gauge | none | Resident memory bytes. |
| `process_start_time_seconds` | gauge | none | Process start time. |
| `process_open_fds` | gauge | none | Open file descriptors. |

Histogram buckets are explicit. HTTP/context buckets span 1 ms to 60 s, LiteLLM and streaming buckets span 100 ms to 10 min, database/Qdrant buckets span 1 ms to 5 s, embedder/sentiment buckets span 1 ms to 5 s, and token buckets span 16 to 32768 tokens.
