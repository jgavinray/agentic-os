# Operations

## Single-Node Contract

Exactly one orchestrator process may own a given Postgres database. Startup acquires a Postgres advisory lock before running migrations or serving traffic. A second orchestrator pointed at the same database exits with a clear error instead of running background work.

Postgres, Qdrant, LiteLLM, and the local llama.cpp summarizer can run as ordinary single-node services. The in-process context cache, summarizer loop, and startup migration runner are intentionally not distributed.

## Summarizer

The summarizer ticks every 60 seconds when `SUMMARIZER_ENABLED=true`. It scans for sessions with enough unsummarized source events, locks each session/target level with a Postgres advisory lock, writes summary events, and marks source events summarized. Candidate sessions are summarized serially so background summarization cannot spawn an unbounded burst of model calls.

Compose runs summarization through a dedicated llama.cpp sidecar at `http://summarizer:8080/v1` by default. `./setup-models.sh` downloads `qwen2.5-3b-instruct-q4_k_m.gguf` into `models/summarizer`, which is mounted read-only into that container. The host-facing summarizer port is `SUMMARIZER_PORT`, default `8089`.

Set `SUMMARIZER_ENABLED=false` to pause memory compaction. Set `SUMMARIZER_BASE_URL` to a different OpenAI-compatible endpoint only when you intentionally want summaries to run somewhere else, such as LiteLLM or a GPU-backed summarizer service.

## Context Cache

The cache is in process only. Full context packs are derived state and are not rebuilt on the user-facing request path. Model requests use the newest cached pack for the repo/task/session scope when one exists, fall back to a minimal context when none exists, and enqueue a coalesced background refresh for the next turn. The explicit `/context-pack` endpoint still builds synchronously because its contract is to return the generated pack.

Stored cache entries are keyed by `repo:task:event_count`; limit overrides become part of the task string. `CONTEXT_CACHE_TTL_MS` controls expiration for exact cache hits. Request-time lookup may use the latest stale entry by key prefix while a background refresh catches up. Writes to memory do not delete cached packs; refreshed packs replace older event-count versions for the same repo/task prefix.

Trajectory request events are still durably inserted into Postgres before forwarding, but their Qdrant indexing runs as background best-effort work.

## Migrations

Schema migrations are embedded with refinery from `orchestrator/migrations/`. The first migration is the baseline schema. On legacy databases created by the old bootstrap DDL, the orchestrator detects the existing tables and marks the baseline as applied without rerunning it.

The main orchestrator runs migrations at boot before serving traffic. When feature extraction is enabled, it then runs an idempotent startup feature backfill before binding the HTTP listener. That job applies deterministic bootstrap tags to historical events, rebuilds `agent_feature_records`, and logs `startup feature backfill completed` with scanned and updated counts. If the startup backfill fails, the orchestrator refuses to serve traffic.

To add a migration:

1. Create `orchestrator/migrations/V<N>__short_name.sql`.
2. Use forward-only SQL.
3. Do not edit previously applied migrations.
4. Run `cargo test`, `cargo clippy -- -D warnings`, and a fresh compose boot.

## Maintenance CLI

`orchestrator-maint` is the one-off maintenance binary built from `orchestrator/src/bin/orchestrator-maint.rs`. It connects directly to Postgres through `DATABASE_URL`, runs embedded migrations, and performs idempotent backfills or rebuilds for operator-driven repair, dry runs, or scoped rebuilds.

Run it locally with Cargo:

```bash
cd orchestrator
cargo run --bin orchestrator-maint -- backfill-signatures --dry-run
cargo run --bin orchestrator-maint -- extract-features --dry-run
cargo run --bin orchestrator-maint -- classify-harness-feedback --dry-run
```

In a packaged deployment, run the built `orchestrator-maint` executable with the same environment as the orchestrator service.

With the Docker Compose stack in this repo, prefer a one-off container on the same network:

```bash
docker compose run --rm --no-deps --entrypoint /usr/local/bin/orchestrator-maint orchestrator extract-features --dry-run
docker compose run --rm --no-deps --entrypoint /usr/local/bin/orchestrator-maint orchestrator extract-features
docker compose run --rm --no-deps --entrypoint /usr/local/bin/orchestrator-maint orchestrator classify-harness-feedback --dry-run
```

If the orchestrator container is already running and the image contains `/usr/local/bin/orchestrator-maint`, this is equivalent:

```bash
docker compose exec orchestrator /usr/local/bin/orchestrator-maint extract-features
```

Current commands:

- `backfill-signatures`: rewrites historical failed outcome events to carry inline deterministic failure signatures.
- `extract-features`: bootstrap-tags historical operational events and rebuilds `agent_feature_records`.
- `classify-harness-feedback`: reclassifies historical events with bounded harness feedback signals and quarantines poisoned benchmark traces from future context memory.

These commands are safe to re-run. Use `--dry-run` first when inspecting a production database.

New events do not require this CLI. The live orchestrator annotates new event metadata and runs best-effort inline feature extraction after event writes. Startup also runs a full idempotent feature backfill by default before serving traffic, and harness feedback repair runs automatically at startup plus periodically in the background. The CLI is mainly for explicit dry-runs, scoped repair, and manual rebuilds.

## Backup And Restore

Run `scripts/backup.sh` against a live stack. It writes a timestamped tarball to `./backups` unless `BACKUP_DIR` is set. The tarball contains a Postgres dump and a Qdrant collection snapshot.

Restore with `scripts/restore.sh ./backups/<file>.tar.gz` against a running or freshly started stack. The restore script verifies the archive structure, recreates the Postgres public schema, restores the SQL dump, uploads the Qdrant snapshot, and verifies both stores are queryable.

Recommended cron entry:

```cron
17 * * * * cd /path/to/agentic-os && BACKUP_DIR=/path/to/agentic-os/backups ./scripts/backup.sh >> ./backups/backup.log 2>&1
```

Round-trip procedure:

1. Start the stack with `docker compose up -d`.
2. Append a known event and verify it appears in `/search`.
3. Run `scripts/backup.sh`.
4. Stop the stack and move `./data` aside.
5. Start the stack again and run `scripts/restore.sh <backup.tar.gz>`.
6. Query `/search` for the known event and verify it is returned.

## Tracing

JSON logs are always emitted to stdout. When built with the `tracing-otlp` feature and `OTEL_EXPORTER_OTLP_ENDPOINT` is set, traces are exported over OTLP. Use the local Jaeger override with:

```bash
docker compose -f compose.yaml -f compose.tracing.yaml up -d
```

Then set:

```bash
OTEL_EXPORTER_OTLP_ENDPOINT=http://jaeger:4317
```

Jaeger UI is available on `http://localhost:16686`.

## Environment Variables

Required:

| Variable | Purpose |
| --- | --- |
| `DATABASE_URL` | Postgres connection string. |
| `QDRANT_URL` | Qdrant base URL. |
| `LITELLM_URL` | LiteLLM base URL including `/v1`. |
| `LITELLM_KEY` | LiteLLM API key. |
| `LITELLM_SALT_KEY` | LiteLLM salt key. |
| `EMBED_MODEL_PATH` | Local ONNX embedder directory. |

Optional:

| Variable | Default | Purpose |
| --- | --- | --- |
| `API_KEYS` | `agent-os,agentic-os` | Semicolon-delimited `token,namespace` entries. |
| `DEFAULT_MODEL` | `qwen36-27b` | Canonical model sent to LiteLLM. |
| `DEFAULT_TASK` | `engineering` | Task label when no header is provided. |
| `SUMMARIZER_ENABLED` | `true` | Enables the background summarizer loop. |
| `SUMMARIZER_BASE_URL` | Compose: `http://summarizer:8080/v1`; otherwise `LITELLM_URL` | OpenAI-compatible summarizer endpoint. |
| `SUMMARIZER_MODEL` | Compose: `qwen2.5-3b-instruct-q4_k_m`; otherwise `DEFAULT_MODEL` | Model name sent to the summarizer endpoint. |
| `SUMMARIZER_KEY` | unset | Optional bearer token for the summarizer endpoint. |
| `SUMMARIZER_MAX_TOKENS` | `384` | Max summary output tokens requested by the orchestrator. |
| `SUMMARIZER_PORT` | `8089` | Host port for the Compose llama.cpp summarizer service. |
| `SUMMARIZER_MODEL_PATH` | `/models/qwen2.5-3b-instruct-q4_k_m.gguf` | Model path inside the llama.cpp summarizer container. |
| `SUMMARIZER_PREDICT_TOKENS` | `384` | llama.cpp generation cap for the summarizer service. |
| `SUMMARIZER_THREADS` | `16` | CPU threads used by the llama.cpp summarizer service. |
| `SUMMARIZER_THREADS_BATCH` | `16` | CPU batch threads used by the llama.cpp summarizer service. |
| `SUMMARIZER_CTX_SIZE` | `4096` | Context window for the llama.cpp summarizer service. |
| `SUMMARIZER_GPU_LAYERS` | `0` | GPU layers for the llama.cpp summarizer service. |
| `SUMMARIZER_PARALLEL` | `1` | Concurrent llama.cpp slots for the summarizer service. Keep low to protect foreground work. |
| `CONTEXT_CACHE_TTL_MS` | `300000` | Context cache TTL. |
| `CONTEXT_DECAY_RATE` | `0.006` | Hybrid retrieval age decay. |
| `EXECUTION_FEEDBACK_ENABLED` | `true` | Enables execution artifact capture and Failure History context. |
| `FAILURE_HISTORY_TOKEN_BUDGET` | `1000` | Token budget for Failure History context. |
| `BACKGROUND_WORK_CONCURRENCY` | `4` | Max concurrent best-effort derived background jobs such as cache refresh, feature extraction, and Qdrant indexing. |
| `FEATURE_EXTRACTION_ENABLED` | `true` | Enables inline feature extraction and Operational Constraints context. |
| `FEATURE_STARTUP_BACKFILL_ENABLED` | `true` | Runs idempotent feature bootstrap tagging and feature record rebuild as a startup gate before serving traffic when feature extraction is enabled. |
| `FEATURE_STARTUP_BACKFILL_BATCH_SIZE` | `500` | Batch size for startup bootstrap metadata updates. |
| `FEATURE_STARTUP_SKIP_BOOTSTRAP_TAGGING` | `false` | Skips startup bootstrap tagging and only rebuilds feature records. Use only after historical tagging is already complete. |
| `HARNESS_FEEDBACK_STARTUP_BACKFILL_ENABLED` | `true` | Runs best-effort startup harness feedback classification before serving traffic. |
| `HARNESS_FEEDBACK_STARTUP_BACKFILL_BATCH_SIZE` | `500` | Batch size for startup harness feedback metadata repair. |
| `HARNESS_FEEDBACK_BACKGROUND_REPAIR_ENABLED` | `true` | Runs conservative periodic harness feedback repair in the background. |
| `HARNESS_FEEDBACK_REPAIR_INTERVAL_SEC` | `300` | Interval for periodic harness feedback repair. |
| `HARNESS_FEEDBACK_REPAIR_LOOKBACK_SEC` | `2 * interval`, minimum `60` | Recent event window scanned by periodic harness feedback repair. |
| `HARNESS_FEEDBACK_REPAIR_BATCH_SIZE` | `500` | Batch size for periodic harness feedback repair. |
| `FEATURE_WINDOW_SEC` | `3600` | Session fallback grouping window when trajectory lineage is absent. |
| `CONSTRAINT_FRESHNESS_WINDOW_SEC` | `1800` | Maximum age of detections and recoveries used for active constraints. |
| `MAX_OPERATIONAL_CONSTRAINTS` | `5` | Maximum constraints emitted by the deterministic builder. |
| `OPERATIONAL_CONSTRAINTS_TOKEN_BUDGET` | `300` | Token budget for Operational Constraints context. |
| `SENTIMENT_MODEL_PATH` | unset | Optional local sentiment model directory. |
| `SENTIMENT_THRESHOLD` | `0.70` | Negative sentiment threshold. |
| `RATE_LIMIT_PER_MINUTE` | `60` | Per-key inference route refill rate. |
| `RATE_LIMIT_BURST` | `30` | Per-key inference route burst. |
| `ALLOWED_ORIGINS` | `*` | CORS origin policy. |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | unset | Enables OTLP trace export when built with `tracing-otlp`. |
