# Operations

## Single-Node Contract

Exactly one orchestrator process may own a given Postgres database. Startup acquires a Postgres advisory lock before running migrations or serving traffic. A second orchestrator pointed at the same database exits with a clear error instead of running background work.

Postgres and Qdrant can run as ordinary single-node services. The in-process context cache, summarizer loop, and startup migration runner are intentionally not distributed.

## Summarizer

The summarizer ticks every 60 seconds. It scans for sessions with enough unsummarized source events, locks each session/target level with a Postgres advisory lock, writes summary events, marks source events summarized, and invalidates context cache entries for the repo.

## Context Cache

The cache is in process only. Keys are `repo:task:event_count`; limit overrides become part of the task string. `CONTEXT_CACHE_TTL_MS` controls expiration. Writes to memory invalidate all cache entries for the repo.

## Migrations

Schema migrations are embedded with refinery from `orchestrator/migrations/`. The first migration is the baseline schema. On legacy databases created by the old bootstrap DDL, the orchestrator detects the existing tables and marks the baseline as applied without rerunning it.

To add a migration:

1. Create `orchestrator/migrations/V<N>__short_name.sql`.
2. Use forward-only SQL.
3. Do not edit previously applied migrations.
4. Run `cargo test`, `cargo clippy -- -D warnings`, and a fresh compose boot.

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
| `DEFAULT_MODEL` | `qwen36-35b-heretic` | Canonical model sent to LiteLLM. |
| `DEFAULT_TASK` | `engineering` | Task label when no header is provided. |
| `CONTEXT_CACHE_TTL_MS` | `300000` | Context cache TTL. |
| `CONTEXT_DECAY_RATE` | `0.006` | Hybrid retrieval age decay. |
| `SENTIMENT_MODEL_PATH` | unset | Optional local sentiment model directory. |
| `SENTIMENT_THRESHOLD` | `0.70` | Negative sentiment threshold. |
| `RATE_LIMIT_PER_MINUTE` | `60` | Per-key inference route refill rate. |
| `RATE_LIMIT_BURST` | `30` | Per-key inference route burst. |
| `ALLOWED_ORIGINS` | `*` | CORS origin policy. |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | unset | Enables OTLP trace export when built with `tracing-otlp`. |
