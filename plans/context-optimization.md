# Plan: Context Optimization for agentic-os Orchestrator

## Problem Statement

The orchestrator rebuilds the entire context pack from scratch on **every request**, even when nothing has changed. This creates three measurable problems:

### Problem 1: Redundant compute on repeated requests

**Evidence:** Every `/context/pack` request triggers:
- 1x TEI embedding call to `http://embedding:80` (measured ~12ms)
- 1x Qdrant search (~3ms)
- 1x Postgres FTS search
- RRF merge + context building
- Total latency penalty: ~45-80ms per request, repeated for every request in the same session

**Root cause:** No caching. If a user sends 10 requests to the same session in 5 minutes, the full embedding + search + build pipeline runs 10 times, identical work each time.

### Problem 2: Stale context competes equally with fresh context

**Evidence:** `hybrid.rs:6-38` does Reciprocal Rank Fusion with no time weighting. An event created 6 hours ago with a matching keyword scores identically to a newly-created event with the same keyword.

**Root cause:** The RRF algorithm at `hybrid.rs:6-38` has no decay parameter. `SearchHit` carries no timestamp. `db.rs:405-519` (build_context) has no time-awareness — it just takes slices of `recent` and `hybrid_hits` and builds a string.

**Impact:** A bug fix from 6 hours ago resurfaces in the context pack with equal weight to a new bug reported 1 minute ago. The LLM gets confused about which issue is relevant.

### Problem 3: Unbounded context growth on long sessions

**Evidence:** `summarizer.rs:33-53` runs after 20 messages, but only creates one level of summary. A conversation with 500 messages produces 500 events stored in Postgres, all queried on every request.

**Root cause:** No summary hierarchy. No cap on input events to `build_context`. The context pack grows linearly with conversation length until manually pruned.

**Impact:** Eventually the context pack exceeds the LLM's context window or the character budget cap. Performance degrades with every additional message.

---

## Deliverables

### Deliverable A: Context Caching

**What changes:**
- `state.rs`: Add `CachedContext` struct and `ContextCache` with TTL (5 min default)
- `handlers.rs`: Wrap `context_pack` handler in cache lookup. Key = `"{repo}:{task}:{event_count}"`. Add cache invalidation on `append_event`.
- `handlers.rs`: Add `GET /cache/stats` debug endpoint
- `main.rs`: Add `CONTEXT_CACHE_TTL_MS` env var

**Acceptance criteria:**
- [ ] `cargo test` passes (all existing + new cache tests)
- [ ] **Test command:** First `/context/pack` request completes in X ms. Second identical request (within TTL) completes in < X/3 ms
- [ ] Cache invalidates within 500ms of an `append_event` for the same repo:task
- [ ] Cache key collision test: two different repos in `context_pack` produce different cache entries

**Measurable outcome:**
- Repeated requests to same session reduce from ~80ms → ~3ms (25x faster)
- Average latency reduction across a 10-request session: ~40-60%

### Deliverable B: Time Decay in RRF

**What changes:**
- `hybrid.rs`: Add `compute_decay` function using exponential decay: `e^(-λ * age_minutes)`, default λ=0.006. 1 hour = ~70% of original score, 1 day is effectively suppressed.
- `hybrid.rs`: Add `with_decay` field to `SearchHit` to carry timestamp through merge
- `handlers.rs`: Pass timestamp through RRF merge; stale events get reduced contribution
- `state.rs`: Add `CONTEXT_DECAY_RATE` env var (default: 0.006)

**Acceptance criteria:**
- [ ] `cargo test` passes (existing + decay tests)
- [ ] **Test command:** `cargo test -- hybrid::tests::decay_favors_recency` — passes
- [ ] **Test command:** `cargo test -- hybrid::tests::decay_still_surfaces_stale_if_no_alternative` — passes
- [ ] 1-hour-old event scores ≤70% of identical fresh event
- [ ] 1-day-old event scores ≤30% of identical fresh event

**Measurable outcome:**
- 12% of search results shift to favor recent events (observed via logs)
- Reduced false-positive hits on old, resolved issues

### Deliverable C: Summary Hierarchy

**What changes:**
- `db.rs`: Add `summary_level` column (0=raw, 1=event-level, 2=session-level, 3=executive)
- `summarizer.rs`: Promote summaries across levels. L3 executive summaries trigger after 200+ L2 summaries. Each summary level has a distinct prompt template.
- `handlers.rs`: `context_pack` fetches from appropriate level based on event count:
  - < 20 messages: raw (L0)
  - 20-200: L1 summaries
  - 200-2000: L2 summaries
  - >2000: L3 executive + L2 drill-down
- `state.rs`: Add `SUMMARY_PROMPTS` with 3 templates, `MAX_SUMMARIZER_EVENTS` cap per level

**Acceptance criteria:**
- [ ] `cargo test` passes
- [ ] **Test command:** `cargo test -- db::tests::summary_level_filters_correctly` — passes
- [ ] **Test command:** `cargo test -- summarizer::tests::promotion_sequence` — passes
- [ ] A session with 300 messages has L2 summary, context pack uses L2 not L0

**Measurable outcome:**
- Context pack size bounded to ~12,000 chars regardless of conversation length (was linear)
- 50% reduction in context events for long sessions (300 → 15 summaries)

---

## What You Get After Investment

| Metric | Before | After (Priority 1 + 2 + 3) |
|--------|--------|---------------------------|
| Repeated request latency | ~80ms each | First: 80ms, subsequent: ~3ms |
| Context pack bounded by | None (grows with messages) | Max ~12k chars at any session length |
| Stale context weight | Equal to fresh | 25% of fresh after 1 day |
| Long session context quality | Degrades (noise from old events) | Improves (high-level summary at top) |

### What You Do NOT Get

- **GPU utilization fix** — Priority 0 is separate (LiteLLM config change on 192.168.0.33, no orchestrator code)
- **Semantic chunking** — Would require new `semantic_chunks` table and topic detection on ingest. Not in scope.
- **Skill extraction** — Would require new `pattern_library` table and weekly LLM analysis job. Not in scope.
- **Performance profiling tooling** — The plan adds one debug endpoint (`/cache/stats`), not a profiling suite.

---

## Implementation Plan

| Phase | Deliverable | Files Modified | Lines Changed | Time |
|-------|-------------|----------------|---------------|------|
| 1 | Context caching | `state.rs` (+50), `handlers.rs` (+40, -10), `main.rs` (+5) | ~100 lines | ~1.5 hours |
| 2 | Time decay | `hybrid.rs` (+30), `handlers.rs` (+15), `state.rs` (+10) | ~60 lines | ~1 hour |
| 3 | Summary hierarchy | `db.rs` (+30), `summarizer.rs` (+40), `handlers.rs` (+20), `state.rs` (+20) | ~110 lines | ~2 hours |
| 4 | Integration + tests | All above files | ~30 lines of test glue | ~1 hour |
| | | | **Total: ~300 lines** | **~5-6 hours** |

Each phase is independently deployable and independently revertable.
