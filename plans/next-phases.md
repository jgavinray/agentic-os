# Next Phases: Layered Memory Policy, Evidence Promotion, Metrics

## Context

The context optimization phase added caching, time-decayed retrieval, and basic summary levels. The next phase should make those levels semantically correct.

The core rule:

Higher memory levels should not replace lower levels. They should compress, index, and route back to the right lower-level evidence.

The MVP loop remains:

1. Capture useful coding episodes
2. Retrieve the right prior episodes at the right time
3. Inject them into a structured context pack
4. Help the model avoid repeated mistakes
5. Validate code changes with tools/tests
6. Persist what worked and what failed
7. Improve the next attempt

---

## Memory Levels

| Level | Meaning | Question Answered |
|-------|---------|-------------------|
| L0 | Raw episodic events | What happened exactly? |
| L1 | Session/task summaries | What happened in this task/session? |
| L2 | Repo-level patterns, conventions, decisions | What reusable repo knowledge emerged? |
| L3 | Durable project memory and architecture truths | What durable architectural/project truth changed? |

Higher levels must retain evidence links to lower levels through source IDs.

---

## Phase 4: Layered Context Policy

### Problem Statement

Current context selection is too coarse: it chooses memory by event count and summary level. That reduces size, but it can drop useful raw evidence or over-promote weak summaries.

The system needs task-aware context assembly:

- Small/debug task: recent L0 tail + matching L1 summaries + failed attempts/remediations
- Broad/refactor task: L1 summaries + L2 repo memory + small L0 tail
- Architecture task: L3/L2 first + selective L1/L0 evidence

### Proposed Architecture

Add a context policy layer that selects memory bands by task category.

```text
ContextPolicy
  task_category: Narrow | Moderate | Broad | Architecture
  allocations:
    l0_recent_budget
    l1_matching_budget
    l2_repo_budget
    l3_project_budget
    failure_budget
```

Suggested starting allocations:

| Task Type | L0 Recent | L1 Matching | L2 Repo | L3 Project | Failures/Remediations |
|-----------|-----------|-------------|---------|------------|------------------------|
| Small/debug | 40% | 30% | 10% | 0% | 20% |
| Moderate | 25% | 35% | 20% | 5% | 15% |
| Broad/refactor | 10% | 35% | 35% | 10% | 10% |
| Architecture | 5% | 20% | 35% | 30% | 10% |

### Implementation Plan

**Phase 4.1: Data model**
- `state.rs`: Add `MemoryLevel`, `MemoryScope`, `PromotionReason`, and `ContextPolicy`
- `db.rs`: Add fields or metadata conventions for:
  - `source_event_ids`
  - `source_summary_ids`
  - `memory_level`
  - `scope`
  - `promotion_reason`
  - `confidence`
  - `evidence`
  - `outcome`
  - `last_referenced_at`

**Phase 4.2: Retrieval by level**
- `db.rs`: Add query helpers for L0, L1, L2, L3, failures, and remediations
- `context_packing/` and `handlers/context.rs`: Replace single-level context retrieval with `ContextPolicy`
- `db.rs`: Build context sections in a stable order:
  1. Durable Project Memory
  2. Repo Patterns and Decisions
  3. Relevant Session Summaries
  4. Recent Evidence
  5. Failed Attempts and Remediations
  6. Open Questions

**Phase 4.3: Context pack structure**
- Keep levels additive, not mutually exclusive
- Always include a small L0 tail when available
- For L2/L3 items, include source IDs or compact evidence references
- Mark memory level in each section item

### Acceptance Criteria

- [x] `cargo test` passes
- [x] `cargo test -- db::tests::context_policy_small_task_includes_l0_and_l1`
- [x] `cargo test -- db::tests::context_policy_broad_task_includes_l1_l2_and_l0_tail`
- [x] `cargo test -- db::tests::context_policy_architecture_task_prioritizes_l2_l3`
- [x] `cargo test -- db::tests::higher_levels_do_not_eliminate_l0_tail`
- [x] `cargo test -- handlers::tests::context_pack_sections_are_stable`

### Deliverables

| Files Modified | Complexity |
|----------------|------------|
| `state.rs` | Medium |
| `db.rs` | Medium-High |
| `handlers/context.rs` | Medium |
| tests | Medium |

---

## Phase 5: Evidence-Backed Promotion

### Problem Statement

Summaries and reusable memory can become unreliable if they are promoted too aggressively. The system needs conservative promotion rules and evidence links.

Promotion rules:

- L0 -> L1: automatic by session/event count
- L1 -> L2: only if repeated, referenced, marked important, or tied to a failed/remediated workflow
- L2 -> L3: only if stable across time or manually confirmed

### Failed Attempts And Remediations

Failed attempts and remediations are high-value memory. They must either be copied upward losslessly or summarized with explicit evidence and outcome.

Required fields:

```text
problem_signature
attempted_approach
observed_outcome
why_it_failed
successful_remediation
evidence
source_event_ids
```

### Implementation Plan

**Phase 5.1: Event types**
- Add first-class handling for:
  - `failed_attempt`
  - `remediation`
  - `decision`
  - `convention`
  - `architecture_truth`

**Phase 5.2: Promotion gates**
- `summarizer.rs`: Keep L0 -> L1 automatic
- `summarizer.rs`: Add L1 -> L2 promotion only when:
  - same pattern appears >= 2 times
  - memory is referenced by later context retrieval
  - event metadata has `important: true`
  - event is a failed attempt with remediation
- `summarizer.rs`: Add L2 -> L3 promotion only when:
  - stable over multiple sessions
  - manually confirmed
  - architecture-related and repeated

**Phase 5.3: Evidence preservation**
- Store source IDs with every promoted memory
- Reject or skip L2/L3 promotion when source IDs are missing
- Include compact evidence references in context packs

### Acceptance Criteria

- [x] `cargo test` passes
- [x] `cargo test -- summarizer::tests::l1_to_l2_requires_promotion_signal`
- [x] `cargo test -- summarizer::tests::l2_to_l3_requires_stability_or_confirmation`
- [x] `cargo test -- db::tests::promoted_memory_requires_source_ids`
- [x] `cargo test -- db::tests::failed_attempt_preserves_evidence_and_outcome`
- [x] `cargo test -- handlers::tests::failed_attempts_are_injected_for_debug_tasks`

### Deliverables

| Files Modified | Complexity |
|----------------|------------|
| `db.rs` | Medium |
| `state.rs` | Medium |
| `summarizer.rs` | Medium-High |
| `handlers/context.rs` | Medium |
| tests | Medium |

---

## Phase 6: Metrics And Harness Tightening

### Problem Statement

The system needs to prove that memory improves coding workflows instead of merely shrinking context. Metrics and harness tests should make regressions visible.

### Metrics

Add structured metrics for:

- `context_pack_build_ms`
- `context_cache_hit_rate`
- `context_pack_chars`
- `context_pack_tokens_estimate`
- `processed_tokens`
- `cached_tokens`
- `generated_tokens`
- `l0_items_injected`
- `l1_items_injected`
- `l2_items_injected`
- `l3_items_injected`
- `retrieval_semantic_hits`
- `retrieval_fts_hits`
- `retrieval_deduped_hits`
- `failed_attempts_injected`
- `remediations_injected`
- `promotion_attempts`
- `promotion_accepted`
- `promotion_rejected`
- `memory_source_coverage`
- `stale_cache_invalidations`

### Harness

Add golden fixtures for realistic coding episodes:

- Small bug fix with one prior failed attempt
- Broad refactor with repo conventions
- Architecture question requiring durable project memory
- Repeated error that promotes from L1 to L2
- Manually confirmed architecture truth that promotes from L2 to L3

The harness should assert:

- Context pack section order is stable
- L2/L3 do not fully replace L0/L1
- Failed attempts include evidence and outcome
- No promoted L2/L3 memory exists without source IDs
- Cache invalidates after every write path
- Debug tasks receive failure/remediation memory
- Architecture tasks prioritize L2/L3 but still include evidence

### Acceptance Criteria

- [x] `cargo test` passes
- [x] `cargo test -- handlers::tests::metrics_include_context_pack_counts`
- [x] `cargo test -- handlers::tests::cache_hit_rate_updates`
- [ ] `cargo test -- harness::tests::golden_debug_context_pack`
- [ ] `cargo test -- harness::tests::golden_architecture_context_pack`
- [ ] `cargo test -- harness::tests::promoted_memory_has_source_coverage`
- [ ] `./litmus.sh` passes with services running
- [ ] `./mvp_test.sh` passes with services running

### Deliverables

| Files Modified | Complexity |
|----------------|------------|
| `state.rs` | Medium |
| `handlers/context.rs` | Medium |
| `db.rs` | Medium |
| `summarizer.rs` | Medium |
| test fixtures | Medium-High |

---

## Implementation Order

1. Layered context policy
2. Evidence-backed promotion
3. Metrics
4. Golden harness fixtures
5. Full MVP integration test

This order keeps retrieval behavior visible before promotion gets smarter, then adds metrics and harness checks to prevent memory quality regressions.
