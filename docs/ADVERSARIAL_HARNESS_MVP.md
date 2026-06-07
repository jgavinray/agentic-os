# Adversarial Harness MVP

This subsystem makes orchestration measurable. Its purpose is to prove whether governed request handling improves raw model usability, reduces unnecessary heavy-model usage, and preserves accepted output quality.

The goal is not local-model maximalism, model collection, novelty claims, or multi-agent theater. The goal is accepted engineering output per dollar, watt/GPU-second, token, and human-attention minute.

## Core Thesis

Large ambiguous prompts should not be sent directly to one model and expected to succeed. The orchestrator should classify the request, detect missing context and multiple intents, split bounded subtasks when needed, pack only relevant context, route each subtask to the cheapest sufficient model path, validate deterministically, record objections when risk warrants review, and escalate to frontier models only when local paths cannot confidently resolve the decision.

Every claim about value must be measured against comparable baselines.

## MVP Decisions

Implemented first slice:

- `POST /harness/outcomes` captures append-only outcome labels.
- `POST /harness/litellm-callbacks` stores raw LiteLLM callback payloads and normalizes known ledger fields.
- `x-agent-baseline-arm` propagates bounded baseline-arm labels into request metadata and `litellm_call_ledger`.


- Add a minimal outcome writer for accepted-output feedback.
- Run the raw vLLM baseline through LiteLLM with orchestration disabled so ledger rows remain comparable.
- Store raw LiteLLM callback payloads as JSONB and normalize known fields opportunistically.
- Keep adversarial review observational until at least 20 samples exist per task/risk class.
- Disable or reduce adversarial review per class only when valid-objection and decision-change rates fail to justify latency/cost.
- Use eval-operator human-attention buckets initially; add client hooks and posthoc annotation later.

## Existing Foundation

agentic-os already has most of the substrate needed for a measured control loop:

- `agent_events` is the immutable event log.
- Trajectory lineage records `trajectory_id`, `attempt_index`, and `event_role`.
- `trajectory_result` events aggregate retry count, model calls, validations, token totals, latency, final status, and success.
- `agent_request_classifications` stores derived/versioned request classifications.
- `agent_orchestration_policies` stores append-only policy envelopes.
- `context_compiler_ledger` records context inclusion/suppression decisions.
- `litellm_call_ledger` records model attempts, routed model, route policy, cache policy, context pack ID/hash, terminal status, latency, provider cache counters, and trajectory links.
- `/v1/validations` captures deterministic validation outcomes as immutable execution events.
- `raw_http_exchanges` stores raw prompt/response evidence in a separate capture database when configured, outside memory context.

The MVP should add only the missing measurement and lifecycle records around this foundation.

## System Boundaries

Open-source portable layer:

- orchestration code
- schemas and migrations
- routing/classifier interfaces
- policy engine
- eval runner
- generic prompt templates
- synthetic eval fixtures
- storage interfaces

Workspace-local layer:

- employer traces
- prompts and responses
- repo context and embeddings
- context packs
- labels derived from private work
- trained classifiers
- model behavior profiles from private work
- employer-specific policies and evals

Do not commit workspace-local prompts, responses, incidents, labels, embeddings, traces, or trained classifiers into the open-source core.

## Incremental Deliverables

### Deliverable 1: Outcome Writer

Add a minimal append-only outcome writer so accepted-output feedback can be measured.

Recommended table:

```sql
CREATE TABLE harness_outcome_events (
    outcome_event_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    trajectory_id UUID NOT NULL,
    source_event_ids UUID[] NOT NULL DEFAULT '{}',
    baseline_arm TEXT NOT NULL,
    accepted BOOLEAN NOT NULL,
    invalid_output BOOLEAN NOT NULL DEFAULT false,
    manual_prompt_edits_required INTEGER NOT NULL DEFAULT 0,
    human_attention_bucket TEXT NOT NULL CHECK (
        human_attention_bucket IN ('none', 'low', 'medium', 'high', 'unknown')
    ),
    labeler_type TEXT NOT NULL CHECK (
        labeler_type IN ('eval_operator', 'client_hook', 'human', 'posthoc')
    ),
    notes TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
```

Initial API:

`POST /harness/outcomes`

Required fields:

- `trajectory_id`
- `baseline_arm`
- `accepted`
- `invalid_output`
- `manual_prompt_edits_required`
- `human_attention_bucket`
- `source_event_ids`
- `notes`

Acceptance criteria:

- outcome rows are append-only
- invalid human-attention buckets are rejected
- outcomes can be joined to `trajectory_result` by `trajectory_id`
- no raw prompts or responses are stored in this table

### Deliverable 2: Raw LiteLLM Callback Capture

Store raw LiteLLM callback payloads as facts, then normalize known fields opportunistically.

Recommended table:

```sql
CREATE TABLE litellm_callback_payloads (
    callback_payload_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    attempt_id UUID,
    trajectory_id UUID,
    received_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    callback_type TEXT NOT NULL,
    payload JSONB NOT NULL
);
```

Optional nullable columns to add to `litellm_call_ledger` when callback data is available:

- `litellm_call_id`
- `model_group`
- `deployment_id`
- `api_key_alias`
- `response_cost`
- `key_spend`
- `fallback_depth`
- `raw_callback_payload_id`

Acceptance criteria:

- raw callback payloads are stored without blocking request completion
- known fields are normalized only when present
- missing provider-specific fields do not fail the write path
- payloads are linked to `attempt_id` whenever possible

### Deliverable 3: Baseline Arms

Add explicit baseline-arm attribution.

Initial arms:

- `raw_vllm_via_litellm`
- `orchestrator_policy_disabled`
- `orchestrator_policy_enabled`
- `orchestrator_policy_plus_observational_review`
- `frontier_escalation`

Implementation options:

- store `baseline_arm` on `harness_outcome_events`
- add `baseline_arm` to request metadata and propagate to `litellm_call_ledger`
- include `baseline_arm` in eval-run metadata

Acceptance criteria:

- the same eval case can be run through multiple arms
- each arm produces comparable LiteLLM ledger rows
- orchestration-disabled traffic still goes through LiteLLM for comparable latency/token/cost data

### Deliverable 4: Extended Classification Labels

Keep `agent_request_classifications` for current deterministic classification. Add generic versioned labels for richer harness taxonomies.

Recommended table:

```sql
CREATE TABLE agent_labels (
    label_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    trajectory_id UUID NOT NULL,
    source_event_ids UUID[] NOT NULL,
    label_type TEXT NOT NULL,
    label_value JSONB NOT NULL,
    labeler_type TEXT NOT NULL CHECK (
        labeler_type IN ('rule', 'local_model', 'frontier_model', 'human', 'validator', 'posthoc')
    ),
    taxonomy_version TEXT NOT NULL,
    confidence DOUBLE PRECISION CHECK (confidence IS NULL OR confidence BETWEEN 0 AND 1),
    supersedes_label_id UUID,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
```

Initial label types:

- `intent`
- `secondary_intents`
- `risk_class`
- `risk_subtypes`
- `blast_radius`
- `missing_context`
- `required_validators`
- `review_required`
- `frontier_escalation_allowed`
- `failure_label`

Acceptance criteria:

- labels are append-only and versioned
- a new classifier version can relabel old trajectories without rewriting prior labels
- every label links to source event IDs

### Deliverable 5: Subtask Decomposition Records

Record bounded subtasks when a request has multiple intents.

Recommended table:

```sql
CREATE TABLE orchestration_subtasks (
    subtask_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    trajectory_id UUID NOT NULL,
    parent_event_id UUID,
    intent TEXT NOT NULL,
    model_role TEXT NOT NULL,
    risk_class TEXT NOT NULL,
    required_context JSONB NOT NULL DEFAULT '[]'::jsonb,
    validator_set JSONB NOT NULL DEFAULT '[]'::jsonb,
    acceptance_criteria JSONB NOT NULL DEFAULT '[]'::jsonb,
    status TEXT NOT NULL CHECK (
        status IN ('planned', 'running', 'completed', 'blocked', 'failed', 'cancelled')
    ),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
```

Acceptance criteria:

- composite requests produce multiple scoped subtasks
- each subtask has acceptance criteria and validator requirements
- Slack/docs-style subtasks can be routed lower risk than safety/deployment subtasks
- a vague multi-intent prompt is not sent as one undifferentiated worker prompt

### Deliverable 6: Observational Review And Objection Lifecycle

Adversarial review should be observational until enough data exists.

Recommended lifecycle:

- `raised`
- `classified`
- `accepted_as_valid`
- `rejected_as_invalid`
- `partially_addressed`
- `addressed`
- `unresolved`
- `escalated`

Recommended tables:

```sql
CREATE TABLE review_events (
    review_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    trajectory_id UUID NOT NULL,
    subtask_id UUID,
    reviewer_role TEXT NOT NULL,
    source_event_ids UUID[] NOT NULL,
    decision TEXT NOT NULL CHECK (decision IN ('proceed', 'revise', 'stop')),
    highest_risk_assumption TEXT,
    failure_mode TEXT,
    objection_category TEXT,
    severity TEXT,
    testability TEXT,
    required_fix TEXT,
    validation_required JSONB NOT NULL DEFAULT '[]'::jsonb,
    observational BOOLEAN NOT NULL DEFAULT true,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE objection_events (
    objection_event_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    objection_id UUID NOT NULL,
    trajectory_id UUID NOT NULL,
    review_id UUID,
    status TEXT NOT NULL,
    category TEXT,
    severity TEXT,
    specificity TEXT,
    testability TEXT,
    concrete BOOLEAN,
    requires_escalation BOOLEAN NOT NULL DEFAULT false,
    source_event_ids UUID[] NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
```

Acceptance criteria:

- review can run without blocking output
- objections are first-class lifecycle records
- review policy remains observational until at least 20 samples per task/risk class
- valid-objection and decision-change rates are measurable

### Deliverable 7: Frontier Escalation Packets

Frontier calls should be compact arbitration packets, not full-context retries.

Packet fields:

- task summary
- constraints
- worker proposal
- adversarial objection
- worker rebuttal, if any
- validator results
- unresolved decision

Decision output:

- `proceed`
- `revise`
- `stop`
- unresolved risk
- evidence required to proceed
- local failure cause: `context`, `capability`, `prompt`, `validation_gap`, `unknown`

Acceptance criteria:

- escalation has a reason code
- frontier model receives a compact packet
- frontier decision-change rate is measurable
- frontier calls are attributable to accepted or rejected value

### Deliverable 8: Eval Runner

Create a small synthetic eval pack that tests harness leverage, not model trivia.

Suggested file:

`eval/packs/harness_leverage_v1.yaml`

Initial cases:

1. Crashlooping deployment asks for next commands.
2. Rollout plan depends on previously decided approach.
3. Kubernetes change asks for safety review, rollout commands, validation checklist, and Slack update.
4. Production deployment fix asks for a command.
5. Kafka replay/Postgres writeback plan asks to review and proceed.

CLI shape:

```bash
orchestrator-maint run-eval-pack \
  --pack harness_leverage_v1 \
  --arm raw_vllm_via_litellm

orchestrator-maint run-eval-pack \
  --pack harness_leverage_v1 \
  --arm orchestrator_policy_enabled
```

Each eval result should capture:

- eval pack
- eval case ID
- baseline arm
- trajectory ID
- accepted
- invalid output
- retry count
- time to accepted response
- tokens to accepted response
- manual prompt edits required
- human attention bucket

Acceptance criteria:

- raw baseline and orchestrated paths produce comparable ledger rows
- eval outcomes can be labeled by an eval operator
- synthetic fixtures contain no workspace-private data

### Deliverable 9: MVP Metrics Report

Add a report command before adding more Prometheus metrics.

Suggested command:

```bash
orchestrator-maint harness-report \
  --eval-pack harness_leverage_v1 \
  --since 2026-01-01T00:00:00Z
```

Metrics:

- `first_try_acceptance_rate`
- `eventual_acceptance_rate`
- `average_retries_to_acceptance`
- `invalid_output_rate`
- `manual_prompt_edits_required`
- `time_to_accepted_response`
- `tokens_to_accepted_response`
- `local_tokens_per_accepted_result`
- `frontier_tokens_per_accepted_result`
- `cost_per_accepted_result`
- `human_attention_bucket_distribution`
- `policy_model_delta`
- `valid_objection_rate`
- `false_objection_rate`
- `decision_change_rate`
- `frontier_value_rate`

Minimum formulas:

- `first_try_acceptance_rate = accepted outcomes with retry_count = 0 / all labeled completed outcomes`
- `eventual_acceptance_rate = accepted outcomes / all labeled completed outcomes`
- `average_retries_to_acceptance = avg(retry_count) over accepted outcomes`
- `invalid_output_rate = invalid_output outcomes / all labeled completed outcomes`
- `policy_model_delta = orchestrator_policy_enabled metric - orchestrator_policy_disabled metric`

Acceptance criteria:

- report compares baseline arms over the same eval pack
- report joins outcome rows to trajectory summaries and LiteLLM call ledger rows
- report does not require raw prompt/response access

## Event Flow

1. Request is captured as `agent_events(event_role='request')`.
2. Initial classification writes current request classification plus richer `agent_labels`.
3. Composite requests create `orchestration_subtasks`.
4. Each subtask gets a bounded context pack.
5. Routing decision is recorded with baseline arm, model role, route reason, and policy version.
6. Worker generation writes LiteLLM ledger rows and model response events.
7. Deterministic validators write validation events.
8. Observational review writes review and objection lifecycle events when risk warrants.
9. Frontier escalation writes decision events only when configured escalation criteria are met.
10. Final outcome writer records accepted/invalid/manual-attention labels.
11. Trajectory result remains the aggregate lifecycle summary.
12. Eval/report commands compute deltas from immutable rows.

## Kill Criteria

Disable, reduce, or keep observational any subsystem that fails to show value:

- If adversarial review adds significant latency/cost but rarely changes valid outcomes, reduce it for that class.
- If valid-objection rate is low or false-objection rate is high for a task/risk class, keep review observational.
- If a classifier performs worse than deterministic rules, keep it shadow-only.
- If local routing increases retries or human attention for a class, route that class elsewhere.
- If frontier escalation rarely changes decisions for a category, stop escalating that category.
- If context packing increases token use without improving acceptance, tighten context policy.

## Suggested Implementation Order

1. Add `harness_outcome_events` migration and writer API.
2. Add `litellm_callback_payloads` migration and best-effort persistence hook.
3. Add baseline-arm propagation through request metadata and LiteLLM ledger.
4. Add `agent_labels` for richer harness classification without replacing current classifications.
5. Add subtask decomposition records and pure tests for multi-intent prompts.
6. Add observational review and objection lifecycle records.
7. Add synthetic eval pack and `run-eval-pack` maintenance command.
8. Add `harness-report` command with baseline deltas.
9. Add docs and metrics table updates.
10. Only after enough samples exist, consider learned routing or classifier training.

## Test And Validation Plan

Unit tests:

- valid and invalid human-attention buckets
- append-only outcome writes
- baseline-arm parsing and bounded labels
- classification label supersession
- decomposition of the Kubernetes multi-intent prompt
- objection lifecycle transitions
- metric formulas over small synthetic rows

Integration checks:

```bash
cargo fmt
cargo test
cargo test request_classification
cargo test trajectory
cargo test execution_feedback
```

Manual MVP check:

1. Run `harness_leverage_v1` through `raw_vllm_via_litellm`.
2. Run the same pack through `orchestrator_policy_enabled`.
3. Label outcomes with eval-operator buckets.
4. Run `harness-report`.
5. Confirm the report can answer whether orchestration improved first-try acceptance, retries, invalid output rate, and human attention without needing private raw prompts in the open-source repo.
