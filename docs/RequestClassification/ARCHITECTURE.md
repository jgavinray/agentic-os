# Request Classification Architecture

## Goal

Build a deterministic pre-LLM classification layer that turns raw request events
into bounded labels, features, and route recommendations before model dispatch.

Target flow:

```text
raw request
  -> deterministic parser
  -> feature extraction
  -> intent classifier
  -> domain classifier
  -> risk classifier
  -> complexity classifier
  -> routing policy
  -> deterministic answer / template / tool / small model / strong model
  -> validation
  -> outcome recording
  -> rule/classifier improvement loop
```

## Existing System Fit

`agent_events` remains the source of truth and audit log.

`agent_feature_records` remains operational trajectory-derived state.

`harness_feedback` remains benchmark/harness failure classification and
quarantine.

`context_ledger` remains context-pack decision training substrate.

Request classification adds derived request-level state. It must not replace or
duplicate these systems.

## Core Invariants

- Deterministic: identical input event and policy version produce identical
  features, labels, and route recommendation.
- Rebuildable: all rows can be regenerated from `agent_events`.
- Bounded labels: all labels are closed enums with explicit `unknown`.
- Features before labels: features are facts, labels are interpretations, routes
  are policy decisions.
- Privacy-bounded: no raw request text, raw tool output, auth value, full URL, or
  free-form prompt body is stored in the classification table.
- No metric label cardinality leaks: metrics use bounded labels only.
- No LLM calls in classification.
- No network calls in classification.
- Boot-safe: normal operation must not require an operator to run a secondary
  binary.

## Taxonomy

Intent:

```text
explain
debug
generate_config
modify_config
summarize
classify
search
plan
operate_tool
unknown
```

Domain:

```text
shell
kubernetes
llm_inference
docker
networking
security
medical
legal
finance
generic
unknown
```

Artifact type:

```text
plain_text
code
logs
yaml
json
sql
markdown
image
file
unknown
```

Complexity:

```text
L0_trivial
L1_simple
L2_moderate
L3_complex
L4_tool_required
L5_high_risk
unknown
```

Risk is multi-label:

```text
none
secret_present
destructive_command
external_current_info_required
high_stakes
prompt_injection
unsafe_security
unknown
```

Route:

```text
deterministic_template
small_local_model
strong_local_model
web_required
tool_required
ask_clarification
refuse_or_guardrail
unknown
```

Response contract:

```text
direct_answer
structured_json
markdown_summary
patch_required
validation_required
clarification_question
refusal
unknown
```

## Versioning

Every classification row records:

```text
classification_schema_version
routing_policy_version
classifier_source
```

Changing any enum, feature key, bucket boundary, or route rule requires a
schema or policy version bump. Historical rows are not silently rewritten unless
a backfill explicitly targets the new version.

## Storage Model

The classification table stores structured derived state only. Raw text remains
in `agent_events`.

Recommended table:

```text
agent_request_classifications
```

One row per classified request event per schema/policy version.

## Routing Posture

Initial routing is shadow-only. The system records what it would have done but
does not change model dispatch.

Live enforcement can be promoted only for high-confidence deterministic cases:

- refuse or guardrail for high-stakes/unsafe requests
- web required for current-info requests
- ask clarification for missing target/context
- deterministic template for trivial known-safe requests

Strong model versus small model routing comes later, after shadow data proves the
labels are stable.

## Orchestration Policy Handoff

Request classification does not directly decide which tools a model sees or
which client tool calls are allowed. It produces bounded labels. The
orchestration policy layer consumes those labels and derives the operational
envelope for the request.

The handoff is:

```text
RequestClassification
  -> derive_orchestration_policy
  -> context source eligibility
  -> allowed/required/blocked tool capabilities
  -> edit, validation, git, runtime, prompt-refinement, and risk policies
  -> event metadata and agent_orchestration_policies ledger row
```

This separation keeps the classifier rebuildable and privacy-bounded. The
policy layer can change tool/runtime behavior without adding unbounded text to
`agent_request_classifications`, and policy rows can be appended for later
analysis without rewriting historical classification rows.

See [../ORCHESTRATION_POLICY.md](../ORCHESTRATION_POLICY.md).
