# Phase 5: Narrow Live Policy

## Goal

Promote only high-confidence deterministic route decisions into live behavior.

## Scope

Allow live enforcement for objective cases only:

- high-stakes medical/legal/finance requests
- unsafe security requests
- destructive commands
- current-info requests requiring external lookup
- missing target/context requiring clarification
- trivial deterministic templates

Do not promote ambiguous small-model versus strong-model routing in this phase.

## Live Policy v1

Allowed live actions:

```text
ask_clarification
refuse_or_guardrail
web_required
deterministic_template
```

Routes still shadow-only:

```text
small_local_model
strong_local_model
tool_required
```

Tool-required can be surfaced as a route recommendation, but execution remains
behind existing tool/harness controls.

## Enforcement Rules

Examples:

- `high_stakes` plus medical/legal/finance -> refuse or provide safe general
  information with disclaimer, depending on existing product policy.
- `external_current_info_required` -> do not answer from stale model memory;
  return `web_required` unless a web/tool path is available.
- `destructive_command` -> require confirmation or guardrail.
- missing file/cluster/repo target for modification/operation -> ask
  clarification.
- L0 deterministic template -> answer without invoking a strong model.

## Configuration

Live policy must be explicitly configurable:

```text
REQUEST_CLASSIFICATION_LIVE_POLICY_ENABLED=false by default
REQUEST_CLASSIFICATION_POLICY_VERSION=v1
```

Operators must be able to run shadow-only after deployment.

## Metrics

Add:

```text
request_live_policy_actions_total{action,reason}
request_live_policy_bypassed_total{reason}
```

No raw prompt or path labels.

## Required Tests

1. Live policy is disabled by default.
2. With live policy disabled, dispatch behavior is unchanged.
3. High-stakes requests route to guardrail/refusal when enabled.
4. Current-info requests route to `web_required` when enabled.
5. Destructive command requests do not execute directly.
6. Clarification route returns a bounded response contract.
7. Small/strong model routing remains shadow-only.
8. Policy action metrics use bounded labels.

## Stop Condition

Stop when narrow live policy is feature-flagged, disabled by default, and tested
for objective high-confidence cases only.
