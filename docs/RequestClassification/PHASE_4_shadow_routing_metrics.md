# Phase 4: Shadow Routing And Metrics

## Goal

Record route recommendations and outcome correlations without changing live
dispatch behavior.

## Scope

Implement shadow routing observability only:

- Store recommended route in `agent_request_classifications`.
- Emit route recommendation metrics.
- Join recommendations to observed outcomes when trajectory data exists.
- Add reporting/query helpers for operators.

Do not change which model/tool/template handles the request in this phase.

## Shadow Route Metrics

Add:

```text
request_classifications_total{intent,domain,route}
request_route_recommendations_total{route}
request_risk_flags_total{risk}
request_complexity_total{complexity}
```

All labels must be bounded.

## Outcome Correlation

For classified events with a trajectory, derive coarse outcome signals:

```text
route
final_status
validation_passed
validation_failed
retry_count_bucket
latency_bucket
input_token_bucket
output_token_bucket
```

This can be a view, query helper, or derived reporting function. Do not mutate
the original classification row with inferred outcomes unless a dedicated
versioned outcome table is added.

## Reporting

Add an operator-facing report, CLI or HTTP:

```text
classifications by route
top risk flags
unknown label counts
route recommendation vs outcome summary
sessions with repeated guardrail/refusal recommendations
```

Recommended CLI:

```text
orchestrator-maint request-classification-report
  [--repo REPO]
  [--since TIMESTAMP]
```

## Required Tests

1. Metrics are primed for every bounded route/risk/complexity label.
2. Classification writes increment bounded metrics.
3. Shadow route recommendation does not change request dispatch.
4. Reporting excludes raw request text.
5. Outcome correlation is deterministic for fixed event data.

## Stop Condition

Stop when shadow metrics and reporting work and tests prove dispatch behavior is
unchanged.
