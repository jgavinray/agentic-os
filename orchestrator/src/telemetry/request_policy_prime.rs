use metrics::counter;

pub fn prime_request_policy_metrics() {
    for result in ["success", "failure"] {
        counter!("request_classification_backfill_runs_total", "result" => result).increment(0);
    }
    for result in ["inserted", "updated", "skipped", "dry_run", "error"] {
        counter!("request_classifications_written_total", "result" => result).increment(0);
    }
    for field in [
        "intent",
        "domain",
        "artifact_type",
        "complexity",
        "recommended_route",
        "response_contract",
    ] {
        counter!("request_classification_unknown_labels_total", "field" => field).increment(0);
    }

    let inventory = crate::request_classification::enum_inventory();
    let routes = inventory
        .iter()
        .find(|(name, _)| *name == "RecommendedRoute")
        .map(|(_, values)| *values)
        .unwrap_or(&[]);
    let risks = inventory
        .iter()
        .find(|(name, _)| *name == "RequestRisk")
        .map(|(_, values)| *values)
        .unwrap_or(&[]);
    let complexities = inventory
        .iter()
        .find(|(name, _)| *name == "RequestComplexity")
        .map(|(_, values)| *values)
        .unwrap_or(&[]);

    counter!(
        "request_classifications_total",
        "intent" => "unknown",
        "domain" => "unknown",
        "route" => "unknown"
    )
    .increment(0);
    for route in routes {
        counter!("request_route_recommendations_total", "route" => *route).increment(0);
    }
    for risk in risks {
        counter!("request_risk_flags_total", "risk" => *risk).increment(0);
    }
    for complexity in complexities {
        counter!("request_complexity_total", "complexity" => *complexity).increment(0);
    }

    for action in [
        "ask_clarification",
        "refuse_or_guardrail",
        "web_required",
        "deterministic_template",
    ] {
        for reason in [
            "unsafe_security",
            "objective_risk",
            "external_current_info_required",
            "missing_target_context",
            "l0_trivial",
        ] {
            counter!(
                "request_live_policy_actions_total",
                "action" => action,
                "reason" => reason
            )
            .increment(0);
        }
    }
    for reason in ["disabled", "shadow_only", "unsupported_policy_version"] {
        counter!("request_live_policy_bypassed_total", "reason" => reason).increment(0);
    }

    for decision in ["allow", "deny", "shape", "pass", "unknown"] {
        for reason in [
            "prefer_canonical_tool",
            "no_tools",
            "not_applicable",
            "disabled",
            "unknown",
        ] {
            counter!(
                "tool_mediation_decisions_total",
                "decision" => decision,
                "reason" => reason
            )
            .increment(0);
        }
    }
    for action in ["offered", "allowed", "hidden", "denied", "unknown"] {
        for capability in [
            "file_read",
            "text_search",
            "file_list",
            "file_edit",
            "validation",
            "publishing",
            "shell",
            "unknown",
        ] {
            counter!(
                "tool_mediation_tools_total",
                "action" => action,
                "capability" => capability
            )
            .increment(0);
        }
    }
}
