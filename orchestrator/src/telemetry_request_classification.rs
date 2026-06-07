use metrics::counter;

pub fn record_request_classification_backfill_run(result: &'static str) {
    let result = match result {
        "success" => "success",
        "failure" => "failure",
        _ => "failure",
    };
    counter!("request_classification_backfill_runs_total", "result" => result).increment(1);
}

pub fn record_request_classification_write(result: &str) {
    let result = match result {
        "inserted" => "inserted",
        "updated" => "updated",
        "skipped" => "skipped",
        "dry_run" => "dry_run",
        "error" => "error",
        _ => "error",
    };
    counter!("request_classifications_written_total", "result" => result).increment(1);
}

pub fn record_request_classification_unknown_label(field: &str) {
    let field = match field {
        "intent" => "intent",
        "domain" => "domain",
        "artifact_type" => "artifact_type",
        "complexity" => "complexity",
        "recommended_route" => "recommended_route",
        "response_contract" => "response_contract",
        _ => "response_contract",
    };
    counter!("request_classification_unknown_labels_total", "field" => field).increment(1);
}

pub fn record_request_classification(intent: &str, domain: &str, route: &str) {
    let intent = crate::request_classification::bounded_intent(intent);
    let domain = crate::request_classification::bounded_domain(domain);
    let route = crate::request_classification::bounded_route(route);
    counter!(
        "request_classifications_total",
        "intent" => intent,
        "domain" => domain,
        "route" => route
    )
    .increment(1);
}

pub fn record_request_route_recommendation(route: &str) {
    let route = crate::request_classification::bounded_route(route);
    counter!("request_route_recommendations_total", "route" => route).increment(1);
}

pub fn record_request_risk_flag(risk: &str) {
    let risk = crate::request_classification::bounded_risk(risk);
    counter!("request_risk_flags_total", "risk" => risk).increment(1);
}

pub fn record_request_complexity(complexity: &str) {
    let complexity = crate::request_classification::bounded_complexity(complexity);
    counter!("request_complexity_total", "complexity" => complexity).increment(1);
}

pub fn record_request_live_policy_action(action: &str, reason: &str) {
    let action = crate::request_classification::bounded_live_policy_action(action);
    let reason = crate::request_classification::bounded_live_policy_reason(reason);
    counter!(
        "request_live_policy_actions_total",
        "action" => action,
        "reason" => reason
    )
    .increment(1);
}

pub fn record_request_live_policy_bypass(reason: &str) {
    let reason = crate::request_classification::bounded_live_policy_bypass(reason);
    counter!("request_live_policy_bypassed_total", "reason" => reason).increment(1);
}
