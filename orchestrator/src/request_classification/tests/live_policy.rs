use crate::db::AgentEvent;
use crate::request_classification::{
    bounded_complexity, bounded_live_policy_action, bounded_live_policy_bypass,
    bounded_live_policy_reason, bounded_risk, bounded_route, classify_request_event,
    evaluate_live_policy, LivePolicyConfig, RecommendedRoute, ResponseContract,
};
use chrono::Utc;
use serde_json::json;

#[test]
fn live_policy_is_disabled_by_default() {
    let classification = classify_request_event(&event(
        "e-disabled",
        "Can you diagnose this medical issue?",
        None,
    ));

    assert_eq!(
        evaluate_live_policy(&classification, &LivePolicyConfig::default()),
        None
    );
}

#[test]
fn live_policy_handles_high_stakes_current_info_and_destructive_commands() {
    let enabled = enabled_policy();

    let medical = classify_request_event(&event(
        "e-live-medical",
        "Can you diagnose this medical issue?",
        None,
    ));
    let decision = evaluate_live_policy(&medical, &enabled).expect("medical should be stopped");
    assert_eq!(decision.action, "refuse_or_guardrail");
    assert_eq!(decision.reason, "objective_risk");
    assert_eq!(decision.response_contract, ResponseContract::Refusal);

    let latest = classify_request_event(&event(
        "e-live-latest",
        "What is the latest Docker release today?",
        None,
    ));
    let decision = evaluate_live_policy(&latest, &enabled).expect("latest should need web");
    assert_eq!(decision.action, "web_required");
    assert_eq!(decision.reason, "external_current_info_required");

    let destructive = classify_request_event(&event(
        "e-live-destructive",
        "Please run kubectl delete namespace production",
        None,
    ));
    let decision =
        evaluate_live_policy(&destructive, &enabled).expect("destructive should be stopped");
    assert_eq!(decision.action, "refuse_or_guardrail");
    assert_eq!(decision.reason, "objective_risk");
}

#[test]
fn clarification_route_returns_bounded_contract() {
    let enabled = enabled_policy();
    let classification = classify_request_event(&event("e-clarify", "Please restart it", None));

    assert_eq!(
        classification.recommended_route,
        RecommendedRoute::AskClarification
    );
    let decision =
        evaluate_live_policy(&classification, &enabled).expect("clarification should apply");
    assert_eq!(decision.action, "ask_clarification");
    assert_eq!(
        decision.response_contract,
        ResponseContract::ClarificationQuestion
    );
}

#[test]
fn small_and_strong_model_routes_remain_shadow_only() {
    let enabled = enabled_policy();
    let small = classify_request_event(&event(
        "e-small",
        "Please explain how Docker container networking works at a high level",
        None,
    ));
    assert_eq!(small.recommended_route, RecommendedRoute::SmallLocalModel);
    assert_eq!(evaluate_live_policy(&small, &enabled), None);

    let long_text = format!("Explain this architecture. {}", "detail ".repeat(400));
    let strong = classify_request_event(&event("e-strong", &long_text, None));
    assert_eq!(strong.recommended_route, RecommendedRoute::StrongLocalModel);
    assert_eq!(evaluate_live_policy(&strong, &enabled), None);
}

#[test]
fn live_policy_bounding_helpers_reject_unbounded_labels() {
    assert_eq!(
        bounded_live_policy_action("something else"),
        "refuse_or_guardrail"
    );
    assert_eq!(
        bounded_live_policy_reason("raw path /tmp/foo"),
        "objective_risk"
    );
    assert_eq!(bounded_live_policy_bypass("raw unknown"), "shadow_only");
    assert_eq!(bounded_route("bad route"), "unknown");
    assert_eq!(bounded_risk("bad risk"), "unknown");
    assert_eq!(bounded_complexity("bad complexity"), "unknown");
}

fn enabled_policy() -> LivePolicyConfig {
    LivePolicyConfig {
        enabled: true,
        policy_version: "v1".to_string(),
    }
}

fn event(id: &str, summary: &str, evidence: Option<&str>) -> AgentEvent {
    AgentEvent {
        id: id.to_string(),
        session_id: "session-1".to_string(),
        repo: "agent-os".to_string(),
        actor: "user".to_string(),
        event_type: "user_message".to_string(),
        summary: summary.to_string(),
        evidence: evidence.map(str::to_string),
        metadata: json!({
            "payload": {
                "kind": "test"
            }
        }),
        correlation_id: None,
        parent_event_id: None,
        trajectory_id: None,
        attempt_index: None,
        event_role: Some("request".to_string()),
        created_at: chrono::DateTime::parse_from_rfc3339("2026-05-23T12:00:00Z")
            .unwrap()
            .with_timezone(&Utc),
        summary_level: 0,
    }
}
