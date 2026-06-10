use super::test_support::event;
use super::*;
use serde_json::json;

#[test]
fn secret_candidates_set_secret_present() {
    let row = classify_request_event(&event(
        "e-secret",
        "The request included Authorization: Bearer sk-secret-value",
        None,
    ));

    assert!(row.risk.contains(&RequestRisk::SecretPresent));
    assert_eq!(row.recommended_route, RecommendedRoute::RefuseOrGuardrail);
    assert_eq!(row.features["has_secret_candidate"], true);
}

#[test]
fn destructive_commands_set_destructive_command() {
    let row = classify_request_event(&event(
        "e-destructive",
        "Run rm -rf /tmp/agentic-os-cache to clean everything",
        None,
    ));

    assert!(row.risk.contains(&RequestRisk::DestructiveCommand));
    assert_eq!(row.complexity, RequestComplexity::L5HighRisk);
    assert_eq!(row.recommended_route, RecommendedRoute::RefuseOrGuardrail);
}

#[test]
fn latest_current_phrasing_sets_external_current_info_required() {
    let row = classify_request_event(&event(
        "e-latest",
        "What is the latest vLLM release today?",
        None,
    ));

    assert!(row.risk.contains(&RequestRisk::ExternalCurrentInfoRequired));
    assert_eq!(row.domain, RequestDomain::LlmInference);
    assert_eq!(row.recommended_route, RecommendedRoute::WebRequired);
}

#[test]
fn high_stakes_domains_set_high_stakes_risk() {
    for (id, summary, domain) in [
        (
            "e-medical",
            "Can you diagnose this medical symptom?",
            RequestDomain::Medical,
        ),
        (
            "e-legal",
            "Is this legal contract enforceable?",
            RequestDomain::Legal,
        ),
        (
            "e-finance",
            "Should I make this tax and investment move?",
            RequestDomain::Finance,
        ),
    ] {
        let row = classify_request_event(&event(id, summary, None));
        assert_eq!(row.domain, domain);
        assert!(row.risk.contains(&RequestRisk::HighStakes));
        assert_eq!(row.recommended_route, RecommendedRoute::RefuseOrGuardrail);
    }
}

#[test]
fn high_stakes_domain_markers_do_not_match_embedded_software_terms() {
    for (id, summary) in [
        (
            "e-taxonomy",
            "Implement the prompt intervention taxonomy module.",
        ),
        (
            "e-record-contract",
            "Read docs/PromptInterventions/02-record-contract.md and implement the runtime hook.",
        ),
        (
            "e-restock",
            "Patch the restock_inventory.rs fixture in this repository.",
        ),
    ] {
        let row = classify_request_event(&event(id, summary, None));
        assert_ne!(row.domain, RequestDomain::Medical);
        assert_ne!(row.domain, RequestDomain::Legal);
        assert_ne!(row.domain, RequestDomain::Finance);
        assert!(!row.risk.contains(&RequestRisk::HighStakes));
    }
}

#[test]
fn high_stakes_domain_markers_match_later_standalone_occurrences() {
    let row = classify_request_event(&event(
        "e-later-standalone-marker",
        "The taxonomy module should not match, but tax guidance should.",
        None,
    ));

    assert_eq!(row.domain, RequestDomain::Finance);
    assert!(row.risk.contains(&RequestRisk::HighStakes));
}

#[test]
fn infrastructure_terms_map_to_expected_domains() {
    let row = classify_request_event(&event(
        "e-domains",
        "kubectl deployment, Docker container, vLLM model, DNS proxy, and JWT auth",
        None,
    ));

    assert_eq!(row.domain, RequestDomain::Kubernetes);
    assert!(row.secondary_domains.contains(&RequestDomain::Docker));
    assert!(row.secondary_domains.contains(&RequestDomain::LlmInference));
    assert!(row.secondary_domains.contains(&RequestDomain::Networking));
    assert!(row.secondary_domains.contains(&RequestDomain::Security));
    assert!(row.secondary_domains.contains(&RequestDomain::Shell));
    assert_eq!(
        row.features["detected_domain_terms"],
        json!([
            "kubernetes",
            "docker",
            "llm_inference",
            "networking",
            "security",
            "shell"
        ])
    );
}
