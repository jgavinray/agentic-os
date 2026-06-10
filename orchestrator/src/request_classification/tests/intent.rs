use super::test_support::event;
use super::*;
use serde_json::json;

#[test]
fn composite_requests_emit_bounded_sub_intents() {
    let row = classify_request_event(&event(
        "e-composite",
        "Search the repo for context injection; implement the fix in src/main.rs; run cargo test; summarize the result",
        None,
    ));

    assert_eq!(row.features["is_composite"], true);
    assert_eq!(row.features["decomposition_candidate"], true);
    assert_eq!(row.features["decomposition_reason"], "sequence_separator");
    assert_eq!(row.features["sub_intent_count"], 4);
    assert_eq!(
        row.features["sub_intents"],
        json!(["search", "implement", "operate_tool", "summarize"])
    );
}

#[test]
fn implementation_language_maps_to_implement_intent() {
    let row = classify_request_event(&event(
        "e-implement",
        "Implement the classifier change in src/request_classification.rs",
        None,
    ));

    assert_eq!(row.intent, RequestIntent::Implement);
    assert_eq!(row.response_contract, ResponseContract::ValidationRequired);
}

#[test]
fn implementation_with_reporting_summary_stays_implement() {
    let row = classify_request_event(&event(
        "e-implement-summary",
        "Continue the prompt-intervention implementation in orchestrator/src/tool_mediation/classification.rs, run relevant checks, and leave a concise summary of changes.",
        None,
    ));

    assert_eq!(row.intent, RequestIntent::Implement);
    assert_eq!(row.response_contract, ResponseContract::ValidationRequired);
}

#[test]
fn current_implementation_does_not_require_external_info() {
    let row = classify_request_event(&event(
        "e-current-implementation",
        "Inspect the current implementation and implement the missing policy-to-tool pairing.",
        None,
    ));

    assert_eq!(row.intent, RequestIntent::Implement);
    assert!(!row.risk.contains(&RequestRisk::ExternalCurrentInfoRequired));
}

#[test]
fn bug_fix_language_maps_to_implement_intent() {
    let row = classify_request_event(&event(
        "e-fix-bug",
        "Fix this bug the way the implementation should work.",
        None,
    ));

    assert_eq!(row.intent, RequestIntent::Implement);
    assert_eq!(row.response_contract, ResponseContract::ValidationRequired);
}

#[test]
fn research_requests_are_not_search_intent() {
    // "research" must not substring-match the "search" trigger, which would
    // route the request into a read-only search policy.
    let row = classify_request_event(&event(
        "e-research",
        "Research the existing handler modules and document how streaming works.",
        None,
    ));

    assert_ne!(row.intent, RequestIntent::Search);
}

#[test]
fn additive_code_requests_map_to_implement_intent() {
    let row = classify_request_event(&event(
        "e-add-helper",
        "Add a word-boundary helper to rule_utils.rs and update the call sites.",
        None,
    ));

    assert_eq!(row.intent, RequestIntent::Implement);
}

#[test]
fn refactor_requests_map_to_implement_intent() {
    let row = classify_request_event(&event(
        "e-refactor",
        "Refactor the streaming handler so persistence is a separate module.",
        None,
    ));

    assert_eq!(row.intent, RequestIntent::Implement);
}

#[test]
fn imperative_mutation_requests_map_to_implement() {
    for (id, summary) in [
        (
            "e-make",
            "Make the orchestrator return 429 when the rate limit is exceeded.",
        ),
        ("e-remove", "Remove the dead code path in handlers/mod.rs."),
        ("e-rename", "Rename the hybrid module to retrieval."),
    ] {
        let row = classify_request_event(&event(id, summary, None));
        assert_eq!(row.intent, RequestIntent::Implement, "case {id}");
    }
}

#[test]
fn code_shaped_requests_without_verbs_fall_back_to_implement() {
    // Terse dev shorthand with a file reference but no recognized verb should
    // get the implement surface (the envelope is the safety layer), not a
    // read-only Explain menu that wedges the loop.
    let row = classify_request_event(&event(
        "e-terse",
        "rule_utils.rs: word boundary chars must exclude underscore",
        None,
    ));

    assert_eq!(row.intent, RequestIntent::Implement);
}

#[test]
fn explain_requests_with_file_paths_stay_explain() {
    let row = classify_request_event(&event(
        "e-explain-path",
        "Explain how handlers/mod.rs dispatches requests.",
        None,
    ));

    assert_eq!(row.intent, RequestIntent::Explain);
}

#[test]
fn intent_scoring_records_margin_and_runner_up() {
    let row = classify_request_event(&event(
        "e-scored",
        "Implement the retry policy and summarize what changed.",
        None,
    ));

    assert_eq!(row.intent, RequestIntent::Implement);
    assert_eq!(row.features["intent_runner_up"], "summarize");
    assert!(
        row.features["intent_margin"].as_u64().unwrap() > 0,
        "margin must be recorded for diagnosis"
    );
}

#[test]
fn explanatory_error_request_stays_debug_intent() {
    let row = classify_request_event(&event(
        "e-explain-error",
        "Please explain the Docker compose error",
        Some("ERROR failed to connect to http://localhost:8088"),
    ));

    assert_eq!(row.intent, RequestIntent::Debug);
}

#[test]
fn single_intent_with_conjunction_is_not_decomposed() {
    let row = classify_request_event(&event(
        "e-not-composite",
        "Explain Docker and Kubernetes networking",
        None,
    ));

    assert_eq!(row.features["is_composite"], false);
    assert_eq!(row.features["decomposition_candidate"], false);
    assert_eq!(row.features["decomposition_reason"], "none");
    assert_eq!(row.features["sub_intent_count"], 0);
    assert_eq!(row.features["sub_intents"], json!([]));
}

#[test]
fn unknown_or_empty_events_produce_bounded_safe_defaults() {
    let row = classify_request_event(&event("e-empty", "", None));

    assert_eq!(row.intent, RequestIntent::Unknown);
    assert_eq!(row.domain, RequestDomain::Unknown);
    assert_eq!(row.artifact_type, RequestArtifactType::Unknown);
    assert_eq!(row.risk, vec![RequestRisk::Unknown]);
    assert_eq!(row.complexity, RequestComplexity::Unknown);
    assert_eq!(row.recommended_route, RecommendedRoute::Unknown);
    assert_eq!(row.response_contract, ResponseContract::Unknown);
    assert_eq!(row.features["char_count"], 0);
    assert_eq!(row.features["estimated_tokens"], 0);
}

#[test]
fn non_empty_generic_requests_use_safe_fallback_labels() {
    let row = classify_request_event(&event("e-generic", "Can you help with this?", None));

    assert_eq!(row.intent, RequestIntent::Explain);
    assert_eq!(row.domain, RequestDomain::Generic);
    assert_eq!(row.artifact_type, RequestArtifactType::PlainText);
    assert_eq!(row.risk, vec![RequestRisk::None]);
    assert_ne!(row.complexity, RequestComplexity::Unknown);
    assert_ne!(row.recommended_route, RecommendedRoute::Unknown);
    assert_ne!(row.response_contract, ResponseContract::Unknown);
}
