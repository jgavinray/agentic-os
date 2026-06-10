//! Golden classification corpus.
//!
//! Each case is a paraphrased shape of real captured traffic (agent_events /
//! agent_request_classifications). When a classifier change breaks one of
//! these, the change regresses a request shape that actually occurs in
//! production traffic — fix the rule, or update the corpus deliberately and
//! say why in the commit.

use super::test_support::event;
use super::*;

#[test]
fn golden_corpus_intents() {
    let cases: &[(&str, &str, RequestIntent)] = &[
        // --- implementation shapes ---
        (
            "c-doc-comment",
            "Add a doc comment to main.rs explaining the entry point, then run cargo check to validate, and report the result.",
            RequestIntent::Implement,
        ),
        (
            "c-impl-task",
            "Rust implementation task in this repository. No discovery phase. Apply the change to handlers.rs and run the tests.",
            RequestIntent::Implement,
        ),
        (
            "c-stalled-retry",
            "Work in this repository. The previous run stalled gathering facts and made no edits. Do the implementation now and keep it narrow.",
            RequestIntent::Implement,
        ),
        (
            "c-revise-narrow",
            "Revise the telemetry slice you just implemented. Keep the change narrow. Fix these concerns from review.",
            RequestIntent::Implement,
        ),
        (
            "c-sql-typo",
            "Fix one SQL typo in store.go. Replace transcript_name==excluded with transcript_name=excluded.",
            RequestIntent::Implement,
        ),
        (
            "c-terse-shorthand",
            "rule_utils.rs: word boundary chars must exclude underscore",
            RequestIntent::Implement,
        ),
        // --- config shapes ---
        (
            "c-gen-manifest",
            "Generate a Kubernetes manifest for the embedding service.",
            RequestIntent::GenerateConfig,
        ),
        (
            "c-modify-yaml",
            "Update litellm-config.yaml to add the new model alias.",
            RequestIntent::ModifyConfig,
        ),
        // --- operate shapes ---
        (
            "c-run-fmt",
            "In the orchestrator directory, run cargo fmt. Do not run any other command. Do not restart services.",
            RequestIntent::OperateTool,
        ),
        // --- debug shapes ---
        (
            "c-diagnose-logs",
            "Diagnose this runtime error from logs: connection refused on localhost:8088 after restart.",
            RequestIntent::Debug,
        ),
        // --- read-only shapes ---
        (
            "c-why-question",
            "Why does this need to be run outside of active hours?",
            RequestIntent::Explain,
        ),
        (
            "c-k8s-writeup",
            "Write a detailed technical explanation of the Kubernetes control plane, covering the API server, etcd, and the scheduler.",
            RequestIntent::Explain,
        ),
        (
            "c-explain-pack",
            "Explain how the context pack builder chooses sources.",
            RequestIntent::Explain,
        ),
        ("c-read-file", "Read README.md", RequestIntent::Explain),
        (
            "c-latest-release",
            "What is the latest vLLM release today?",
            RequestIntent::Search,
        ),
        (
            "c-plan-split",
            "Propose an approach for splitting the summarizer into modules; do not change any code yet.",
            RequestIntent::Plan,
        ),
        (
            "c-summarize-session",
            "Summarize what changed in the last session.",
            RequestIntent::Summarize,
        ),
        // --- negation shapes ---
        (
            "c-negated-implement",
            "Do not implement this yet; review the spec for feature 02 and list the open questions.",
            RequestIntent::Explain,
        ),
    ];

    for (id, text, expected) in cases {
        let row = classify_request_event(&event(id, text, None));
        assert_eq!(
            row.intent, *expected,
            "corpus case {id} text: {text:?} (runner_up={:?}, margin={:?})",
            row.features["intent_runner_up"], row.features["intent_margin"]
        );
    }
}
