use orchestrator::prompt_intervention_report;
use orchestrator::request_classification;

pub(super) fn print_request_classification_report(
    report: &request_classification::RequestClassificationReport,
) {
    println!("request-classification-report:");
    println!("by_route:");
    for row in &report.by_route {
        println!("  {} {}", row.label, row.count);
    }
    println!("top_risk_flags:");
    for row in &report.top_risk_flags {
        println!("  {} {}", row.label, row.count);
    }
    println!("unknown_label_counts:");
    for row in &report.unknown_label_counts {
        println!("  {} {}", row.label, row.count);
    }
    println!("repeated_guardrail_sessions:");
    for row in &report.repeated_guardrail_sessions {
        println!("  {} {}", row.session_id, row.count);
    }
    println!("low_margin_intents (corpus labeling candidates):");
    for row in &report.low_margin_intents {
        println!(
            "  {} intent={} runner_up={} margin={}",
            row.event_id,
            row.intent,
            row.runner_up.as_deref().unwrap_or("-"),
            row.margin
        );
    }
}

pub(super) fn print_prompt_intervention_report(
    report: &prompt_intervention_report::PromptInterventionReport,
) {
    for line in prompt_intervention_report::report_lines(report) {
        println!("{line}");
    }
}
