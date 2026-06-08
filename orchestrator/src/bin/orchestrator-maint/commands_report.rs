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
}
