use crate::feature_extraction_types::DetectionTag;
use serde_json::Value;

pub(crate) fn bootstrap_text_tags(
    summary: &str,
    evidence: Option<&str>,
    metadata: &Value,
    source_override: Option<&str>,
) -> Vec<DetectionTag> {
    let source = source_override.unwrap_or("bootstrap_migration");
    let haystack = format!(
        "{}\n{}\n{}",
        summary,
        evidence.unwrap_or_default(),
        metadata
    )
    .to_ascii_lowercase();
    let mut tags = Vec::new();

    if haystack.contains("repeated read")
        || haystack.contains("read loop")
        || haystack.contains("read tool loop")
        || haystack.contains("tool 'read' called")
        || haystack.contains("tool \"read\" called")
        || haystack.contains("loop warning: tool 'read'")
    {
        tags.push(DetectionTag::tool_loop(source, "Read"));
    } else if haystack.contains("repeated bash")
        || haystack.contains("bash loop")
        || haystack.contains("bash tool loop")
        || haystack.contains("tool 'bash' called")
        || haystack.contains("tool \"bash\" called")
        || haystack.contains("loop warning: tool 'bash'")
    {
        tags.push(DetectionTag::tool_loop(source, "Bash"));
    } else if haystack.contains("tool loop") || haystack.contains("repeat identical tool") {
        tags.push(DetectionTag::tool_loop(source, "unknown"));
    }

    if haystack.contains("user interruption")
        || haystack.contains("user interrupted")
        || haystack.contains("user correction")
        || haystack.contains("request interrupted by user")
        || haystack.contains("interrupted by user for tool use")
    {
        tags.push(DetectionTag::new("user_interruption", source));
    }
    if haystack.contains("missing authorization")
        || haystack.contains("missing auth")
        || haystack.contains("authorization header")
        || haystack.contains("unauthorized")
        || haystack.contains("without authorization header")
        || haystack.contains("without the authorization header")
        || (haystack.contains("without") && haystack.contains("bearer"))
    {
        tags.push(DetectionTag::new("missing_auth", source));
    }
    if (haystack.contains("wrong endpoint") || haystack.contains("incorrect endpoint"))
        || (haystack.contains("localhost") && haystack.contains("correct endpoint"))
        || (haystack.contains("localhost") && haystack.contains("should be using"))
        || haystack.contains("trying localhost")
        || haystack.contains("not localhost")
    {
        tags.push(DetectionTag::new("wrong_endpoint", source));
    }
    if haystack.contains("summarization failure")
        || haystack.contains("summary failure")
        || haystack.contains("empty summary")
        || (haystack.contains("summarization") && haystack.contains("empty response"))
    {
        tags.push(DetectionTag::new("summarization_failure", source));
    }
    if haystack.contains("migration failure")
        || haystack.contains("baseline migration")
        || haystack.contains("extension creation")
    {
        tags.push(DetectionTag::new("migration_failure", source));
    }
    if haystack.contains("acknowledge")
        && (haystack.contains("correction") || haystack.contains("interruption"))
    {
        tags.push(DetectionTag::new("correction_acknowledged", source));
    }

    tags
}
