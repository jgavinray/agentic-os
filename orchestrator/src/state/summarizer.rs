/// Default max_tokens used by the internal summarizer; summaries are short.
pub const SUMMARIZER_MAX_TOKENS: u64 = 384;

/// Cap source events consumed per summarization promotion pass.
pub const MAX_SUMMARIZER_EVENTS: i64 = 10;

/// Prompt templates for summary hierarchy promotion:
/// index 0 => raw events to L1, index 1 => L1 to L2, index 2 => L2 to L3.
pub const SUMMARY_PROMPTS: [&str; 3] = [
    "\
You are a precise technical summarizer. Extract information from conversation messages \
into these exact sections. Include only what is explicitly stated. \
Output nothing else — no preamble, no explanation.

DECISIONS:
(one decision per line, or the word \"none\")
OPEN_QUESTIONS:
(one question per line, or the word \"none\")
FAILED_APPROACHES:
(one failed approach per line, or the word \"none\")
KEY_FACTS:
(one key fact per line, or the word \"none\")

Messages:
{messages}",
    "\
You are consolidating event-level engineering summaries into a session-level summary. \
Keep recurring decisions, unresolved questions, failed approaches, and facts that would \
help resume the work later. Output only the consolidated summary.

Event summaries:
{messages}",
    "\
You are producing an executive engineering memory summary from session-level summaries. \
Keep durable architecture decisions, recurring risks, known failed approaches, and the \
current project state. Output only concise durable memory.

Session summaries:
{messages}",
];
