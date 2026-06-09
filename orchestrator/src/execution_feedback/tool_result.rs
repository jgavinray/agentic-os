#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CapturedToolResult {
    pub tool_name: String,
    pub content: String,
    pub exit_code: i32,
    pub duration_ms: u64,
    pub stdout_summary: String,
    pub stderr_summary: String,
}
