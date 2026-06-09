use axum::http::HeaderMap;

#[derive(Debug, Clone)]
pub(crate) struct HandlerRequestScope {
    pub namespace: String,
    pub repo: String,
    pub task: String,
}

impl HandlerRequestScope {
    pub(crate) fn from_headers(headers: &HeaderMap, namespace: String, default_task: &str) -> Self {
        let repo = headers
            .get("x-agent-repo")
            .and_then(|v| v.to_str().ok())
            .map(str::to_string)
            .unwrap_or_else(|| namespace.clone());
        let task = headers
            .get("x-agent-task")
            .and_then(|v| v.to_str().ok())
            .map(str::to_string)
            .unwrap_or_else(|| default_task.to_string());
        Self {
            namespace,
            repo,
            task,
        }
    }

    pub(crate) fn apply_to_capture(&self, capture: &mut crate::client_capture::RawHttpCapture) {
        capture.namespace = Some(self.namespace.clone());
        capture.repo = Some(self.repo.clone());
        capture.task = Some(self.task.clone());
    }
}
