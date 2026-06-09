use metrics::{counter, histogram};
use std::time::Instant;

pub struct StreamTracker {
    path: &'static str,
    started: Instant,
    first_token_seen: bool,
    completed: bool,
}

impl StreamTracker {
    pub fn new(path: &'static str, started: Instant) -> Self {
        Self {
            path,
            started,
            first_token_seen: false,
            completed: false,
        }
    }

    pub fn first_token(&mut self) {
        if !self.first_token_seen {
            histogram!("stream_first_token_seconds", "path" => self.path)
                .record(self.started.elapsed().as_secs_f64());
            self.first_token_seen = true;
        }
    }

    pub fn finish(&mut self) {
        self.completed = true;
        histogram!("stream_duration_seconds", "path" => self.path)
            .record(self.started.elapsed().as_secs_f64());
    }

    pub fn fail(&mut self, reason: &'static str) {
        self.completed = true;
        histogram!("stream_duration_seconds", "path" => self.path)
            .record(self.started.elapsed().as_secs_f64());
        counter!("stream_disconnects_total", "path" => self.path, "reason" => reason).increment(1);
    }
}

impl Drop for StreamTracker {
    fn drop(&mut self) {
        if !self.completed {
            histogram!("stream_duration_seconds", "path" => self.path)
                .record(self.started.elapsed().as_secs_f64());
            counter!(
                "stream_disconnects_total",
                "path" => self.path,
                "reason" => "client_disconnect"
            )
            .increment(1);
        }
    }
}
