use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::Instant;

#[derive(Clone)]
pub struct RateLimiter {
    inner: Arc<Mutex<HashMap<String, Bucket>>>,
    rate_per_second: f64,
    burst: f64,
}

#[derive(Debug)]
struct Bucket {
    tokens: f64,
    updated_at: Instant,
}

impl RateLimiter {
    pub fn new(requests_per_minute: u32, burst: u32) -> Self {
        let rpm = requests_per_minute.max(1);
        let burst = burst.max(1);
        Self {
            inner: Arc::new(Mutex::new(HashMap::new())),
            rate_per_second: rpm as f64 / 60.0,
            burst: burst as f64,
        }
    }

    pub fn check(&self, key: &str) -> Result<(), u64> {
        let now = Instant::now();
        let mut buckets = self.inner.lock().unwrap();
        let bucket = buckets.entry(key_hash(key)).or_insert(Bucket {
            tokens: self.burst,
            updated_at: now,
        });

        let elapsed = now.duration_since(bucket.updated_at).as_secs_f64();
        bucket.tokens = (bucket.tokens + elapsed * self.rate_per_second).min(self.burst);
        bucket.updated_at = now;

        if bucket.tokens >= 1.0 {
            bucket.tokens -= 1.0;
            Ok(())
        } else {
            let wait_seconds = ((1.0 - bucket.tokens) / self.rate_per_second)
                .ceil()
                .max(1.0) as u64;
            Err(wait_seconds)
        }
    }
}

impl Default for RateLimiter {
    fn default() -> Self {
        Self::new(60, 30)
    }
}

pub fn key_hash(key: &str) -> String {
    let digest = Sha256::digest(key.as_bytes());
    hex_prefix(&digest, 16)
}

fn hex_prefix(bytes: &[u8], chars: usize) -> String {
    let mut out = String::with_capacity(chars);
    for byte in bytes {
        if out.len() >= chars {
            break;
        }
        out.push_str(&format!("{byte:02x}"));
    }
    out.truncate(chars);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rate_limit_applies_per_key() {
        let limiter = RateLimiter::new(60, 1);
        assert!(limiter.check("key-a").is_ok());
        assert!(limiter.check("key-a").is_err());
        assert!(limiter.check("key-b").is_ok());
    }

    #[test]
    fn key_hash_never_exposes_raw_key() {
        let hash = key_hash("secret-token");
        assert_eq!(hash.len(), 16);
        assert!(!hash.contains("secret"));
    }
}
