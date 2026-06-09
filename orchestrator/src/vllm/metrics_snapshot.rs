#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct VllmCacheSnapshot {
    pub prefix_cache_queries: i64,
    pub prefix_cache_hits: i64,
    pub prompt_tokens_total: i64,
    pub prompt_tokens_cached: i64,
    pub prompt_tokens_local_compute: i64,
    pub prompt_tokens_local_cache_hit: i64,
    pub prompt_tokens_external_kv: i64,
    pub kv_cache_usage_perc: Option<f64>,
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct VllmCacheDelta {
    pub prefix_cache_queries_delta: i64,
    pub prefix_cache_hits_delta: i64,
    pub prompt_tokens_total_delta: i64,
    pub prompt_tokens_cached_delta: i64,
    pub prompt_tokens_local_compute_delta: i64,
    pub prompt_tokens_local_cache_hit_delta: i64,
    pub prompt_tokens_external_kv_delta: i64,
    pub kv_cache_usage_before: Option<f64>,
    pub kv_cache_usage_after: Option<f64>,
}

impl VllmCacheSnapshot {
    pub fn delta_since(self, before: Self) -> VllmCacheDelta {
        VllmCacheDelta {
            prefix_cache_queries_delta: nonnegative_delta(
                self.prefix_cache_queries,
                before.prefix_cache_queries,
            ),
            prefix_cache_hits_delta: nonnegative_delta(
                self.prefix_cache_hits,
                before.prefix_cache_hits,
            ),
            prompt_tokens_total_delta: nonnegative_delta(
                self.prompt_tokens_total,
                before.prompt_tokens_total,
            ),
            prompt_tokens_cached_delta: nonnegative_delta(
                self.prompt_tokens_cached,
                before.prompt_tokens_cached,
            ),
            prompt_tokens_local_compute_delta: nonnegative_delta(
                self.prompt_tokens_local_compute,
                before.prompt_tokens_local_compute,
            ),
            prompt_tokens_local_cache_hit_delta: nonnegative_delta(
                self.prompt_tokens_local_cache_hit,
                before.prompt_tokens_local_cache_hit,
            ),
            prompt_tokens_external_kv_delta: nonnegative_delta(
                self.prompt_tokens_external_kv,
                before.prompt_tokens_external_kv,
            ),
            kv_cache_usage_before: before.kv_cache_usage_perc,
            kv_cache_usage_after: self.kv_cache_usage_perc,
        }
    }
}

fn nonnegative_delta(after: i64, before: i64) -> i64 {
    after.saturating_sub(before).max(0)
}

pub async fn fetch_cache_snapshot(
    http: &reqwest::Client,
    metrics_url: &str,
) -> Result<VllmCacheSnapshot, anyhow::Error> {
    let text = http
        .get(metrics_url)
        .send()
        .await?
        .error_for_status()?
        .text()
        .await?;
    Ok(parse_cache_snapshot(&text))
}

pub fn parse_cache_snapshot(text: &str) -> VllmCacheSnapshot {
    let mut snapshot = VllmCacheSnapshot::default();
    for line in text.lines().map(str::trim) {
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let Some((series, value)) = line.rsplit_once(' ') else {
            continue;
        };
        let Ok(value) = value.parse::<f64>() else {
            continue;
        };
        let metric = series
            .split_once('{')
            .map(|(name, _)| name)
            .unwrap_or(series);
        let rounded = value.round() as i64;
        match metric {
            "vllm:prefix_cache_queries_total" => snapshot.prefix_cache_queries += rounded,
            "vllm:prefix_cache_hits_total" => snapshot.prefix_cache_hits += rounded,
            "vllm:prompt_tokens_total" => snapshot.prompt_tokens_total += rounded,
            "vllm:prompt_tokens_cached_total" => snapshot.prompt_tokens_cached += rounded,
            "vllm:kv_cache_usage_perc" => {
                snapshot.kv_cache_usage_perc = Some(
                    snapshot
                        .kv_cache_usage_perc
                        .map_or(value, |current| current.max(value)),
                );
            }
            "vllm:prompt_tokens_by_source_total" => {
                if series.contains(r#"source="local_compute""#) {
                    snapshot.prompt_tokens_local_compute += rounded;
                } else if series.contains(r#"source="local_cache_hit""#) {
                    snapshot.prompt_tokens_local_cache_hit += rounded;
                } else if series.contains(r#"source="external_kv_transfer""#) {
                    snapshot.prompt_tokens_external_kv += rounded;
                }
            }
            _ => {}
        }
    }
    snapshot
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_vllm_cache_metrics() {
        let text = r#"
# HELP vllm:prefix_cache_queries_total Prefix cache queries.
vllm:prefix_cache_queries_total{engine="0",model_name="m"} 100.0
vllm:prefix_cache_hits_total{engine="0",model_name="m"} 60.0
vllm:prompt_tokens_total{engine="0",model_name="m"} 200.0
vllm:prompt_tokens_cached_total{engine="0",model_name="m"} 70.0
vllm:prompt_tokens_by_source_total{engine="0",model_name="m",source="local_compute"} 130.0
vllm:prompt_tokens_by_source_total{engine="0",model_name="m",source="local_cache_hit"} 65.0
vllm:prompt_tokens_by_source_total{engine="0",model_name="m",source="external_kv_transfer"} 5.0
vllm:kv_cache_usage_perc{engine="0",model_name="m"} 0.25
"#;
        let snapshot = parse_cache_snapshot(text);
        assert_eq!(snapshot.prefix_cache_queries, 100);
        assert_eq!(snapshot.prefix_cache_hits, 60);
        assert_eq!(snapshot.prompt_tokens_total, 200);
        assert_eq!(snapshot.prompt_tokens_cached, 70);
        assert_eq!(snapshot.prompt_tokens_local_compute, 130);
        assert_eq!(snapshot.prompt_tokens_local_cache_hit, 65);
        assert_eq!(snapshot.prompt_tokens_external_kv, 5);
        assert_eq!(snapshot.kv_cache_usage_perc, Some(0.25));
    }

    #[test]
    fn computes_nonnegative_delta() {
        let before = VllmCacheSnapshot {
            prefix_cache_queries: 100,
            prefix_cache_hits: 40,
            ..Default::default()
        };
        let after = VllmCacheSnapshot {
            prefix_cache_queries: 125,
            prefix_cache_hits: 50,
            ..Default::default()
        };
        let delta = after.delta_since(before);
        assert_eq!(delta.prefix_cache_queries_delta, 25);
        assert_eq!(delta.prefix_cache_hits_delta, 10);
    }
}
