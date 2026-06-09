use super::*;

#[test]
fn sampling_capture_p99_under_one_ms() {
    let original = json!({
        "temperature": 0.3,
        "top_p": 0.9,
        "max_tokens": 4096,
        "seed": 123
    });
    let mut durations = Vec::new();

    for _ in 0..1000 {
        let started = std::time::Instant::now();
        let mut forwarded = original.clone();
        let audit = capture_and_maybe_override(
            &original,
            &mut forwarded,
            SamplingConfig::new(true, false).unwrap(),
            &NoOpSamplingPolicy,
        );
        assert!(audit.is_some());
        durations.push(started.elapsed());
    }

    durations.sort();
    let p99 = durations[(durations.len() * 99) / 100 - 1];
    assert!(p99 < std::time::Duration::from_millis(1), "p99 was {p99:?}");
}

#[test]
fn noop_override_p99_under_one_ms() {
    let original = json!({
        "temperature": 0.3,
        "top_p": 0.9,
        "max_tokens": 4096,
        "seed": 123
    });
    let mut durations = Vec::new();

    for _ in 0..1000 {
        let started = std::time::Instant::now();
        let mut forwarded = original.clone();
        let audit = capture_and_maybe_override(
            &original,
            &mut forwarded,
            SamplingConfig::new(true, true).unwrap(),
            &NoOpSamplingPolicy,
        );
        assert!(audit.is_some());
        durations.push(started.elapsed());
    }

    durations.sort();
    let p99 = durations[(durations.len() * 99) / 100 - 1];
    assert!(p99 < std::time::Duration::from_millis(1), "p99 was {p99:?}");
}
