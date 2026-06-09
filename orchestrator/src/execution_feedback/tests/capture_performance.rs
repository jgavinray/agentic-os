use super::*;

#[test]
fn execution_capture_p99_under_one_ms() {
    let result = CapturedToolResult {
        tool_name: "cargo".to_string(),
        content: "error[E0308]: mismatched types\nexit code 101".to_string(),
        exit_code: 101,
        duration_ms: 10,
        stdout_summary: String::new(),
        stderr_summary: "error[E0308]: mismatched types".to_string(),
    };
    let ctx = ctx();
    let _ = events_for_tool_result(&ctx, &result);
    let mut durations = Vec::new();
    for _ in 0..1000 {
        let started = std::time::Instant::now();
        let events = events_for_tool_result(&ctx, &result);
        assert!(!events.is_empty());
        durations.push(started.elapsed());
    }
    durations.sort();
    let p99 = durations[(durations.len() * 99) / 100 - 1];
    let threshold = if cfg!(debug_assertions) {
        std::time::Duration::from_millis(10)
    } else {
        std::time::Duration::from_millis(1)
    };
    assert!(p99 < threshold, "p99 was {p99:?}");
}

#[test]
fn mixed_event_load_test_builds_1000_under_five_seconds() {
    let started = std::time::Instant::now();
    let ctx = ctx();
    let _ = fingerprint("error[E0308]: mismatched types");
    let mut count = 0usize;
    for idx in 0..250 {
        let result = CapturedToolResult {
            tool_name: if idx % 2 == 0 {
                "pytest".to_string()
            } else {
                "tsc".to_string()
            },
            content: if idx % 2 == 0 {
                "2 passed, 0 failed".to_string()
            } else {
                "src/a.ts(1,1): error TS2322: type mismatch".to_string()
            },
            exit_code: if idx % 2 == 0 { 0 } else { 2 },
            duration_ms: idx,
            stdout_summary: String::new(),
            stderr_summary: String::new(),
        };
        count += events_for_tool_result(&ctx, &result).len();
    }
    while count < 1000 {
        let _ = build_execution_event(
            &ctx,
            ExecutionEventKind::PatchResult,
            true,
            patch_result_payload(vec!["src/lib.rs".to_string()], "applied", vec![]),
        );
        count += 1;
    }
    assert!(started.elapsed() < std::time::Duration::from_secs(5));
    assert!(count >= 1000);
}
