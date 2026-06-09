use super::*;

#[test]
fn fingerprint_rules_cover_initial_classes() {
    let cases = [
        ("error[E0382]: use of moved value", "rust:borrow-checker"),
        ("error[E0308]: mismatched types", "rust:type-mismatch"),
        (
            "ModuleNotFoundError: No module named 'x'",
            "python:import-error",
        ),
        ("src/a.ts:1:2 - error TS2322", "typescript:TS2322"),
        ("JSONDecodeError: Expecting value", "json:parse-error"),
        ("process exited with code 2", "process:non-zero-exit"),
    ];
    for (input, expected) in cases {
        assert_eq!(fingerprint(input).signature, expected);
    }
}

#[test]
fn fingerprint_is_deterministic() {
    let input = "error[E0382]: borrow of moved value";
    assert_eq!(fingerprint(input), fingerprint(input));
}

#[test]
fn unknown_fingerprint_preserves_excerpt() {
    let fp = fingerprint("very strange failure");
    assert_eq!(fp.signature, "unknown");
    assert!(fp.raw_excerpt.contains("very strange failure"));
}
