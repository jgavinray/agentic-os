/// Version markers and bucket boundary constants.
///
/// Changing any bucket boundary or enum definition requires bumping
/// `FEATURE_SCHEMA_VERSION` (enforced by the enum-version-bump test).

use super::enums::*;

/// Schema version — bumped whenever bucket boundaries or enum
/// definitions change.
pub const FEATURE_SCHEMA_VERSION: i32 = 1;

/// Context policy version string.
pub const CONTEXT_POLICY_VERSION: &str = "1.0.0";

/// Token cost bucket boundaries (inclusive upper bounds).
///
/// Mapping from threshold array to bucket variant:
///
/// | Threshold array value | Bucket variant   | Range              |
/// |-----------------------|------------------|--------------------|
/// | 50                    | Bucket0_50       | 0–50               |
/// | 200                 | Bucket51_200     | 51–200             |
/// | 500                 | Bucket201_500    | 201–500            |
/// | 1000                | Bucket501_1000   | 501–1000           |
/// | 2000                | Bucket1001_2000  | 1001–2000          |
/// | anything above        | Bucket2001Plus   | 2001+              |
pub const TOKEN_COST_BUCKET_BOUNDARIES: [i32; 5] = [50, 200, 500, 1000, 2000];

/// Age bucket boundaries in seconds.
///
/// Mapping from threshold array to bucket variants:
///
/// | Threshold array value | Bucket variant | Range     |
/// |-----------------------|----------------|-----------|
/// | 60                  | <Bucket0_60s>  | <1m        |
/// | 600                 | <Bucket600s>   | <10m       |
/// | 3600                | <Bucket3600s>  | <1h        |
/// | 21600               | <Bucket21600s> | <6h        |
/// | 86400               | <Bucket86400s> | <24h       |
/// | 604800              | <Bucket604800s>| <7d        |
/// | anything above       | BucketOlder    | older      |
pub const AGE_BUCKET_BOUNDARIES_SECONDS: [i32; 6] =
    [60, 600, 3600, 21600, 86400, 604800];

/// Deterministic score bucket boundaries (inclusive upper bounds).
///
/// Mapping from threshold array to bucket variant:
///
/// | Threshold array value | Bucket variant | Range           |
/// |-----------------------|----------------|-----------------|
/// | 0.25                | VeryLow        | <0.25           |
/// | 0.40                | Low            | 0.25–0.40       |
/// | 0.60                | Medium         | 0.40–0.60       |
/// | 0.80                | High           | 0.60–0.80       |
/// | anything above       | VeryHigh       | >0.80           |
pub const DETERMINISTIC_SCORE_BUCKET_BOUNDARIES: [f64; 4] =
    [0.25, 0.40, 0.60, 0.80];

/// Request latency bucket boundaries in microseconds.
pub const LATENCY_BUCKET_BOUNDARIES_MICROS: [i64; 4] =
    [500_000, 1_000_000, 3_000_000, 10_000_000];

/// Input token bucket boundaries (inclusive upper bounds).
pub const INPUT_TOKEN_BUCKET_BOUNDARIES: [i32; 4] =
    [1000, 4000, 8000, 32000];

/// Output token bucket boundaries (inclusive upper bounds).
pub const OUTPUT_TOKEN_BUCKET_BOUNDARIES: [i32; 4] =
    [128, 512, 2048, 8192];

/// Outcome join window — 24 hours in microseconds.
pub const OUTCOME_JOIN_WINDOW_MICROS: i64 = 86_400_000_000;
