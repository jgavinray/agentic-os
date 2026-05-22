//! Enum variant inventory for schema-version bump detection.
//!
//! This file holds the checked-in DefaultHasher fingerprint of the enum variant
//! inventory. When enum variants change (including renaming or reordering),
//! the developer must bump `FEATURE_SCHEMA_VERSION` and update this constant
//! with the newly computed hash.

/// Computed hash of the canonical enum variant inventory.
///
/// Updated by running `cargo test enum_version_bump_test` and copying
/// the printed hash value into this constant.
pub const ENUM_VARIANT_HASH: &str = "4abe6dfaf97d1e7e";
