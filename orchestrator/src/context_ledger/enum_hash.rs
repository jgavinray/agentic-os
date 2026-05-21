//! Enum variant inventory for schema-version bump detection.
//!
//! This file holds the checked-in SHA-256 hash of the canonical enum variant
//! inventory. When enum variants change (including renaming or reordering),
//! the developer must bump `FEATURE_SCHEMA_VERSION` and update this constant
//! with the newly computed hash.

/// Computed SHA-256 hash of the canonical enum variant inventory.
///
/// Updated by running `cargo test enum_version_bump_test` and copying
/// the printed hash value into this constant.
pub const ENUM_VARIANT_HASH: &str = "placeholder";
