use super::*;
use std::hash::{Hash, Hasher};

#[test]
fn enum_version_bump_test() {
    let hash_input: String = enums::enum_inventory()
        .iter()
        .map(|(name, variants)| format!("{}:{}", name, variants.join(",")))
        .collect::<Vec<_>>()
        .join("\n");
    let checked_in = enum_hash::ENUM_VARIANT_HASH;

    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    hash_input.hash(&mut hasher);
    let computed_hash = hasher.finish();

    if checked_in != format!("{computed_hash:x}") {
        panic!(
            "Enum variants changed - bump FEATURE_SCHEMA_VERSION and update context_ledger::enum_hash::ENUM_VARIANT_HASH with the new hash.\nComputed: {computed_hash:x}\nCheckin:  {checked_in}\nEnums:    {hash_input}"
        );
    }
}
