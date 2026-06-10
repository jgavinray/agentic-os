pub(crate) fn contains_any(value: &str, needles: &[&str]) -> bool {
    needles.iter().any(|needle| value.contains(needle))
}

/// Check whether any needle appears as a whole word in the value.
///
/// A word boundary is defined as a non-alphanumeric character or the
/// start/end of the string. This prevents substring false positives
/// (e.g., "tax" inside "taxonomy", "stock" inside "restock").
pub(crate) fn contains_word(value: &str, needles: &[&str]) -> bool {
    let lower = value.to_lowercase();
    needles.iter().any(|needle| {
        let needle = needle.to_lowercase();
        if needle.is_empty() {
            return false;
        }
        let mut search_start = 0usize;
        while let Some(relative_pos) = lower[search_start..].find(&needle) {
            let pos = search_start + relative_pos;
            let before_ok = pos == 0 || is_clean_word_boundary(lower.as_bytes()[pos - 1]);
            let after_pos = pos + needle.len();
            let after_ok =
                after_pos >= lower.len() || is_clean_word_boundary(lower.as_bytes()[after_pos]);
            if before_ok && after_ok {
                return true;
            }
            search_start = after_pos;
        }
        false
    })
}

fn is_clean_word_boundary(byte: u8) -> bool {
    !byte.is_ascii_alphanumeric() && !matches!(byte, b'_' | b'-' | b'.' | b'/')
}

pub(crate) fn push_if<T: PartialEq + Copy>(items: &mut Vec<T>, condition: bool, item: T) {
    if condition && !items.contains(&item) {
        items.push(item);
    }
}
