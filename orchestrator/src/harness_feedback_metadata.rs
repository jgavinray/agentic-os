use serde_json::{json, Value};

pub(crate) fn object_or_empty(value: Value) -> Value {
    if value.is_object() {
        value
    } else {
        json!({})
    }
}

pub(crate) fn contains_any(haystack: &str, needles: &[&str]) -> bool {
    needles.iter().any(|needle| haystack.contains(needle))
}

pub(crate) fn is_poisoned_context_ledger_path(path: &str) -> bool {
    let path = path.to_ascii_lowercase();
    contains_any(
        &path,
        &["context_leder", "context_ledler", "context-ledger"],
    )
}

pub(crate) fn bool_path(value: &Value, path: &[&str]) -> bool {
    path_value(value, path)
        .and_then(Value::as_bool)
        .unwrap_or(false)
}

pub(crate) fn string_path<'a>(value: &'a Value, path: &[&str]) -> Option<&'a str> {
    path_value(value, path).and_then(Value::as_str)
}

pub(crate) fn string_array_path<'a>(value: &'a Value, path: &[&str]) -> Vec<&'a str> {
    path_value(value, path)
        .and_then(Value::as_array)
        .map(|values| values.iter().filter_map(Value::as_str).collect())
        .unwrap_or_default()
}

pub(crate) fn changed_file_count(metadata: &Value) -> Option<usize> {
    usize_path(metadata, &["harness", "files_changed"])
        .or_else(|| usize_path(metadata, &["harness_feedback", "files_changed"]))
        .or_else(|| array_len_path(metadata, &["harness", "changed_files"]))
        .or_else(|| array_len_path(metadata, &["harness_feedback", "changed_files"]))
}

fn usize_path(value: &Value, path: &[&str]) -> Option<usize> {
    path_value(value, path)
        .and_then(Value::as_u64)
        .and_then(|value| usize::try_from(value).ok())
}

fn array_len_path(value: &Value, path: &[&str]) -> Option<usize> {
    path_value(value, path)
        .and_then(Value::as_array)
        .map(Vec::len)
}

fn path_value<'a>(value: &'a Value, path: &[&str]) -> Option<&'a Value> {
    let mut current = value;
    for segment in path {
        current = current.get(*segment)?;
    }
    Some(current)
}
