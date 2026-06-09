use serde_json::Value;

pub(crate) fn string_path<'a>(value: &'a Value, path: &[&str]) -> Option<&'a str> {
    let mut current = value;
    for key in path {
        current = current.get(*key)?;
    }
    current.as_str()
}

pub(crate) fn i64_path(value: &Value, path: &[&str]) -> Option<i64> {
    let mut current = value;
    for key in path {
        current = current.get(*key)?;
    }
    current
        .as_i64()
        .or_else(|| current.as_u64().and_then(|value| i64::try_from(value).ok()))
        .or_else(|| {
            current.as_f64().and_then(|value| {
                if value.is_finite() && value >= i64::MIN as f64 && value <= i64::MAX as f64 {
                    Some(value as i64)
                } else {
                    None
                }
            })
        })
        .or_else(|| current.as_str().and_then(|value| value.parse::<i64>().ok()))
}

pub(crate) fn bool_path(value: &Value, path: &[&str]) -> bool {
    bool_path_value(value, path).unwrap_or(false)
}

pub(crate) fn bool_path_value(value: &Value, path: &[&str]) -> Option<bool> {
    let mut current = value;
    for key in path {
        current = current.get(*key)?;
    }
    current.as_bool()
}

pub(crate) fn array_path_has_values(value: &Value, path: &[&str]) -> bool {
    let mut current = value;
    for key in path {
        let Some(next) = current.get(*key) else {
            return false;
        };
        current = next;
    }
    current
        .as_array()
        .map(|items| !items.is_empty())
        .unwrap_or(false)
}
