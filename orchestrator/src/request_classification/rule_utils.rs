pub(crate) fn contains_any(value: &str, needles: &[&str]) -> bool {
    needles.iter().any(|needle| value.contains(needle))
}

pub(crate) fn push_if<T: PartialEq + Copy>(items: &mut Vec<T>, condition: bool, item: T) {
    if condition && !items.contains(&item) {
        items.push(item);
    }
}
