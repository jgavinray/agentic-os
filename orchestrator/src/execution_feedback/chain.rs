use crate::db::AgentEvent;
use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet};
use uuid::Uuid;

#[allow(dead_code)]
pub fn patch_validation_ids_resolve(chain: &[AgentEvent], patch_event: &AgentEvent) -> bool {
    let ids: BTreeSet<&str> = chain.iter().map(|event| event.id.as_str()).collect();
    patch_event
        .metadata
        .get("payload")
        .and_then(|p| p.get("validation_event_ids"))
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(Value::as_str)
        .all(|id| ids.contains(id))
}

pub fn group_by_parent(events: &[AgentEvent]) -> BTreeMap<Option<Uuid>, Vec<AgentEvent>> {
    let mut grouped: BTreeMap<Option<Uuid>, Vec<AgentEvent>> = BTreeMap::new();
    for event in events {
        grouped
            .entry(event.parent_event_id)
            .or_default()
            .push(event.clone());
    }
    for siblings in grouped.values_mut() {
        siblings.sort_by(|a, b| {
            a.created_at
                .cmp(&b.created_at)
                .then_with(|| a.id.cmp(&b.id))
        });
    }
    grouped
}
