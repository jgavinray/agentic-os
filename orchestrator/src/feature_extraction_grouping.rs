use crate::db::AgentEvent;
use chrono::{DateTime, Duration, TimeZone, Utc};
use std::collections::BTreeMap;
use uuid::Uuid;

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) enum FeatureGroupKey {
    Trajectory(Uuid),
    SessionWindow {
        repo: String,
        session_id: String,
        window_start: DateTime<Utc>,
        window_end: DateTime<Utc>,
    },
}

pub(crate) fn group_events_by_feature_window(
    events: &[AgentEvent],
    feature_window_sec: i64,
) -> BTreeMap<FeatureGroupKey, Vec<AgentEvent>> {
    let mut groups: BTreeMap<FeatureGroupKey, Vec<AgentEvent>> = BTreeMap::new();
    for event in events {
        groups
            .entry(group_key_for_event(event, feature_window_sec))
            .or_default()
            .push(event.clone());
    }

    for grouped_events in groups.values_mut() {
        grouped_events.sort_by(|a, b| {
            a.created_at
                .cmp(&b.created_at)
                .then_with(|| a.id.cmp(&b.id))
        });
    }
    groups
}

fn group_key_for_event(event: &AgentEvent, feature_window_sec: i64) -> FeatureGroupKey {
    if let Some(trajectory_id) = event.trajectory_id {
        FeatureGroupKey::Trajectory(trajectory_id)
    } else {
        let window_start = floor_to_window(event.created_at, feature_window_sec);
        let window_end = window_start + Duration::seconds(feature_window_sec);
        FeatureGroupKey::SessionWindow {
            repo: event.repo.clone(),
            session_id: event.session_id.clone(),
            window_start,
            window_end,
        }
    }
}

fn floor_to_window(ts: DateTime<Utc>, window_sec: i64) -> DateTime<Utc> {
    let seconds = ts.timestamp();
    let window_start = seconds - seconds.rem_euclid(window_sec);
    Utc.timestamp_opt(window_start, 0).single().unwrap_or(ts)
}
