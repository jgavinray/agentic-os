use crate::db::AgentEvent;
use crate::trajectory_event_payload::{event_payload, event_success, payload_bool};
use crate::trajectory_types::{BoundaryReason, EventRole, FinalStatus};

pub fn derive_status(
    events: &[AgentEvent],
    boundary_reason: Option<BoundaryReason>,
) -> FinalStatus {
    if has_reverted_patch(events) {
        return FinalStatus::Reverted;
    }
    if has_successful_patch_with_latest_validation(events) {
        return FinalStatus::Succeeded;
    }
    match boundary_reason {
        Some(BoundaryReason::NewUserMessage) => FinalStatus::Abandoned,
        Some(BoundaryReason::IdleTimeout) | None => FinalStatus::Unresolved,
    }
}

fn has_reverted_patch(events: &[AgentEvent]) -> bool {
    let mut applied_seen = false;
    for event in events {
        if event.event_role.as_deref() != Some(EventRole::Patch.as_str()) {
            continue;
        }
        let payload = event_payload(event);
        if payload_bool(payload, "patch_applied").unwrap_or(false) {
            applied_seen = true;
        }
        if applied_seen && payload_bool(payload, "patch_reverted").unwrap_or(false) {
            return true;
        }
    }
    false
}

fn has_successful_patch_with_latest_validation(events: &[AgentEvent]) -> bool {
    let mut latest_patch_applied = false;
    for event in events {
        if event.event_role.as_deref() == Some(EventRole::Patch.as_str()) {
            let payload = event_payload(event);
            if payload_bool(payload, "patch_reverted").unwrap_or(false) {
                latest_patch_applied = false;
            } else if payload_bool(payload, "patch_applied").unwrap_or(false) {
                latest_patch_applied = true;
            }
        }
    }
    if !latest_patch_applied {
        return false;
    }
    events
        .iter()
        .rev()
        .find(|event| event.event_role.as_deref() == Some(EventRole::Validation.as_str()))
        .map(event_success)
        .unwrap_or(false)
}
