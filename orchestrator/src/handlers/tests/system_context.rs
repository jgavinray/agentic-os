use super::*;
use serde_json::json;

#[test]
fn inject_inserts_system_message_when_none_exists() {
    let mut payload = json!({
        "messages": [{"role": "user", "content": "hello"}]
    });
    inject_system_context(&mut payload, "prior context");
    let msgs = payload["messages"].as_array().unwrap();
    assert_eq!(msgs.len(), 2);
    assert_eq!(msgs[0]["role"], "system");
    assert_eq!(msgs[0]["content"], "prior context");
    assert_eq!(msgs[1]["role"], "user");
}

#[test]
fn inject_appends_to_existing_system_message_not_duplicates_it() {
    let mut payload = json!({
        "messages": [
            {"role": "system", "content": "base prompt"},
            {"role": "user", "content": "hello"}
        ]
    });
    inject_system_context(&mut payload, "prior context");
    let msgs = payload["messages"].as_array().unwrap();
    assert_eq!(msgs.len(), 2, "must not insert a second system message");
    let content = msgs[0]["content"].as_str().unwrap();
    assert!(content.contains("base prompt"));
    assert!(content.contains("prior context"));
}

#[test]
fn inject_keeps_system_message_at_index_zero() {
    let mut payload = json!({
        "messages": [
            {"role": "system", "content": "base prompt"},
            {"role": "user", "content": "hello"}
        ]
    });
    inject_system_context(&mut payload, "context");
    let msgs = payload["messages"].as_array().unwrap();
    assert_eq!(msgs[0]["role"], "system");
}

#[test]
fn inject_is_noop_when_messages_missing() {
    let mut payload = json!({"model": "gpt-4"});
    inject_system_context(&mut payload, "context");
    assert!(payload.get("messages").is_none());
}

#[test]
fn anthropic_inject_sets_system_when_absent() {
    let mut payload = json!({"messages": [{"role": "user", "content": "hi"}]});
    inject_system_context_anthropic(&mut payload, "ctx");
    assert_eq!(
        payload["system"],
        json!([{
            "type": "text",
            "text": "ctx"
        }])
    );
}

#[test]
fn anthropic_inject_appends_to_string_system() {
    let mut payload = json!({"system": "base", "messages": []});
    inject_system_context_anthropic(&mut payload, "ctx");
    let sys = payload["system"].as_array().unwrap();
    assert_eq!(sys[0], json!({"type": "text", "text": "base"}));
    assert_eq!(
        sys[1],
        json!({
            "type": "text",
            "text": "ctx"
        })
    );
}

#[test]
fn anthropic_inject_flattens_array_system_and_appends() {
    let mut payload = json!({
        "system": [{"type": "text", "text": "part1"}, {"type": "text", "text": "part2"}],
        "messages": []
    });
    inject_system_context_anthropic(&mut payload, "ctx");
    let sys = payload["system"].as_array().unwrap();
    assert_eq!(sys[0], json!({"type": "text", "text": "part1"}));
    assert_eq!(sys[1], json!({"type": "text", "text": "part2"}));
    assert_eq!(
        sys[2],
        json!({
            "type": "text",
            "text": "ctx"
        })
    );
}

#[test]
fn anthropic_inject_ignores_non_text_system_type() {
    let mut payload = json!({"system": 42, "messages": []});
    inject_system_context_anthropic(&mut payload, "ctx");
    assert_eq!(
        payload["system"],
        json!([{
            "type": "text",
            "text": "ctx"
        }])
    );
}

#[test]
fn anthropic_inject_caches_stable_prefix_not_dynamic_tail() {
    let mut payload = json!({"messages": []});
    let context =
        "== Stable Context Artifacts ==\n[repo:service_topology:active]\nstable\n\nRepository: repo\nTask: task\n";
    inject_system_context_anthropic(&mut payload, context);
    let sys = payload["system"].as_array().unwrap();
    assert_eq!(sys.len(), 2);
    assert_eq!(
        sys[0]["cache_control"],
        json!({"type": "ephemeral"}),
        "stable compiler prefix should be provider-cacheable"
    );
    assert!(sys[0]["text"]
        .as_str()
        .unwrap()
        .contains("Stable Context Artifacts"));
    assert_eq!(
        sys[1].get("cache_control"),
        None,
        "dynamic repository/task tail must not become a provider cache breakpoint"
    );
    assert!(sys[1]["text"]
        .as_str()
        .unwrap()
        .starts_with("Repository: repo"));
}
