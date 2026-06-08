use super::*;
use serde_json::json;

#[path = "handlers_policy_authorization_tests.rs"]
mod authorization_tests;

#[path = "handlers_policy_context_tests.rs"]
mod context_tests;

#[path = "handlers_policy_auth_header_tests.rs"]
mod auth_header_tests;
