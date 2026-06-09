pub(crate) fn describe_metrics() {
    crate::telemetry_descriptions_transport::describe_transport_metrics();
    crate::telemetry_descriptions_model_context::describe_model_context_metrics();
    crate::telemetry_descriptions_execution::describe_execution_metrics();
    crate::telemetry_descriptions_request::describe_request_metrics();
    crate::telemetry_descriptions_runtime::describe_runtime_metrics();
}
