use metrics::{describe_counter, describe_gauge, describe_histogram};

pub(crate) fn describe_runtime_metrics() {
    describe_gauge!("db_pool_size", "Configured Postgres pool capacity.");
    describe_gauge!(
        "db_pool_available",
        "Postgres pool connections currently available."
    );
    describe_gauge!(
        "db_pool_waiters",
        "Tasks waiting for a Postgres pool connection."
    );
    describe_histogram!(
        "db_query_duration_seconds",
        "Postgres query latency in seconds."
    );
    describe_counter!("db_query_errors_total", "Postgres query failures.");
    describe_counter!("qdrant_requests_total", "Qdrant API requests.");
    describe_histogram!(
        "qdrant_request_duration_seconds",
        "Qdrant API latency in seconds."
    );
    describe_counter!(
        "process_cpu_seconds_total",
        "CPU seconds consumed by this process."
    );
    describe_gauge!(
        "process_resident_memory_bytes",
        "Resident memory used by this process."
    );
    describe_gauge!(
        "process_start_time_seconds",
        "Process start time since Unix epoch."
    );
    describe_gauge!(
        "process_open_fds",
        "Open file descriptors for this process."
    );
}
