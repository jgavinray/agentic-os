use deadpool_postgres::Pool;
use metrics::{counter, gauge};

pub fn record_pool_gauges(pool: &Pool) {
    let status = pool.status();
    gauge!("db_pool_size").set(status.max_size as f64);
    gauge!("db_pool_available").set(status.available as f64);
    gauge!("db_pool_waiters").set(status.waiting as f64);
}

pub fn record_process_metrics() {
    let metrics = process_metrics();
    counter!("process_cpu_seconds_total").absolute(metrics.cpu_seconds.floor() as u64);
    gauge!("process_resident_memory_bytes").set(metrics.resident_memory_bytes as f64);
    gauge!("process_start_time_seconds").set(metrics.start_time_seconds);
    gauge!("process_open_fds").set(metrics.open_fds as f64);
}

#[derive(Default)]
struct ProcessMetrics {
    cpu_seconds: f64,
    resident_memory_bytes: u64,
    start_time_seconds: f64,
    open_fds: u64,
}

fn process_metrics() -> ProcessMetrics {
    #[cfg(target_os = "linux")]
    {
        linux_process_metrics()
    }
    #[cfg(not(target_os = "linux"))]
    {
        ProcessMetrics {
            start_time_seconds: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_secs_f64())
                .unwrap_or_default(),
            ..Default::default()
        }
    }
}

#[cfg(target_os = "linux")]
fn linux_process_metrics() -> ProcessMetrics {
    const CLOCK_TICKS_PER_SECOND: f64 = 100.0;
    const PAGE_SIZE_BYTES: u64 = 4096;

    let stat = std::fs::read_to_string("/proc/self/stat").unwrap_or_default();
    let after_comm = stat.rsplit_once(") ").map(|(_, rest)| rest).unwrap_or("");
    let fields: Vec<&str> = after_comm.split_whitespace().collect();
    let utime = fields
        .get(11)
        .and_then(|v| v.parse::<f64>().ok())
        .unwrap_or_default();
    let stime = fields
        .get(12)
        .and_then(|v| v.parse::<f64>().ok())
        .unwrap_or_default();
    let start_ticks = fields
        .get(19)
        .and_then(|v| v.parse::<f64>().ok())
        .unwrap_or_default();
    let cpu_seconds = (utime + stime) / CLOCK_TICKS_PER_SECOND;

    let resident_memory_bytes = std::fs::read_to_string("/proc/self/statm")
        .ok()
        .and_then(|s| {
            s.split_whitespace()
                .nth(1)
                .and_then(|pages| pages.parse::<u64>().ok())
        })
        .map(|pages| pages * PAGE_SIZE_BYTES)
        .unwrap_or_default();

    let boot_time = std::fs::read_to_string("/proc/stat")
        .ok()
        .and_then(|s| {
            s.lines().find_map(|line| {
                line.strip_prefix("btime ")
                    .and_then(|value| value.parse::<f64>().ok())
            })
        })
        .unwrap_or_default();

    let open_fds = std::fs::read_dir("/proc/self/fd")
        .map(|entries| entries.count() as u64)
        .unwrap_or_default();

    ProcessMetrics {
        cpu_seconds,
        resident_memory_bytes,
        start_time_seconds: boot_time + start_ticks / CLOCK_TICKS_PER_SECOND,
        open_fds,
    }
}
