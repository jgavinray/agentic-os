use orchestrator::logging;
use std::env;

#[path = "orchestrator-maint/commands.rs"]
mod commands;
#[path = "orchestrator-maint/commands_report.rs"]
mod commands_report;
#[path = "orchestrator-maint/commands_runtime.rs"]
mod commands_runtime;
#[path = "orchestrator-maint/options.rs"]
mod options;
#[path = "orchestrator-maint/signature_backfill.rs"]
mod signature_backfill;

#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    let logging_guard = logging::init_logging()?;
    let result = run().await;
    logging_guard.shutdown();
    result
}

async fn run() -> Result<(), anyhow::Error> {
    let mut args = env::args().skip(1);
    let Some(command) = args.next() else {
        commands::print_usage();
        anyhow::bail!("missing command");
    };

    commands::run_command(&command, args.collect()).await
}
