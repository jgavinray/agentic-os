use tracing_subscriber::prelude::*;
use tracing_subscriber::EnvFilter;

pub struct LoggingGuard {
    #[cfg(feature = "tracing-otlp")]
    provider: Option<opentelemetry_sdk::trace::SdkTracerProvider>,
}

impl LoggingGuard {
    pub fn shutdown(self) {
        #[cfg(feature = "tracing-otlp")]
        if let Some(provider) = self.provider {
            if let Err(e) = provider.shutdown() {
                eprintln!("failed to shut down OTLP tracer provider: {e}");
            }
        }
    }
}

pub fn init_logging() -> Result<LoggingGuard, anyhow::Error> {
    let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"));
    let fmt_layer = tracing_subscriber::fmt::layer()
        .json()
        .with_file(true)
        .with_line_number(true)
        .with_thread_ids(true);

    #[cfg(feature = "tracing-otlp")]
    if std::env::var("OTEL_EXPORTER_OTLP_ENDPOINT")
        .ok()
        .filter(|v| !v.trim().is_empty())
        .is_some()
    {
        let provider = init_otlp_provider()?;
        let tracer = opentelemetry::trace::TracerProvider::tracer(&provider, "agentic-os");
        let otel_layer = tracing_opentelemetry::layer().with_tracer(tracer);
        tracing_subscriber::registry()
            .with(filter)
            .with(fmt_layer)
            .with(otel_layer)
            .init();
        tracing::info!("OTLP tracing enabled");
        return Ok(LoggingGuard {
            provider: Some(provider),
        });
    }

    tracing_subscriber::registry()
        .with(filter)
        .with(fmt_layer)
        .init();

    #[cfg(feature = "tracing-otlp")]
    let provider = None;

    #[cfg(not(feature = "tracing-otlp"))]
    if std::env::var("OTEL_EXPORTER_OTLP_ENDPOINT")
        .ok()
        .filter(|v| !v.trim().is_empty())
        .is_some()
    {
        tracing::warn!(
            "OTEL_EXPORTER_OTLP_ENDPOINT is set, but the binary was built without the tracing-otlp feature"
        );
    }

    Ok(LoggingGuard {
        #[cfg(feature = "tracing-otlp")]
        provider,
    })
}

#[cfg(feature = "tracing-otlp")]
fn init_otlp_provider() -> Result<opentelemetry_sdk::trace::SdkTracerProvider, anyhow::Error> {
    use opentelemetry::KeyValue;
    use opentelemetry_otlp::{Protocol, SpanExporter, WithExportConfig};
    use opentelemetry_sdk::trace::{Sampler, SdkTracerProvider};
    use opentelemetry_sdk::Resource;

    let endpoint = std::env::var("OTEL_EXPORTER_OTLP_ENDPOINT")?;
    let exporter = SpanExporter::builder()
        .with_tonic()
        .with_protocol(Protocol::Grpc)
        .with_endpoint(endpoint)
        .build()?;

    let provider = SdkTracerProvider::builder()
        .with_resource(
            Resource::builder()
                .with_attributes([KeyValue::new("service.name", "agentic-os")])
                .build(),
        )
        .with_sampler(Sampler::AlwaysOn)
        .with_batch_exporter(exporter)
        .build();

    Ok(provider)
}
