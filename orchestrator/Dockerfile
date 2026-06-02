FROM rust:latest AS builder

WORKDIR /app
COPY Cargo.toml Cargo.lock ./
COPY src ./src
COPY migrations ./migrations
RUN cargo build --release && \
    mkdir -p /app/libs && \
    find /app/target/release -maxdepth 1 -name 'libonnxruntime*' -exec cp {} /app/libs/ \;

FROM debian:trixie-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates curl libssl3 \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -r -s /bin/false orchestrator

WORKDIR /app
COPY --from=builder /app/target/release/orchestrator /usr/local/bin/orchestrator
COPY --from=builder /app/target/release/orchestrator-maint /usr/local/bin/orchestrator-maint
COPY --from=builder /app/libs/ /usr/local/lib/

ENV LD_LIBRARY_PATH=/usr/local/lib

USER orchestrator
EXPOSE 8088
ENTRYPOINT ["/usr/local/bin/orchestrator"]
