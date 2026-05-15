FROM cgr.dev/chainguard/rust:latest-dev AS builder

WORKDIR /app

COPY Cargo.toml Cargo.lock ./
COPY src ./src

RUN cargo build --release

FROM cgr.dev/chainguard/wolfi-base:latest

WORKDIR /app

COPY --from=builder /app/target/release/orchestrator /usr/local/bin/orchestrator

EXPOSE 8088

ENTRYPOINT ["/usr/local/bin/orchestrator"]
