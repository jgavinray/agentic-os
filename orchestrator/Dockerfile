FROM cgr.dev/chainguard/rust:latest-dev AS builder

WORKDIR /app
COPY Cargo.toml Cargo.lock ./
COPY src ./src
RUN cargo build --release

# Use same image as builder since binary is dynamically linked against glibc
FROM cgr.dev/chainguard/rust:latest-dev

WORKDIR /app
COPY --from=builder /app/target/release/orchestrator /usr/local/bin/orchestrator

EXPOSE 8088
ENTRYPOINT ["/usr/local/bin/orchestrator"]
