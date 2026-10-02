# syntax=docker/dockerfile:1.4
#
# Dockerfile for engram-parser (pure-Rust, zero-dep GGUF/MoE parser).
# Single-stage build for CI verification and reproducible builds.
#
# engram-parser is a library crate with no binary target, so there is no
# standalone artifact to deploy to a runtime image. This image is used for:
#   - CI build/test verification
#   - Reproducible build environment
#   - Base image for downstream crates that depend on engram-parser
#
# Usage:
#   docker build -t engram-parser .
#   docker run --rm engram-parser cargo test --all-features
#
# See .github/workflows/docker-build.yml and issue #9 for CI (GHCR on main).

ARG RUST_VERSION=1.99.0

FROM rust:${RUST_VERSION}-slim

ARG RUST_VERSION

RUN useradd -m -u 10001 appuser

WORKDIR /app

# Copy manifests and lock file for reproducibility
COPY Cargo.toml Cargo.lock ./

# Copy source
COPY . .

# Build and test the crate with the pinned toolchain.
# RUSTUP_TOOLCHAIN is scoped to this RUN so it does not leak into the final image.
RUN export RUSTUP_TOOLCHAIN=${RUST_VERSION} && \
    rustc --version && cargo --version && \
    cargo build --release --all-features && \
    cargo test --release --all-features

RUN chown -R appuser:appuser /app

USER appuser

CMD ["cargo", "test", "--release", "--all-features"]
