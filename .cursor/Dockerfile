# SPDX-License-Identifier: MIT OR Apache-2.0
# Cursor Cloud Agent environment for engram-parser.
# Do not COPY the repository; Cursor checks out the target commit itself.
ARG RUST_VERSION=1.99.0

FROM rust:${RUST_VERSION}-slim

ENV DEBIAN_FRONTEND=noninteractive \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    CARGO_HOME=/usr/local/cargo \
    RUSTUP_HOME=/usr/local/rustup \
    PATH=/usr/local/cargo/bin:/usr/local/rustup/toolchains/${RUST_VERSION}-x86_64-unknown-linux-gnu/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        git \
        sudo \
    && rm -rf /var/lib/apt/lists/* \
    && rustup component add rustfmt clippy llvm-tools-preview

RUN groupadd --gid 1000 ubuntu \
    && useradd --uid 1000 --gid 1000 --create-home --shell /bin/bash ubuntu \
    && printf 'ubuntu ALL=(ALL) NOPASSWD:ALL\n' > /etc/sudoers.d/ubuntu \
    && chmod 0440 /etc/sudoers.d/ubuntu

RUN printf '%s\n' \
        'export CARGO_HOME="/usr/local/cargo"' \
        'export RUSTUP_HOME="/usr/local/rustup"' \
        'export PATH="/usr/local/cargo/bin:/usr/local/rustup/toolchains/1.99.0-x86_64-unknown-linux-gnu/bin:$PATH"' \
        > /etc/profile.d/cargo.sh \
    && chmod 644 /etc/profile.d/cargo.sh \
    && bash -c 'for f in /usr/local/cargo/bin/*; do ln -sf "$f" /usr/local/bin/; done' \
    && chown -R ubuntu:ubuntu /usr/local/cargo /usr/local/rustup

WORKDIR /workspace

USER ubuntu
