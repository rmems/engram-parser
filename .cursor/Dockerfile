# SPDX-License-Identifier: MIT OR Apache-2.0
# Cursor Cloud Agent environment for engram-parser.
# Do not COPY the repository; Cursor checks out the target commit itself.
ARG RUST_VERSION=1.97.1

FROM rust:${RUST_VERSION}-slim

ENV DEBIAN_FRONTEND=noninteractive \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8

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

WORKDIR /workspace

USER ubuntu
