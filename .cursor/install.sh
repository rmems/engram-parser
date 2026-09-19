#!/usr/bin/env bash
# Idempotent Cloud Agent install: match CI gates (fmt, clippy, test --all-features).
set -euo pipefail

rustc --version | grep -F '1.97.1 ' \
  || {
    echo "install.sh requires Rust 1.97.1 (got $(rustc --version))" >&2
    exit 1
  }

cargo fetch --locked
cargo fmt --check
cargo clippy --all-targets --all-features -- -D warnings
cargo test --all-features
