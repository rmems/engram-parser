#!/usr/bin/env bash
# Idempotent Cloud Agent install: prefetch and compile only (no fmt/clippy/test execution).
set -euo pipefail

rustc --version | grep -F '1.99.0 ' \
  || {
    echo "install.sh requires Rust 1.99.0 (got $(rustc --version))" >&2
    exit 1
  }

cargo fetch --locked
cargo build --all-features --locked
cargo test --no-run --all-features --locked
