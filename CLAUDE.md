@AGENTS.md

# CLAUDE.md

Claude-specific notes only. Layout, toolchain, feature list, CI job list, and commit conventions are in
`AGENTS.md` (imported above); do not duplicate them here.

## Commands

Run from the repo root. "Verified" means it ran successfully in a fresh checkout on Rust 1.99.0 (the pin in
`rust-toolchain.toml`) or 1.97.1 (MSRV), Linux, CPU only. `Cargo.lock` is tracked, so `--locked` is correct.

| Command | Status |
|---|---|
| `cargo fmt --check` | verified (1.99.0 and 1.97.1) |
| `cargo clippy --all-targets --locked -- -D warnings` | verified |
| `cargo clippy --all-targets --all-features --locked -- -D warnings` | verified (1.99.0 and 1.97.1) |
| `cargo build --locked` | verified |
| `cargo build --all-features --locked` | verified (1.99.0 and 1.97.1) |
| `cargo test --locked` | verified |
| `cargo test --all-features --locked` | verified (1.99.0 and 1.97.1) |
| `RUSTDOCFLAGS='-D warnings' cargo doc --all-features --no-deps` | verified (from `RELEASE.md`) |
| `cargo llvm-cov --all-targets --all-features --locked --lcov --output-path lcov.info` | verified (needs `cargo install cargo-llvm-cov --locked`; `lcov.info` is gitignored) |
| `ENGRAM_GGUF=/path/to/model.gguf cargo test --test real_gguf -- --ignored --nocapture` | not run (needs multi-GB local GGUF weights) |
| `qlty check --all --no-formatters --summary` | not run (CI-only Qlty CLI; Codecov upload also needs a secret) |
| `cargo package --locked`, `cargo publish --dry-run --locked` | not run here (release gate in `RELEASE.md`; the maintainer runs it at release time, not an agent session) |

CI runs the test matrix on Linux, Windows, and macOS and then fails if the working tree is dirty, so do not
commit generated files. Run fmt and clippy before committing.

## Architecture map

- `src/lib.rs`: crate root. `error.rs` holds the shared error and limit types.
- `src/gguf/`: GGUF v3 parsing (`cursor.rs`, `layout.rs`, `tensor.rs`), `limits.rs` (`ParseLimits`), `map.rs` (mmap), `dequant.rs`.
- `src/safetensors/`: manifest, discovery, payload access (feature `safetensors`).
- `src/checkpoint/`: format-independent `Checkpoint` API over GGUF and Safetensors (`open.rs` picks the format).
- `src/analysis/moe/` (`expert.rs`, `extract.rs`): MoE expert discovery and raw extraction. `src/moe/` is a legacy re-export shim of it; new code uses `analysis::moe`.
- `examples/`: `inspect_gguf`, `inspect_checkpoint`, `checkpoint_smoke` (requires `mmap` and `safetensors`).
- `tests/`: integration tests; `tests/fixtures/` holds goldens; `real_gguf.rs` is `#[ignore]` and path-gated by `ENGRAM_*` env vars.
- `REVIEW.md` is the pre-merge gate; `RELEASE.md` is the publication gate; `docs/` holds smoke-run notes.

## Invariants and non-goals

- Default build has zero dependencies. A new dependency must be `optional = true` behind a feature.
- No CUDA, GGML compute, or `myelin-accelerator` dependency, optional or not (`REVIEW.md`). GPU work belongs to other repos.
- Safetensors stays behind the off-by-default `safetensors` feature; `mmap` likewise.
- Parsing must stay bounded by `ParseLimits` and return typed errors, never hang on FIFOs, sockets, or devices (tests cover this).
- GGUF wire `ggml_type` 31 is `Q4_0_4_4`, not IQ3_M (`REVIEW.md`).
- `Cargo.toml` `include` is an allowlist (manifest, `Cargo.lock`, README, changelog, licenses, `src/**`, `examples/**`). `AGENTS.md` is not listed, so `CLAUDE.md` is not either; no manifest change was needed.
- Every `.rs` file starts with `// SPDX-License-Identifier: MIT OR Apache-2.0` (all current files do). Keep it on new files.

## Gotchas

- Never run `cargo publish` or create release tags from an agent session; releases follow `RELEASE.md` and are done by the maintainer.
- The MSRV toolchain needs its own components: `rustup component add --toolchain 1.97.1 rustfmt clippy`.
- `.claude/` is gitignored, so keep shared agent guidance in this file or `AGENTS.md`.
- Edit only what the task needs.
