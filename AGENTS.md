# AGENTS.md

Guidance for coding agents (Amp, Codex, Cursor, Claude Code, and others) working in this repository.

## Purpose

`engram-parser` is a pure-Rust GGUF + Safetensors checkpoint/tensor substrate (see `README.md`). It
provides a format-independent `Checkpoint` API (tensor inventory, metadata, raw payload access)
over GGUF v3 and Safetensors, GGUF parsing with documented `ParseLimits`, packed dequant (Q8_0, Q5_K,
Q6_K, IQ3_M block), and MoE expert discovery/extraction under `engram_parser::analysis::moe`.
Per `REVIEW.md`, the default build has **zero dependencies**, and there is **no CUDA or GGML
compute** in this repo (CUDA host-register belongs to `myelin-accelerator`).

## Layout

| Path | Contents |
|------|----------|
| `src/gguf/` | GGUF header/KV/tensor-directory parsing, limits, dequant |
| `src/safetensors/` | Safetensors support (`safetensors` feature) |
| `src/checkpoint/` | Format-independent `Checkpoint` API |
| `src/analysis/` | Model analysis (MoE) |
| `tests/` (+ `tests/fixtures/`, `tests/common/`) | Integration tests (smoke, limits, mmap, parity matrix, public API surface) |
| `examples/` | Usage examples |

## Toolchain

- `rust-toolchain.toml`: channel `stable` with rustfmt, clippy, llvm-tools-preview.
- MSRV **1.97.1** (`rust-version`). The CI `msrv` job repeats fmt/clippy/build/test on 1.97.1.
- Features: `mmap` (`memmap2`), `safetensors` (upstream `safetensors` crate). Both off by default.
- No GPU or system packages needed.

## Commands (from `.github/workflows/ci.yml`)

```bash
# default features
cargo clippy --all-targets --locked -- -D warnings
cargo build --locked
cargo test --locked
# all features
cargo fmt --check
cargo clippy --all-targets --all-features -- -D warnings
cargo build --all-features
cargo test --all-features
# MSRV job (RUSTUP_TOOLCHAIN=1.97.1 overrides rust-toolchain.toml)
cargo +1.97.1 fmt --check
cargo +1.97.1 clippy --all-targets --all-features -- -D warnings
cargo +1.97.1 build --all-features
cargo +1.97.1 test --all-features
```

Coverage: `cargo llvm-cov --all-targets --all-features --locked --lcov --output-path lcov.info`.
The release/publication gate and its ordering are in `RELEASE.md` and `release.yml`
(`cargo package --locked`, `cargo publish --dry-run --locked`). Per `RELEASE.md`, never create or
push a release tag by hand.

## Conventions visible in the repo

- Keep the default build dependency-free. `[dependencies]` holds only optional crates (`memmap2`,
  `safetensors`) enabled by features, so any new dependency must be `optional = true` behind a feature.
- Every Rust source file has an SPDX license identifier header.
- `REVIEW.md` is the pre-merge quality gate. `CHANGELOG.md` is maintained.
- Commit subjects use Conventional Commits with scopes (`fix(gguf):`, `feat(checkpoint):`,
  `api:`), usually with Linear IDs (`RM-1792`) and the PR number.
