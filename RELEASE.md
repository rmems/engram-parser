# Release process

## First public release

`v0.3.0` is the first crates.io publication target for `engram-parser`.
`v0.2.0` was an unpublished development version: do not publish it, create a
tag for it, or create a GitHub Release for it.

## Publication gate

Start from a clean commit on `main` whose `Cargo.toml`, `Cargo.lock`, and
changelog all identify the intended version. Run `rustup check`, then record
`rustc --version` and `cargo --version` to confirm that the release gate uses
the latest stable Rust. Keep `rust-version = "1.97.1"` and the separate MSRV CI
job: the compiler used to publish need not raise the minimum for consumers.
Run every command below:

```bash
cargo fmt --check
cargo clippy --all-targets -- -D warnings
cargo build
cargo test
cargo clippy --all-targets --all-features -- -D warnings
cargo build --all-features
cargo test --all-features
RUSTDOCFLAGS='-D warnings' cargo doc --all-features --no-deps
cargo package --list
cargo package --locked
cargo publish --dry-run --locked
```

Inspect the complete output of `cargo package --list` before continuing. It
must contain the intended source, license, README, and changelog and must not
contain local artifacts, fixtures that are unsuitable for distribution, or
secrets. The package allowlist intentionally excludes `tests/`, so Cargo may
warn that the 11 integration-test targets in that directory are ignored when
packaging. Check that each such warning names an excluded `tests/*.rs` target;
these known exclusions are expected. Any other packaging warning, rustdoc
warning, or failed command closes the gate.

## v0.3.0 ordering

1. Merge the exact release commit only after CI's default-feature,
   all-feature, and MSRV jobs are green.
2. Run and review the complete publication gate above on that commit.
3. Publish with `cargo publish --locked`, then verify that version `0.3.0` is
   visible on crates.io and that its documentation builds successfully.
4. Only after steps 1–3 succeed, dispatch the Release workflow with `v0.3.0`
   and the published commit SHA. The workflow repeats the package gate before
   it creates and pushes the tag, then creates the GitHub Release.
5. Update the existing Linear `engram-parser` release for version `0.3.0` with
   the published SHA and links to crates.io, docs.rs, and the GitHub Release.
   Verify that the release umbrella and publication issues are attached, add
   release notes backed by those artifacts, and move it to `Released` only
   after the registry publication and GitHub Release are confirmed.

Never create or push the tag manually in advance. A tag or GitHub Release is
evidence of a completed publication, not a trigger for publication.
