# Release process

## First public release

`v0.3.0` is the first crates.io publication target for `engram-parser`.
`v0.2.0` was an unpublished development version: do not publish it, create a
tag for it, or create a GitHub Release for it.

## Publication gate

Start from a clean commit on `main` whose `Cargo.toml`, `Cargo.lock`, and
changelog all identify the intended version. Run every command below:

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
secrets. Treat rustdoc warnings, packaging warnings, or any failed command as
a closed gate.

## v0.3.0 ordering

1. Merge the exact release commit only after CI's default-feature,
   all-feature, and MSRV jobs are green.
2. Run and review the complete publication gate above on that commit.
3. Publish with `cargo publish --locked`, then verify that version `0.3.0` is
   visible on crates.io and that its documentation builds successfully.
4. Only after steps 1–3 succeed, dispatch the Release workflow with `v0.3.0`
   and the published commit SHA. The workflow repeats the package gate before
   it creates and pushes the tag, then creates the GitHub Release.

Never create or push the tag manually in advance. A tag or GitHub Release is
evidence of a completed publication, not a trigger for publication.
