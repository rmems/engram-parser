# Release checklist — v0.3.0 (first crates.io publication)

Gate run on the release commit by Devin (RM-1796). Publication itself is done
locally by the maintainer after validating against a real model.

## Package-content gate

`package.include` allowlist in `Cargo.toml`:

| Path | Purpose |
| --- | --- |
| `Cargo.toml` / `Cargo.lock` | Manifest and resolved dependency set |
| `README.md` | crates.io/readme documentation |
| `CHANGELOG.md` | Release history |
| `LICENSE-MIT`, `LICENSE-APACHE-2.0` | Dual license texts |
| `src/**` | Crate source |
| `examples/**` | User-facing inspection examples (`inspect_gguf`, `inspect_checkpoint`, `checkpoint_smoke`) |

Excluded: `tests/` + `tests/fixtures/` (dev verification only), `docs/` (internal
notes/smoke data), `.github/` CI, `Dockerfile`, `REVIEW.md`, `RELEASE.md`,
`rust-toolchain.toml`, editor/agent tooling (`.agents`, `.cursor`, `.codacy.yml`,
`.yamllint`, `.dockerignore`, `.gitignore`).

`cargo package --list` final result: 41 files — the allowlist above plus
Cargo-managed metadata (`Cargo.toml.orig`, `.cargo_vcs_info.json`). Archive:
`engram-parser-0.3.0.crate`, 103.5 KiB compressed (423.6 KiB packaged). No
unexpected files.

## Lightweight audit gate

- `cargo audit` (RustSec db, 31 locked dependencies): **0 advisories**.
- `.crate` archive scan for secrets/private files (`PRIVATE KEY`, `api_key`,
  `secret`, `token`, `password`): clean — only the words "token"/"tokenizer" in
  docs and JSON-parser code. No dotfiles other than `.cargo_vcs_info.json`.
- Dependency licenses reviewed: `memmap2` MIT/Apache-2.0, `safetensors` MIT OR
  Apache-2.0, `serde`/`serde_json`/`syn`/`proc-macro2`/`quote`/`unicode-ident`/
  `itoa`/`memchr`/`cfg-if`/`once_cell`/`hashbrown`/`equivalent`/`allocator-api2`/
  `foldhash`/`bitflags`/`errno`/`libc`/`linux-raw-sys`/`rustix`/`getrandom`/
  `windows-sys`/`windows-link`/`tempfile`/`fastrand`/`zmij` — MIT or Apache-2.0
  (or both); `r-efi` MIT OR Apache-2.0 OR LGPL-2.1-or-later (permissive options
  satisfy distribution). All compatible.

## Publication gate run (all green)

`cargo fmt --check`, `cargo clippy --all-targets -- -D warnings` (default and
`--all-features`), `cargo test` in all four feature combos (default, `mmap`,
`safetensors`, `--all-features`), `RUSTDOCFLAGS="-D warnings" cargo doc --no-deps`
(default and `--all-features`), `cargo package --locked`,
`cargo publish --dry-run --locked`.

## Remaining (maintainer, local)

1. `cargo publish --locked` after the on-model test.
2. Verify `engram-parser 0.3.0` on crates.io + docs.rs build.
3. Dispatch the Release workflow with tag `v0.3.0` and the published commit SHA
   (never tag manually — the workflow creates tag + GitHub Release).
