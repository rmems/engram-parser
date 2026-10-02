# Release checklist — v0.3.0 (first crates.io publication)

Run the publication gate in [`RELEASE.md`](../RELEASE.md) on the final clean,
merged release commit. This document separates package policy from a dated
audit snapshot: earlier test results, archive sizes, and hashes do not qualify
a later commit. Publication remains a maintainer action under
[GitHub #89](https://github.com/rmems/engram-parser/issues/89) / RM-1786.

## Package-content policy

`package.include` in `Cargo.toml` allows only these publication paths:

| Path | Purpose |
| --- | --- |
| `Cargo.toml` / `Cargo.lock` | Manifest and resolved dependency set |
| `README.md` | Public installation and API documentation |
| `CHANGELOG.md` | Release history |
| `LICENSE-MIT`, `LICENSE-APACHE-2.0` | Dual license texts |
| `src/**` | Crate source, including in-module unit tests |
| `examples/**` | Public checkpoint inspection and smoke examples |

Integration tests and fixtures under `tests/`, internal reports under `docs/`,
CI, `REVIEW.md`, `RELEASE.md`, `rust-toolchain.toml`, and local agent/quality
configuration stay outside the archive. Cargo adds `Cargo.toml.orig` and,
when packaging a Git checkout, `.cargo_vcs_info.json`.

Inspect `cargo package --list` and the archive itself for each candidate.
Verify its VCS SHA matches the candidate and every file has an intended
purpose. The 11 warnings for deliberately excluded integration-test targets
are expected under the policy in `RELEASE.md`; other packaging warnings still
close the gate. A dry run's explicit upload-aborted notice is expected.

## Audit snapshot — 2026-10-02

Compile, test, rustdoc, and hosted-CI results below were measured on
[`485f0445a93c1ec06e8142a2d602c33030d753d7`](https://github.com/rmems/engram-parser/commit/485f0445a93c1ec06e8142a2d602c33030d753d7)
with Rust 1.99.0 / Cargo 1.99.0. The crate's declared MSRV is 1.97.1. This
documentation fix does not change crate source, so those results still describe
the code, but the hosted run URLs belong to `485f044`, not to this commit.
Archive measurements are recorded separately because the README change changes
the crate bytes.

| Check on `485f044` | Result |
| --- | --- |
| Formatting, default/all-feature Clippy with warnings denied, default/all-feature builds | Passed |
| Default-feature tests | 123 passed, 2 opt-in pilots ignored |
| All-feature tests | 227 passed, 3 opt-in pilots ignored |
| Release-mode all-feature tests | 227 passed, 3 opt-in pilots ignored |
| All-feature rustdoc with warnings denied | Passed |
| `cargo package --locked` and `cargo publish --dry-run --locked` | Passed; 11 expected test-exclusion warnings |
| Unpacked crate all-feature tests and rustdoc | 118 tests passed; rustdoc passed |
| Default normal dependency tree | Zero dependencies |
| Hosted OS matrix, MSRV, coverage, and quality | [Passed](https://github.com/rmems/engram-parser/actions/runs/37036534538) |
| Hosted RustSec audit | [Passed](https://github.com/rmems/engram-parser/actions/runs/37036534413) |

A clean `cargo package --locked` of the README change (Rust 1.99.0 / Cargo
1.99.0) produced **41 entries** and Cargo's reported **421.6 KiB packaged /
102.7 KiB compressed**. Do not treat a recorded SHA256 as acceptance evidence
for a later commit: Cargo embeds that commit in `.cargo_vcs_info.json`, so the
archive checksum changes even when this checklist — which is not packaged —
is the only edit. On each candidate, confirm the VCS SHA matches the commit
being packaged. Inspection of the README-changed archive found only
allowlisted/Cargo-managed paths, no symlinks, and no private-key headers.
This is a bounded archive inspection, not a claim that every possible secret
pattern was audited. The README includes the intended public Codecov badge
query token; registry upload credentials are not package content.

These archive measurements must be regenerated after changing the README,
manifest, source, or other packaged files. Do not compare a new candidate's
archive against this snapshot's checksum as an acceptance test.

### Locked dependency license inventory

The snapshot's `Cargo.lock` SHA256 is
`6d6856fd48826d21526a9073968b3f3a38d2866cd4386f4fd73f3422cf854cb4`.
It contains 31 package records: this crate plus **30 registry packages**.
The following SPDX expressions were read from the exact cached registry
package manifests. Regenerate the inventory if the lockfile changes, and
preserve the applicable license notices.

| SPDX expression | Locked packages |
| --- | --- |
| `MIT OR Apache-2.0` | allocator-api2 0.2.21; bitflags 2.13.2; cfg-if 1.0.5; errno 0.3.14; getrandom 0.4.3; hashbrown 0.16.1; itoa 1.0.18; libc 0.2.189; memmap2 0.9.11; once_cell 1.21.4; proc-macro2 1.0.107; quote 1.0.47; serde 1.0.229; serde_core 1.0.229; serde_derive 1.0.229; serde_json 1.0.151; syn 3.0.6; tempfile 3.27.0; windows-link 0.2.1; windows-sys 0.61.2 |
| `Apache-2.0 OR MIT` | equivalent 1.0.2; fastrand 2.5.0 |
| `Apache-2.0` | safetensors 0.8.0 |
| `Zlib` | foldhash 0.2.0 |
| `(MIT OR Apache-2.0) AND Unicode-3.0` | unicode-ident 1.0.26 |
| `Apache-2.0 WITH LLVM-exception OR Apache-2.0 OR MIT` | linux-raw-sys 0.12.1; rustix 1.1.5 |
| `Unlicense OR MIT` | memchr 2.8.3 |
| `MIT OR Apache-2.0 OR LGPL-2.1-or-later` | r-efi 6.0.0 |
| `MIT` | zmij 1.0.23 |

### Real-checkpoint evidence on the audited snapshot

Six fresh-process mmap smoke runs passed on `485f044`: first and warm runs
for each local fixture below. The arguments used 128 MiB resident-growth and
64 MiB Rust heap-peak budgets. All memory figures in this table are bytes.

| Fixture | Resident growth, first / warm | Rust heap peak, first / warm |
| --- | ---: | ---: |
| Ollama Granite 4.2 8B GGUF | 4,108,288 / 4,014,080 | 268,354 / 268,354 |
| Nemotron 3 Nano 4B single-file Safetensors | 1,462,272 / 1,474,560 | 523,687 / 523,687 |
| Granite 4.1 3B two-shard Safetensors | 1,941,504 / 2,007,040 | 515,795 / 515,795 |

The GGUF and Nemotron runs used the checked-in independent TSVs. Granite's
six sample expectations were independently derived from its index and shard
headers. Weight-file SHA256 values matched the Ollama blob identity or local
Hugging Face download metadata; the Granite index matched its Git blob SHA-1
ETag. These sampled metadata/byte checks do not establish inference or
numerical dequantization correctness. Cache state was uncontrolled.

The [smoke guide](checkpoint-smoke.md) describes the validation method and
fixture selection. The [Qwen report](smoke/2026-10-02-local-qwen-release-prep.md),
[Nemotron report](smoke/2026-10-02-local-nemotron.md), and
[Ollama Granite report](smoke/2026-10-02-local-ollama-granite.md) retain the
source identity and measurements of their earlier runs; they are not reports
for an eventual final release commit.

## Final publication checklist

1. Select the reviewed, clean, merged commit and rerun every gate in
   `RELEASE.md`. Record that SHA, toolchain versions, package file list, archive
   size/checksum, dependency-license inventory, and current model evidence in
   the release issue. Neither this snapshot nor a PR dry run substitutes for
   qualification of the published commit.
2. Have the maintainer confirm registry access and credential validity, then
   authorize and run `cargo publish --locked`. A dry run is not an upload.
3. Verify `engram-parser 0.3.0` on crates.io and the docs.rs all-feature build.
4. Dispatch the Release workflow with `v0.3.0` and the published SHA. The
   workflow creates the tag and GitHub Release; never tag manually in advance.
5. Update the existing [Linear v0.3.0 release](https://linear.app/rpd-34/pipeline/engram-parser/release/first-cratesio-checkpoint-substrate-cf986d0537be)
   with the published SHA, registry/docs/GitHub links, and verified release
   notes. Mark it Released only after external publication is confirmed, then
   reconcile GitHub #89 / RM-1786 and the release umbrella.
