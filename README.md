# engram-parser

[![CI](https://github.com/rmems/engram-parser/actions/workflows/ci.yml/badge.svg)](https://github.com/rmems/engram-parser/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/rmems/engram-parser/graph/badge.svg?token=zIX63gAh4q)](https://codecov.io/gh/rmems/engram-parser)
[![Maintainability](https://qlty.sh/gh/rmems/projects/engram-parser/maintainability.svg)](https://qlty.sh/gh/rmems/projects/engram-parser)
[![Rust 1.99.0](https://img.shields.io/badge/Rust-1.99.0-orange)](https://github.com/rmems/engram-parser/blob/main/rust-toolchain.toml)
[![MSRV 1.97.1](https://img.shields.io/badge/MSRV-1.97.1-blue)](https://github.com/rmems/engram-parser/blob/main/Cargo.toml)
[![License: MIT OR Apache-2.0](https://img.shields.io/badge/License-MIT%20OR%20Apache--2.0-blue.svg)](LICENSE-MIT)

Pure-Rust **GGUF + Safetensors checkpoint/tensor substrate** with optional format-specific helpers and model-analysis modules.

`engram-parser` ships a format-independent `Checkpoint` API over GGUF v3 and Safetensors: tensor inventory, metadata, and raw payload access without branching on container format. Format-specific surfaces remain for callers that need them: GGUF wire types and packed K-quant dequant (Q8_0 / Q5_K / Q6_K / IQ3_M block layout), an optional `mmap` reader, and Safetensors header/manifest/MoE discovery — the latter behind the off-by-default `safetensors` Cargo feature, which uses the upstream `safetensors` crate for canonical validation. MoE per-expert raw-weight extraction is a GGUF-specific model-analysis specialization under `engram_parser::analysis::moe`.

> **Safetensors status:** shipped behind `--features safetensors`. Default builds are GGUF-only. Raw payload reads are bounded per-tensor; enable `--features safetensors,mmap` for borrowed mmap-backed access. No Hugging Face `config.json` policy.
>
> **mmap / K-quant status:** shipped. Default `load_gguf` still uses `fs::read`. Enable `--features mmap` for `load_gguf_mmap` (`memmap2`). Packed dequant for Q8_0 / Q5_K / Q6_K / IQ3_M is on the default crate. CUDA host-register is out of scope.

## Install

v0.3.0 is not yet published to crates.io. Until publication completes, use a git dependency pinned to a commit or tag:

```toml
[dependencies]
engram-parser = { git = "https://github.com/rmems/engram-parser", rev = "<commit-or-tag>" }
```

Once published (see [Releases](#releases)), pick one — GGUF only, no dependencies enabled by default:

```toml
[dependencies]
engram-parser = "0.3"
```

with Safetensors support:

```toml
[dependencies]
engram-parser = { version = "0.3", features = ["safetensors"] }
```

or with Safetensors plus borrowed mmap payloads:

```toml
[dependencies]
engram-parser = { version = "0.3", features = ["safetensors", "mmap"] }
```

Equivalently: `cargo add engram-parser` (add `--features safetensors` / `mmap` as needed).

| Feature | Adds | Enables |
|---|---|---|
| *(default)* | — | GGUF parsing, `Checkpoint` API, MoE extraction, K-quant dequant |
| `mmap` | optional `memmap2` | `load_gguf_mmap`, `open_checkpoint_mmap`, page-aligned tensor slices |
| `safetensors` | optional upstream `safetensors` crate | Safetensors header/manifest/discovery, raw payload access |

## Quick start

Format-agnostic checkpoint inventory — the shared entry point, complete and runnable:

```rust
use engram_parser::{Checkpoint, open_checkpoint};

fn main() -> Result<(), engram_parser::ParserError> {
    // Works for a .gguf file, a .safetensors file, an HF shard index,
    // or a checkpoint directory (Safetensors input needs the feature).
    let ckpt = open_checkpoint("./model.gguf")?;
    println!("format = {:?}", ckpt.format());

    for t in ckpt.tensors() {
        println!("{} {} {:?} {}B", t.name, t.dtype, t.shape.dims(), t.byte_len);
    }

    // Read raw, undecoded payload bytes for a tensor discovered above.
    if let Some(t) = ckpt.tensors().first() {
        let raw = ckpt.tensor_bytes(&t.name)?; // Cow<[u8]>
        println!("{}: {} bytes", t.name, raw.len());
    }
    Ok(())
}
```

GGUF-specific access (when you need GGUF semantics — metadata helpers, `ggml_type`, dequant):

```rust
use engram_parser::{ggml_type_label, load_gguf};

fn main() -> Result<(), engram_parser::ParserError> {
    let layout = load_gguf("./model.gguf")?;
    println!("architecture = {}", layout.metadata.architecture());
    for (name, tensor) in &layout.tensors {
        println!("{} {:?} ({})", name, tensor.dims, ggml_type_label(tensor.ggml_type));
    }
    Ok(())
}
```

MoE expert analysis — a GGUF-specific specialization under `engram_parser::analysis::moe`:

```rust
use engram_parser::analysis::moe::{extract_expert, list_experts};
use engram_parser::load_gguf;

fn main() -> Result<(), engram_parser::ParserError> {
    let layout = load_gguf("./moe-model.gguf")?;

    for (block, expert) in list_experts(&layout) {
        let weights = extract_expert(&layout, block, expert)?;
        if let Some(gate) = &weights.gate {
            println!(
                "blk.{block}.expert{expert}.gate: dims={:?} dtype={:?} bytes={}",
                gate.dims,
                gate.dtype,
                gate.bytes.len()
            );
        }
    }
    Ok(())
}
```

**Legacy MoE imports still work.** `engram_parser::moe::{extract_expert, list_experts, MoeExpertWeights, RawTensor}` and the crate-root `engram_parser::{extract_expert, list_experts, MoeExpertWeights, RawTensor}` paths re-export the same implementation and keep compiling in v0.3; `engram_parser::analysis::moe` is the canonical documented path going forward.

Safetensors (requires `--features safetensors`):

```rust
use engram_parser::safetensors::{inspect_safetensors_checkpoint, open_safetensors_checkpoint};

fn main() -> Result<(), engram_parser::ParserError> {
    let manifest = inspect_safetensors_checkpoint("./model.safetensors")?;
    println!(
        "kind={} tensors={}",
        manifest.checkpoint.input_kind, manifest.checkpoint.tensor_count
    );
    for tensor in &manifest.tensors {
        println!("{} {:?} {}", tensor.name, tensor.shape, tensor.dtype);
    }

    let checkpoint = open_safetensors_checkpoint("./model.safetensors")?;
    // "a.weight" is a placeholder — use a name from the manifest above.
    let raw: Vec<u8> = checkpoint.tensor_bytes("a.weight")?; // bounded read
    println!("payload bytes = {}", raw.len());
    Ok(())
}
```

With `--features safetensors,mmap`, `open_safetensors_checkpoint_mmap` returns borrowed `&[u8]` slices instead of owned copies.

## What it does

### Shipped now — GGUF

- Parses GGUF v3 magic, header, KV metadata, and tensor directory into an in-memory `GgufLayout`.
- Applies a documented `ParseLimits` budget (KV/tensor counts, string sizes, array work, tensor rank, metadata bytes) before allocation or loops proportional to file-declared values. Defaults are generous; trusted callers can override via `load_gguf_with_limits` / `parse_bytes_with_limits` without weakening the default path.
- Enumerates MoE experts discovered in a checkpoint (model analysis under `engram_parser::analysis::moe`).
- Extracts the raw byte buffers for one expert's `gate`, `up`, and `down` projections.
- Supports stacked (`blk.{B}.ffn_{role}_exps.weight`) and per-expert (`blk.{B}.ffn_{role}.{E}.weight`) conventions.
- Models packed GGUF dtype sizes without pulling in a numerical runtime.
- Packed dequant to `f32` for **Q8_0**, **Q5_K**, **Q6_K**, and the internal **IQ3_M block** layout (`GGML_TYPE_IQ3_M_BLOCK = 0x4949334D`). Existing `dequantize_f16` is unchanged.
- Optional **`mmap` feature** (`memmap2`): `load_gguf_mmap` maps the file instead of `fs::read` into a `Vec<u8>`. Default builds still read the whole file; no dependencies are enabled by default.

### Shipped behind feature — Safetensors

The off-by-default `safetensors` feature owns reusable support for:

- Safetensors header deserialization (canonical validation via the upstream `safetensors` crate);
- deterministic tensor manifests;
- single-file, Hugging Face shard-index, and directory layouts;
- checkpoint-relative shard resolution with path-escape rejection;
- unique tensor ownership and missing-shard diagnostics;
- tensor name/dtype/shape/offset/shard metadata;
- MoE router/expert candidate discovery and grouping;
- raw tensor payload access by name (`open_safetensors_checkpoint`, bounded per-tensor reads);
- mmap-backed borrowed payload slices with `--features safetensors,mmap` (`open_safetensors_checkpoint_mmap`);
- metadata-only layout-family inference.

The `safetensors` feature adds the upstream `safetensors` crate as an optional dependency; upstream types (`SafeTensors`, `TensorView`, `Dtype`) never appear in the public API. Engram keeps checkpoint discovery, shard/path policy, duplicate-key detection, stricter contiguous-range validation, and engram-owned dtype/shape/shard types.

## What it does NOT do

- No neural-network math: no `matmul`, forward pass, routing execution, or softmax.
- No CUDA, GPU, SIMD, or runtime inference engine. mmap exposes page-aligned tensor byte ranges so a later GPU layer can host-register; this crate does not.
- No tokenization or SNN dynamics.
- No full model-family routing policy.
- No Safetensors payload execution/matmul, and no Hugging Face `config.json` interpretation: payload access is byte-level only, with no runtime or model-policy concerns pulled in.

## Supported GGUF dtypes

Layout-aware parsing (**packed byte sizes**; plus the K-quant dequant helpers below) supports:

- `F32`, `F16`, `BF16`, `F64`;
- `I8`–`I64`;
- `Q4_0`, `Q4_1`, `Q5_0`, `Q5_1`, `Q8_0`, `Q8_1`;
- K-quants `Q2_K`, `Q3_K`, `Q4_K`, `Q5_K`, `Q6_K`, `Q8_K`;
- IQ packed layouts including `IQ2_XXS`, `IQ2_XS`, `IQ2_S`, `IQ3_XXS`, `IQ3_S`, `IQ1_S`, `IQ1_M`, `IQ4_NL`, `IQ4_XS`;
- Internal `IQ3_M_BLOCK` (111 bytes / 256 values) — **not** wire type 31.

Remaining codes use `DType::Other(u32)`. Unknown packed layouts fail closed when element count cannot be converted to a modeled byte length.

Numeric helpers: `dequantize_f16`, `dequantize_q8_0`, `dequantize_q5_k`, `dequantize_q6_k`, `dequantize_iq3_m`. The parser's primary contract is still layout/raw bytes; these unpackers are optional CPU helpers, not a GGML compute path.

### GGUF wire types vs "GGML"

GGUF stores each tensor dtype as a `ggml_type` integer. `engram-parser` maps those codes to labels and packed byte sizes so payload and MoE slices remain in range. Packed dequant is implemented for Q8_0, Q5_K, Q6_K, and the internal IQ3_M **block** id — not a GGML runtime.

Historical wire type **31** is `Q4_0_4_4`, **not** the Hugging Face preset name "IQ3_M". HuggingFace IQ3_M is a mixed-quant preset; the 111-byte block decoder uses internal `GGML_TYPE_IQ3_M_BLOCK` (`0x4949334D`). Wire 31 still has no packed byte-size model (`DType::Other(31)`), so such a checkpoint is labeled correctly but rejected during layout parsing.

## Public API

Current GGUF surface includes:

- `load_gguf`, `parse_bytes` (and `*_with_limits` for an explicit `ParseLimits` policy);
- `#[cfg(feature = "mmap")] load_gguf_mmap` / `load_gguf_mmap_with_limits` → `GgufLayoutMmap` (page-aligned tensor slices via `tensor_page_aligned_bytes`);
- `GgufLayout`, `GgufMetadata`, `Tensor`, `DType`, `ParseLimits`;
- `dequantize_f16` (on `Tensor`), `dequantize_q8_0`, `dequantize_q5_k`, `dequantize_q6_k`, `dequantize_iq3_m`;
- `extract_expert`, `list_experts` (canonical: `engram_parser::analysis::moe`);
- `MoeExpertWeights`, `RawTensor` (also at `engram_parser::analysis::moe`);
- `ParserError`, `ParseLimitKind`, `HostSizeField`, `Result`;
- public `GGML_TYPE_*` / `GGUF_VALUE_TYPE_*` constants and the `ggml_type_label` label function.

Safetensors surface (`--features safetensors`) includes:

- `inspect_safetensors_checkpoint`, `write_safetensors_manifest`;
- `open_safetensors_checkpoint` → `SafetensorsCheckpoint` (`tensor`, `tensor_bytes`, `resolve_tensor_bytes`, `manifest`);
- `#[cfg(feature = "mmap")] open_safetensors_checkpoint_mmap` → `SafetensorsCheckpointMmap` (borrowed `&[u8]` payload slices);
- `SafetensorsManifest`, `SafetensorsCheckpointSource`, `SafetensorsTensorRecord`;
- `classify_tensor`, `discover_candidates`;
- `SafetensorsCandidateSummary`, `SafetensorsRouterCandidate`, `SafetensorsExpertGroup`;
- `dtype_size_bytes`;
- `ParserError::DuplicateTensorOwnership` and `ParserError::MissingShard` for shard-index diagnostics.

## Format-independent checkpoint API

`engram_parser::checkpoint` (re-exported at the crate root, default features) is one contract over GGUF and Safetensors, so code can inventory tensors, read metadata, and fetch raw payloads without branching on format.

- **Trait:** `Checkpoint` (object safe, `Send + Sync`): `source`, `format`, `tensors` (sorted by name), `tensor`, `metadata`, `tensor_bytes`.
- **Detection:** `open_checkpoint` treats a directory as Safetensors, a file with `GGUF` magic as GGUF (any extension), and `*.safetensors` / `*.safetensors.index.json` as Safetensors. Anything else returns `UnsupportedFormat`. Safetensors input without the feature returns `ParserError::FeatureDisabled`; no other backend is substituted. With `mmap`, `open_checkpoint_mmap` maps GGUF files and Safetensors shards.
- **Shape order:** `TensorShape::dims()` is outermost-first for both formats. GGUF dims (innermost-first on disk) are reversed; `native_dims()` / `native_order()` recover the on-disk order.
- **Dtypes:** scalar types normalize to shared `TensorDType` variants. GGML quantized layouts stay `TensorDType::GgufPacked { ggml_type }` with the exact wire code, and `TensorInfo::native_dtype` keeps the source label (`Q4_K`, `BF16`, …).
- **Payloads:** bytes are returned exactly as stored, with length equal to `byte_len`. They are borrowed for GGUF and all mmap backends, and owned (bounded per-tensor read) for plain Safetensors.
- **Location:** `source` is relative to `CheckpointSource::root`, `data_offset` is relative to the tensor-data section, and `file_offset` is absolute.
- **Metadata:** GGUF scalars map to `MetadataValue::{String, UInt, Int, F32, F64}`; GGUF arrays are not captured (same as `GgufMetadata`), and `general.alignment` is a layout field (`layout().alignment`), not a metadata entry. Safetensors metadata is the manifest's flattened string map.

| Backend | Wraps | Feature | Payload |
|---|---|---|---|
| `GgufBackend` | `GgufLayout` | default | borrowed |
| `GgufMmapBackend` | `GgufLayoutMmap` | `mmap` | borrowed |
| `SafetensorsBackend` | `safetensors::SafetensorsCheckpoint` | `safetensors` | owned |
| `SafetensorsMmapBackend` | `safetensors::SafetensorsCheckpointMmap` | `safetensors` + `mmap` | borrowed |

**Migration.** Existing APIs are unchanged. To go from `load_gguf` to the shared contract, use `GgufBackend::open(path)` (or `from_layout(layout)` / `from_bytes`). `layout()` / `into_layout()` still reach `list_experts`, `extract_expert`, `find_tensors_with_suffix`, and dequant — now canonically at `engram_parser::analysis::moe`, with `engram_parser::moe` and the crate-root symbols kept as re-export shims. `AnyCheckpoint::as_gguf()`, `as_safetensors()`, and the mmap variants return the format-specific handles. The Safetensors manifest and candidate-discovery APIs are unchanged.

Format-agnostic inventory example: `cargo run --example inspect_checkpoint -- <path>` (add `--features safetensors` for Safetensors, `--mmap` with the `mmap` feature).

## Development

This crate has no dependencies enabled by default. Enable `mmap` for `memmap2`. Enable `safetensors` for header/manifest/discovery plus raw payload access (upstream `safetensors` crate); combine with `mmap` for borrowed payload slices.

```bash
cargo fmt --check
cargo clippy --all-targets -- -D warnings
cargo build
cargo test
cargo clippy --all-targets --all-features -- -D warnings
cargo test --features mmap
cargo test --features safetensors
cargo build --all-features
cargo test --all-features

# Coverage (requires cargo-llvm-cov)
cargo llvm-cov --all-targets --all-features --locked --lcov --output-path lcov.info
```

`load_gguf` reads and retains the complete file. For multi-GB checkpoints use `cargo test --features mmap` / `load_gguf_mmap`. CI mmap tests include packed Q8_0/Q5_K/Q6_K/IQ3_M dequant from the mapping and a sparse 2 GiB file that is mapped without `fs::read`. Real on-disk GGUF pilots still require local files (`ENGRAM_GGUF`) and are `#[ignore]`. CUDA host-register is still out of scope.

```bash
ENGRAM_GGUF=~/.models/gguf/.../model.gguf ENGRAM_EXPECT_MOE=1 \
  cargo test --test real_gguf -- --ignored --nocapture

cargo run --example inspect_gguf -- ~/.models/gguf/.../model.gguf
```

For directory scans, `ENGRAM_GGUF_MAX`, and MoE extraction sample counts, see [the T1 pilot guidance in `REVIEW.md`](https://github.com/rmems/engram-parser/blob/main/REVIEW.md#t1--real-gguf-pilots-this-repo-cpu-only).

Opt-in large GGUF and multi-shard Safetensors mmap validation, fixture inputs,
memory bounds and reproducible release reports: [checkpoint smoke guide](https://github.com/rmems/engram-parser/blob/main/docs/checkpoint-smoke.md).

See [`REVIEW.md`](https://github.com/rmems/engram-parser/blob/main/REVIEW.md) for quality gates.

## CI

- GitHub Actions: `.github/workflows/ci.yml` runs default and all-feature
  tests separately on Linux, Windows, and macOS, plus a Rust 1.97.1 MSRV job.
- Coverage: Linux `cargo llvm-cov` produces a downloadable LCOV artifact and
  uploads to Codecov when the repository token is available.
- Quality: Qlty checks workflow syntax and reports maintainability; Rust
  formatting and Clippy remain the authoritative Rust checks. Coverage is
  reported only through Codecov.
- Security: `.github/workflows/security.yml`

Cursor Cloud Agents use the separate `.cursor/Dockerfile` selected by
`.cursor/environment.json` to prepare their coding environment. It is not a
published crate artifact or a Docker image build in GitHub Actions.

## Releases

Version 0.3.0 is the first crates.io publication target; 0.2.0 was an
unpublished development version. See [`RELEASE.md`](https://github.com/rmems/engram-parser/blob/main/RELEASE.md) for the
mandatory package-inspection, rustdoc, packaging, and publish-dry-run gate.
Tags and GitHub Releases are created only after that gate and publication
have succeeded.

## MSRV (Minimum Supported Rust Version)

**MSRV: Rust 1.97.1.**

- Declared through `rust-version` in `Cargo.toml`.
- Tested in CI alongside the pinned Rust 1.99.0 build toolchain.
- MSRV bumps require justification and are treated as compatibility-significant changes.

See [#14](https://github.com/rmems/engram-parser/issues/14).

## Background and provenance (optional)

*Not required for installing or using the crate — retained for attribution.*

`engram-parser` was extracted from an internal research codebase as the
canonical reusable implementation of its GGUF and Safetensors parsing
surfaces; the extraction is one-way and this crate has no upstream research
dependencies. Wire-type labels and the Safetensors manifest format were
validated against the original reference implementation.

Historical issue trackers for the modularization work:

- GGUF modularization: [engram-parser#7](https://github.com/rmems/engram-parser/issues/7)
- Safetensors promotion: [engram-parser#10](https://github.com/rmems/engram-parser/issues/10), README consistency: [#61](https://github.com/rmems/engram-parser/issues/61)
- mmap + K-quant tracking: [engram-parser#45](https://github.com/rmems/engram-parser/issues/45)

## Wiki

This repository intentionally keeps documentation in version-controlled files such as `README.md` and [`REVIEW.md`](https://github.com/rmems/engram-parser/blob/main/REVIEW.md) rather than a GitHub Wiki.

## License

Licensed under either of:

- Apache License, Version 2.0 ([LICENSE-APACHE-2.0](LICENSE-APACHE-2.0)); or
- MIT license ([LICENSE-MIT](LICENSE-MIT)).
