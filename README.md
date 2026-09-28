# engram-parser

[![CI](https://github.com/rmems/engram-parser/actions/workflows/ci.yml/badge.svg)](https://github.com/rmems/engram-parser/actions/workflows/ci.yml)
[![License: MIT OR Apache-2.0](https://img.shields.io/badge/License-MIT%20OR%20Apache--2.0-blue.svg)](LICENSE-MIT)

Pure-Rust **checkpoint metadata parsing and Mixture-of-Experts extraction**.

Today, `engram-parser` ships GGUF v3 deserialization, per-expert raw-weight extraction, packed K-quant dequant (Q8_0 / Q5_K / Q6_K / IQ3_M block layout), and an optional `mmap` reader. Safetensors support — header/manifest/MoE discovery plus raw tensor payload access — ships behind the off-by-default `safetensors` Cargo feature, which uses the upstream `safetensors` crate for canonical validation.

> **Safetensors status:** shipped behind `--features safetensors`. Default builds remain GGUF-only. Raw payload reads are bounded per-tensor; enable `--features safetensors,mmap` for borrowed mmap-backed access. No Hugging Face `config.json` policy. Tracked in [#10](https://github.com/rmems/engram-parser/issues/10) and [#86](https://github.com/rmems/engram-parser/issues/86).
>
> **mmap / K-quant status:** shipped. Default `load_gguf` still uses `fs::read`. Enable `--features mmap` for `load_gguf_mmap` (`memmap2`). Packed dequant for Q8_0 / Q5_K / Q6_K / IQ3_M is on the default crate. Tracked in [#45](https://github.com/rmems/engram-parser/issues/45). CUDA host-register remains out of scope.

## What it does

### Shipped now — GGUF

- Parses GGUF v3 magic, header, KV metadata, and tensor directory into an in-memory [`GgufLayout`].
- Applies a documented [`ParseLimits`] budget (KV/tensor counts, string sizes, array work, tensor rank, metadata bytes) before allocation or loops proportional to file-declared values. Defaults are generous; trusted callers can override via `load_gguf_with_limits` / `parse_bytes_with_limits` without weakening the default path.
- Enumerates MoE experts discovered in a checkpoint.
- Extracts the raw byte buffers for one expert's `gate`, `up`, and `down` projections.
- Supports stacked (`blk.{B}.ffn_{role}_exps.weight`) and per-expert (`blk.{B}.ffn_{role}.{E}.weight`) conventions.
- Models packed GGUF dtype sizes without pulling in a numerical runtime.
- Packed dequant to `f32` for **Q8_0**, **Q5_K**, **Q6_K**, and the internal **IQ3_M block** layout (`GGML_TYPE_IQ3_M_BLOCK = 0x4949334D`). Existing `dequantize_f16` is unchanged.
- Optional **`mmap` feature** (`memmap2`): `load_gguf_mmap` maps the file instead of `fs::read` into a `Vec<u8>`. Default builds still read the whole file and keep `[dependencies]` empty. CUDA host-register remains out of scope (myelin-accelerator).

Tracking for mmap + K-quant: [engram-parser#45](https://github.com/rmems/engram-parser/issues/45) (option 1), [corinth-canal#144](https://github.com/rmems/corinth-canal/issues/144).

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
- metadata-only layout-family inference where it is reusable outside Corinth.

The `safetensors` feature adds the upstream `safetensors` crate as an optional dependency; upstream types (`SafeTensors`, `TensorView`, `Dtype`) never appear in the public API. Engram keeps checkpoint discovery, shard/path policy, duplicate-key detection, stricter contiguous-range validation, and engram-owned dtype/shape/shard types.

Tracking: [engram-parser#10](https://github.com/rmems/engram-parser/issues/10), [engram-parser#61](https://github.com/rmems/engram-parser/issues/61), and [corinth-canal#116](https://github.com/rmems/corinth-canal/issues/116).

## What it does NOT do

- No neural-network math: no `matmul`, forward pass, routing execution, or softmax.
- No CUDA, GPU, SIMD, or runtime inference engine. mmap exposes page-aligned tensor byte ranges so a later GPU crate can host-register; this crate does not.
- No tokenization or SNN dynamics.
- No full model-family routing policy; see [`cortex-tensor`](https://github.com/rmems/cortex-tensor).
- No Safetensors payload execution/matmul.
- No Corinth experiment orchestration or SAAQ calibration.

Safetensors payload **execution/matmul** and Hugging Face `config.json` interpretation remain outside this crate: payload access is byte-level only, with no runtime or model-policy concerns pulled in.

## Scope / boundaries

This crate **owns**:

- GGUF v3 deserialization: header, KV metadata, tensor directory.
- MoE expert enumeration and per-expert raw-weight extraction.
- Zero-dependency-by-default, layout-aware dtype handling; optional `mmap`.
- Packed dequant for Q8_0 / Q5_K / Q6_K / IQ3_M block layout.
- Feature-gated Safetensors header parsing, manifests, MoE candidate discovery, and raw tensor payload access (`--features safetensors`; upstream `safetensors` crate).

This crate **does not own**:

- tensor/Transformer/MoE numerical execution → [`cortex-tensor`](https://github.com/rmems/cortex-tensor);
- Transformer↔SNN orchestration/contracts → [`hybrid-fusion`](https://github.com/rmems/hybrid-fusion);
- neuron/network dynamics → [`neuromod`](https://github.com/Limen-Neural/neuromod);
- CUDA acceleration → [`myelin-accelerator`](https://github.com/Limen-Neural/myelin-accelerator);
- end-to-end SAAQ experimentation → [`corinth-canal`](https://github.com/rmems/corinth-canal).

**Allowed dependencies:** none on the default path. Cargo enables no crates unless `--features mmap` (optional `memmap2`) or `--features safetensors` (optional upstream `safetensors` crate, which transitively uses `serde`/`serde_json`). Engram-specific index JSON, manifest serialization, and duplicate-key detection stay in-crate.

**Forbidden dependencies:** inference frameworks, GPU backends, domain-specific adapters, and any dependency on `corinth-canal`.

| Crate | Responsibility |
|---|---|
| `engram-parser` | GGUF metadata + MoE raw extraction + optional mmap/K-quant dequant; feature-gated Safetensors metadata/manifest/discovery/raw payload access |
| `cortex-tensor` | Tensor/Transformer math + real-weight MoE routing |
| `hybrid-fusion` | Backend-agnostic Transformer↔SNN orchestration/contracts |
| `neuromod` | SNN neuron/network dynamics |
| `myelin-accelerator` | Reusable CUDA kernels |
| `corinth-canal` | Experimental end-to-end SAAQ reference/integration lab |

## Origin / modularization — GGUF (#7)

GGUF layout parsing and MoE expert **raw-byte** extraction were expanded using one-way inspiration from the experimental [`rmems/corinth-canal`](https://github.com/rmems/corinth-canal) reference implementation.

`engram-parser` never depends on Corinth. The intended mature direction for GGUF is the opposite dependency: once the reusable parser satisfies Corinth's mmap/K-quant/runtime requirements, Corinth can consume this crate and remove replaceable local parser duplication.

- Primary tracker: [engram-parser#7](https://github.com/rmems/engram-parser/issues/7)
- Corinth migration: [corinth-canal#115](https://github.com/rmems/corinth-canal/issues/115)
- Corinth blocker analysis: [corinth-canal#144](https://github.com/rmems/corinth-canal/issues/144)
- Cortex coordination: [cortex-tensor#8](https://github.com/rmems/cortex-tensor/issues/8)

### GGUF wire types vs “GGML”

GGUF stores each tensor dtype as a `ggml_type` integer. `engram-parser` maps those codes to labels and packed byte sizes so payload and MoE slices remain in range. Packed dequant is implemented for Q8_0, Q5_K, Q6_K, and the internal IQ3_M **block** id — not a GGML runtime.

Wire-type labels follow the Corinth reference discipline: historical wire type **31** is `Q4_0_4_4`, **not** the Hugging Face preset name “IQ3_M”. HuggingFace IQ3_M is a mixed-quant preset; the 111-byte block decoder uses internal `GGML_TYPE_IQ3_M_BLOCK` (`0x4949334D`). Wire 31 still has no packed byte-size model (`DType::Other(31)`), so such a checkpoint is labeled correctly but rejected during layout parsing.

## Origin / modularization — Safetensors (#10)

The previous plan proposed a dedicated `safetensors-parser` repository and described `engram-parser` as permanently GGUF-only. **That plan is superseded. No separate parser repository should be created.**

The reusable Safetensors metadata surface is being promoted into `engram-parser` because it shares the same invariants as the GGUF parser:

- checkpoint metadata deserialization;
- dtype/shape/offset modeling;
- deterministic manifests;
- MoE candidate discovery;
- no numerical runtime;
- no GPU dependency;
- minimal optional dependencies (the `safetensors` feature adds only the upstream `safetensors` crate).

The initial extraction remains one-way:

```text
corinth-canal reference implementation
        ↓ generalize + parity
engram-parser::safetensors
```

However, one-way extraction is an **intermediate promotion mechanism**, not a permanent requirement to maintain duplicate implementations forever. Under the Corinth promotion program ([corinth-canal#161](https://github.com/rmems/corinth-canal/issues/161)), future Corinth dependency adoption should be evaluated once the reusable Safetensors feature is complete enough and passes semantic, payload-boundary, mmap/performance, and parity gates:

```text
engram-parser Safetensors feature complete
        ↓ shared fixtures / parity
release tag or immutable pin
        ↓
optional Corinth dependency adoption
        ↓
remove replaceable local duplicate
```

Hard invariant: **engram-parser never depends on `corinth-canal`**.

### Initial Safetensors extraction boundary

Promote/generalize from `corinth-canal/src/moe/safetensors/`:

- manifest types/generation;
- metadata JSON parsing;
- path/source resolution;
- reusable validation;
- tensor classification and MoE candidate discovery.

Do not automatically promote:

- Corinth-specific Hugging Face `config.json` adapter policy;
- runtime/payload mmap tied to Corinth assumptions;
- GPU registration or dequantization;
- SAAQ campaign/orchestration logic;
- machine-local onboarding configuration.

- Primary tracker: [engram-parser#10](https://github.com/rmems/engram-parser/issues/10)
- README consistency: [engram-parser#61](https://github.com/rmems/engram-parser/issues/61)
- Corinth source tracker: [corinth-canal#116](https://github.com/rmems/corinth-canal/issues/116)
- Corinth architecture umbrella / milestone program: [corinth-canal#161](https://github.com/rmems/corinth-canal/issues/161)
- Hybrid contract consumer: [hybrid-fusion#27](https://github.com/rmems/hybrid-fusion/issues/27)

## Quick start

```rust
use engram_parser::{extract_expert, list_experts, load_gguf};

let layout = load_gguf("./model.gguf")?;
println!("architecture = {}", layout.metadata.architecture());

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

# Ok::<(), engram_parser::ParserError>(())
```

Safetensors (requires `--features safetensors`):

```rust
use engram_parser::safetensors::{inspect_safetensors_checkpoint, open_safetensors_checkpoint};

let manifest = inspect_safetensors_checkpoint("./model.safetensors")?;
println!(
    "kind={} tensors={}",
    manifest.checkpoint.input_kind, manifest.checkpoint.tensor_count
);
for tensor in &manifest.tensors {
    println!("{} {:?} {}", tensor.name, tensor.shape, tensor.dtype);
}

let checkpoint = open_safetensors_checkpoint("./model.safetensors")?;
let raw: Vec<u8> = checkpoint.tensor_bytes("a.weight")?; // bounded read

# Ok::<(), engram_parser::ParserError>(())
```

With `--features safetensors,mmap`, `open_safetensors_checkpoint_mmap` returns borrowed `&[u8]` slices instead of owned copies.

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

## Public API

Current GGUF surface includes:

- `load_gguf`, `parse_bytes` (and `*_with_limits` for an explicit [`ParseLimits`] policy);
- `#[cfg(feature = "mmap")] load_gguf_mmap` / `load_gguf_mmap_with_limits` → `GgufLayoutMmap` (page-aligned tensor slices via `tensor_page_aligned_bytes`);
- `GgufLayout`, `GgufMetadata`, `Tensor`, `DType`, `ParseLimits`;
- `dequantize_f16` (on `Tensor`), `dequantize_q8_0`, `dequantize_q5_k`, `dequantize_q6_k`, `dequantize_iq3_m`;
- `extract_expert`, `list_experts`;
- `MoeExpertWeights`, `RawTensor`;
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

## Ecosystem / promotion model

`corinth-canal` is the experimental proving ground; reusable mechanisms graduate into focused libraries.

For parser work:

```text
corinth-canal
   ├─ GGUF reusable parser/extraction ───────► engram-parser
   └─ Safetensors metadata/discovery ───────► engram-parser::safetensors
```

The destination library becomes the canonical reusable implementation. Corinth may then adopt that library **only after** parity/capability/performance gates justify the dependency swap.

This is coordinated by [corinth-canal#161](https://github.com/rmems/corinth-canal/issues/161) under Corinth's `v0.3.0` modularization/extraction milestone.

## Development

This crate is zero-dependency **by default**. Enable `mmap` for `memmap2`. Enable `safetensors` for header/manifest/discovery plus raw payload access (upstream `safetensors` crate); combine with `mmap` for borrowed payload slices.

```bash
cargo fmt --check
cargo clippy --all-targets --all-features -- -D warnings
cargo build
cargo test
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

For directory scans, `ENGRAM_GGUF_MAX`, and MoE extraction sample counts, see [the T1 pilot guidance in `REVIEW.md`](REVIEW.md#t1--real-gguf-pilots-this-repo-cpu-only).

GPU experiments belong in `blackwell-kernel-lab` / `myelin-accelerator`, not as dependencies of this crate.

See [`REVIEW.md`](REVIEW.md) for quality gates.

## Docker

```bash
docker build -t engram-parser .
docker run --rm engram-parser

docker pull ghcr.io/rmems/engram-parser:main
```

## CI

- GitHub Actions: `.github/workflows/ci.yml`
- Security: `.github/workflows/security.yml`
- Azure Pipelines: `azure-pipelines.yml`
- Docker: `Dockerfile` + `.github/workflows/docker-build.yml`

Related CI/DX trackers include #8, #9, #11–#16.

## MSRV (Minimum Supported Rust Version)

**MSRV: Rust 1.97.1.**

- Declared through `rust-version` in `Cargo.toml`.
- Tested in CI alongside stable.
- MSRV bumps require justification and are treated as compatibility-significant changes.

See [#14](https://github.com/rmems/engram-parser/issues/14).

## Wiki

This repository intentionally keeps documentation in version-controlled files such as `README.md` and `REVIEW.md` rather than a GitHub Wiki.

## License

Licensed under either of:

- Apache License, Version 2.0 ([LICENSE-APACHE-2.0](LICENSE-APACHE-2.0)); or
- MIT license ([LICENSE-MIT](LICENSE-MIT)).
