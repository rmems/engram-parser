# engram-parser

[![CI](https://github.com/rmems/engram-parser/actions/workflows/ci.yml/badge.svg)](https://github.com/rmems/engram-parser/actions/workflows/ci.yml)
[![License: MIT OR Apache-2.0](https://img.shields.io/badge/License-MIT%20OR%20Apache--2.0-blue.svg)](LICENSE-MIT)

Pure-Rust, **zero-dependency checkpoint metadata parsing and Mixture-of-Experts extraction**.

Today, `engram-parser` ships GGUF v3 deserialization and per-expert raw-weight extraction. Safetensors **header/manifest/MoE-discovery support is the next parser surface** and is being promoted from the `corinth-canal` reference implementation behind a planned, off-by-default `safetensors` Cargo feature.

> **Safetensors status:** planned / in flight, **not shipped yet**. Until the port tracked in [#10](https://github.com/rmems/engram-parser/issues/10) lands, `cargo build --features safetensors` will fail because the feature does not yet exist.

## What it does

### Shipped now — GGUF

- Parses GGUF v3 magic, header, KV metadata, and tensor directory into an in-memory [`GgufLayout`].
- Enumerates MoE experts discovered in a checkpoint.
- Extracts the raw byte buffers for one expert's `gate`, `up`, and `down` projections.
- Supports stacked (`blk.{B}.ffn_{role}_exps.weight`) and per-expert (`blk.{B}.ffn_{role}.{E}.weight`) conventions.
- Models packed GGUF dtype sizes without pulling in a numerical runtime.

### Planned / in flight — Safetensors

The `safetensors` feature will own reusable **metadata-side** support for:

- Safetensors header deserialization;
- deterministic tensor manifests;
- single-file, Hugging Face shard-index, and directory layouts;
- tensor name/dtype/shape/offset/shard metadata;
- MoE router/expert candidate discovery and grouping;
- metadata-only layout-family inference where it is reusable outside Corinth.

This means **engram-parser will support the Safetensors format itself**. It does **not** mean adding the upstream Rust `safetensors` crate as a dependency. The zero-dependency charter remains in force.

Tracking: [engram-parser#10](https://github.com/rmems/engram-parser/issues/10), [engram-parser#61](https://github.com/rmems/engram-parser/issues/61), and [corinth-canal#116](https://github.com/rmems/corinth-canal/issues/116).

## What it does NOT do

- No neural-network math: no `matmul`, forward pass, routing execution, or softmax.
- No CUDA, GPU, SIMD, or runtime inference engine.
- No tokenization or SNN dynamics.
- No full model-family routing policy; see [`cortex-tensor`](https://github.com/rmems/cortex-tensor).
- No Safetensors payload execution/matmul.
- No Corinth experiment orchestration or SAAQ calibration.

Safetensors payload **mmap/tensor extraction** and Hugging Face `config.json` interpretation remain outside the initial metadata feature when they depend on runtime/model-policy concerns. Those boundaries can be reconsidered through separate source-level promotion issues rather than silently expanding this parser crate.

## Scope / boundaries

This crate **owns**:

- GGUF v3 deserialization: header, KV metadata, tensor directory.
- MoE expert enumeration and per-expert raw-weight extraction.
- Zero-dependency, layout-aware dtype handling.
- Planned feature-gated Safetensors header parsing, manifests, and MoE candidate discovery.

This crate **does not own**:

- tensor/Transformer/MoE numerical execution → [`cortex-tensor`](https://github.com/rmems/cortex-tensor);
- Transformer↔SNN orchestration/contracts → [`hybrid-fusion`](https://github.com/rmems/hybrid-fusion);
- neuron/network dynamics → [`neuromod`](https://github.com/Limen-Neural/neuromod);
- CUDA acceleration → [`myelin-accelerator`](https://github.com/Limen-Neural/myelin-accelerator);
- end-to-end SAAQ experimentation → [`corinth-canal`](https://github.com/rmems/corinth-canal).

**Allowed dependencies:** none. `[dependencies]` is intentionally empty and is expected to remain empty under the planned `safetensors` feature. Header/shard-index JSON parsing is implemented in-crate rather than through `serde_json` or the upstream `safetensors` crate.

**Forbidden dependencies:** inference frameworks, GPU backends, domain-specific adapters, and any dependency on `corinth-canal`.

| Crate | Responsibility |
|---|---|
| `engram-parser` | GGUF metadata + MoE raw extraction; planned Safetensors metadata/manifest/discovery |
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

GGUF stores each tensor dtype as a `ggml_type` integer. `engram-parser` maps those codes to labels and packed byte sizes so payload and MoE slices remain in range. It does **not** implement the GGML runtime, GPU kernels, or general dequantized inference.

Wire-type labels follow the Corinth reference discipline: for example, historical wire type **31** is `Q4_0_4_4`, not the Hugging Face preset name “IQ3_M”.

## Origin / modularization — Safetensors (#10)

The previous plan proposed a dedicated `safetensors-parser` repository and described `engram-parser` as permanently GGUF-only. **That plan is superseded. No separate parser repository should be created.**

The reusable Safetensors metadata surface is being promoted into `engram-parser` because it shares the same invariants as the GGUF parser:

- checkpoint metadata deserialization;
- dtype/shape/offset modeling;
- deterministic manifests;
- MoE candidate discovery;
- no numerical runtime;
- no GPU dependency;
- zero-dependency parsing.

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

A Safetensors quick-start example will be added when the feature actually lands; the README deliberately does not document an API that is not yet shipped.

## Supported GGUF dtypes

Layout-aware parsing (**packed byte sizes only — no general dequant or GGML compute**) supports:

- `F32`, `F16`, `BF16`, `F64`;
- `I8`–`I64`;
- `Q4_0`, `Q4_1`, `Q5_0`, `Q5_1`, `Q8_0`, `Q8_1`;
- K-quants `Q2_K`, `Q3_K`, `Q4_K`, `Q5_K`, `Q6_K`, `Q8_K`;
- IQ packed layouts including `IQ2_XXS`, `IQ2_XS`, `IQ2_S`, `IQ3_XXS`, `IQ3_S`, `IQ1_S`, `IQ1_M`, `IQ4_NL`, `IQ4_XS`.

Remaining codes use `DType::Other(u32)`. Unknown packed layouts fail closed when element count cannot be converted to a modeled byte length.

Only F32/F16 have in-crate numeric helpers; the parser's primary contract is layout/raw bytes, not general tensor compute.

## Public API

Current GGUF surface includes:

- `load_gguf`, `parse_bytes`;
- `GgufLayout`, `GgufMetadata`, `Tensor`, `DType`;
- `extract_expert`, `list_experts`;
- `MoeExpertWeights`, `RawTensor`;
- `ParserError`, `Result`;
- public GGML/GGUF constants and labels.

Safetensors public types/functions will be documented only after #10 lands.

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

This is a pure-Rust, zero-dependency crate.

```bash
cargo fmt --check
cargo clippy --all-targets --all-features -- -D warnings
cargo build --all-features
cargo test --all-features

# Coverage (requires cargo-llvm-cov)
cargo llvm-cov --all-targets --all-features --locked --lcov --output-path lcov.info
```

Real GGUF pilots require local checkpoint files and are intentionally not CI fixtures:

```bash
ENGRAM_GGUF=~/.models/gguf/.../model.gguf \
  cargo test --test real_gguf -- --ignored --nocapture

cargo run --example inspect_gguf -- ~/.models/gguf/.../model.gguf
```

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