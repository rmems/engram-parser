# Changelog

All notable changes to this project are documented in this file.

## [Unreleased]

### Added

- **BF16→F32 decode ([RM-2366](https://linear.app/rpd-34/issue/RM-2366) / #126):**
  `Tensor::dequantize_bf16`, `bf16_bits_to_f32`, `dequantize_bf16_bytes`
  (Safetensors `BF16` payloads; rejects odd lengths), and
  `dequantize_packed(DType::BF16, …)`. Lossless bit widening; NaN payloads,
  signed zeros, infinities, and subnormals are preserved. Path-gated
  real-checkpoint smoke test in `tests/bf16_decode.rs`.
- **Transformer topology discovery ([RM-1785](https://linear.app/rpd-34/issue/RM-1785) / #88):**
  new `analysis::transformer` module (default features, zero dependencies).
  `discover_topology(&dyn Checkpoint)` / `discover_topology_from_parts`
  classify every tensor of a GGUF or Safetensors inventory into a
  `TensorRole` (token/position embedding, output head/norm, attention
  norm/Q/K/V/fused-QKV/O/Q-norm/K-norm, MLP norm/gate/up/down/fused
  gate-up, MoE router, rotary frequencies, unknown) with a `RoleScope`
  (global, layer, `Expert(i)`, stacked experts, shared expert), a
  `ParamKind`, a `RoleConfidence` (exact / heuristic / ambiguous /
  unknown), listed `alternatives` for ambiguous names, and evidence
  strings. Layers assemble into `TransformerTopology` → `LayerTopology`
  (`AttentionKind`, `FeedForwardKind`, router / expert-count /
  shared-expert flags) plus a `TopologySummary` (role and confidence
  counts, missing layers, hidden/vocab/tied-embedding hints). MoE roles
  reuse the dense MLP roles under an expert scope, so dense and MoE
  checkpoints share one topology layer. Covers llama.cpp GGUF (stacked
  `*_exps` and per-expert `ffn_<role>.<E>` / `ffn_<role>_<E>`), Hugging
  Face Llama / Mixtral / Qwen-MoE / Gemma-2, GPT-2, GPT-NeoX, original
  Llama, and BERT naming. Layers are grouped by `(stack, index)` so
  multi-tower checkpoints (`visual.blocks.N` next to `model.layers.N`)
  are never merged; `TopologySummary::primary_stack` / `layer_stacks`
  expose the split. `missing_layers` uses saturating arithmetic and is
  capped at `MAX_MISSING_LAYERS` (4096) with a `missing_layers_truncated`
  flag, so hostile layer indices or `block_count` values cannot force
  large allocations. Output is order-independent and deterministic;
  `Display` gives a text inventory and `to_json()` a dependency-free JSON
  rendering. Unknown tensors stay `Unknown`; a bare `mlp.gate` stays
  `Ambiguous` unless the checkpoint holds expert tensors or the shape is
  router-like. Descriptive only: no forward pass, tokenizer, or
  family-specific runtime policy. Adds the `transformer_topology` example.

### Changed

- **Coverage CI ([RM-1951](https://linear.app/rpd-34/issue/RM-1951)):** the
  Coverage job skips the Codecov upload when `CODECOV_TOKEN` is unset on a
  pull request and keeps the LCOV artifact. A `push` or `schedule` run
  without that secret still fails the job. `cargo llvm-cov` still runs and
  still fails the job when tests fail. An upload that has the token still
  fails the job on uploader errors.

### Fixed

- **GGUF tensor offset alignment ([RM-2204](https://linear.app/rpd-34/issue/RM-2204) / #120):**
  `finalize_tensor_offsets` now rejects any tensor whose `relative_offset` is
  not a multiple of `general.alignment` (default 32) with `InvalidLayout`
  naming the tensor, offset, and alignment, matching the GGUF spec and
  llama.cpp. Files with unaligned tensor offsets that previously parsed now
  fail at parse time. The check runs after the offset-overflow guard and before
  the bounds and overlap checks.

## [0.3.0] - 2026-10-02

**First public crates.io publication.** Version
0.2.0 was used only during development and was never published.

### Added

- **Format-independent checkpoint API ([RM-1784](https://linear.app/rpd-34/issue/RM-1784) / #87):**
  new `checkpoint` module (default features, re-exported at the root)
  with an object-safe `Checkpoint` trait and engram-owned
  `CheckpointSource`, `TensorInfo`, `TensorDType`, `TensorShape`,
  `TensorLocation`, and `MetadataValue` types. The `GgufBackend` /
  `GgufMmapBackend` / `SafetensorsBackend` / `SafetensorsMmapBackend`
  adapters wrap the existing handles, which stay reachable via
  `layout()` / `checkpoint()`. `open_checkpoint` (and
  `open_checkpoint_mmap`) detect the format by path type, `GGUF` magic,
  and Safetensors extensions and return `AnyCheckpoint`. Shared shapes
  are outermost-first (GGUF dims reversed, native order recoverable);
  GGML quantized layouts stay `TensorDType::GgufPacked { ggml_type }`;
  payloads are raw `Cow<[u8]>`. Adds `ParserError::FeatureDisabled` and
  the `inspect_checkpoint` example. Existing APIs are unchanged.
- **Safetensors raw payload access ([RM-1783](https://linear.app/rpd-34/issue/RM-1783) / #86):**
  the `safetensors` feature now pulls the upstream `safetensors` crate
  (0.8) for canonical header validation and adds
  `open_safetensors_checkpoint` → `SafetensorsCheckpoint` with bounded
  per-tensor `tensor_bytes` reads over single-file, shard-index, and
  directory layouts. With `--features safetensors,mmap`,
  `open_safetensors_checkpoint_mmap` → `SafetensorsCheckpointMmap`
  returns payload `&[u8]` borrowed from per-shard `memmap2` mappings, so
  large sharded checkpoints need no whole-checkpoint copies. MoE
  discovery results (`SafetensorsTensorRecord`, candidate names) resolve
  to real raw bytes via `resolve_tensor_bytes`/`tensor_bytes`. Upstream
  `SafeTensors`/`TensorView`/`Dtype` types are not part of the public
  API; discovery, shard path policy, duplicate-key detection, and
  contiguous-range validation remain engram-owned. Each shard header is
  revalidated against the manifest on the file handle retained at open,
  and reads pin to that handle under a lock.
- **Safetensors feature (`#10`):** off-by-default `safetensors` cargo feature
  for header-only inspection, deterministic manifests, single-file / Hugging
  Face shard-index / directory layouts, and MoE router/expert candidate
  discovery. Public API lives in `engram_parser::safetensors`. Does
  not include Hugging Face `config.json` policy. As of RM-1783 the
  feature uses the upstream `safetensors` crate for canonical
  validation; the earlier zero-extra-crates claim no longer applies.
- **GGUF `ParseLimits` (RM-1358):** documented resource budgets for untrusted
  GGUF headers — KV count, tensor count, string bytes, array work items,
  tensor rank, and metadata bytes. File-declared `u64` sizes convert with
  typed [`HostSizeField`] errors; exhausted budgets return
  [`ParserError::LimitExceeded`] naming the limit. Alignment, tensor offsets,
  element counts, and packed byte sizes use checked arithmetic. Default and
  `mmap` readers share the same policy. Trusted callers override via
  `load_gguf_with_limits` / `parse_bytes_with_limits` /
  `load_gguf_mmap_with_limits` without changing default safety.
- **Safetensors shard-index invariants ([RM-1360](https://linear.app/rpd-34/issue/RM-1360)):**
  index shard paths resolve relative to the checkpoint root; absolute paths
  and `..` traversal that would leave the root are rejected; existing paths
  are canonicalized to reject symlink escape where the platform can follow
  links. Each indexed tensor must map to exactly one existing shard.
  Duplicate/conflicting ownership fails with
  `ParserError::DuplicateTensorOwnership` (tensor name + shard list). Missing
  referenced shards fail with `ParserError::MissingShard` before a manifest is
  returned. Unreferenced on-disk shards are reported via
  `index:unreferenced_shards` rather than rejected. Directory listings and
  tensor records are sorted so repeated inspection is byte-identical.
- **Optional `mmap` feature (#45 option 1 / RM-367):** `load_gguf_mmap` maps a
  GGUF with `memmap2` 0.9.11 instead of `fs::read` into a `Vec<u8>`. Default
  builds stay zero-dep (`default = []`). Packed CPU dequant for **Q8_0**,
  **Q5_K**, **Q6_K**, and the internal **IQ3_M block** layout
  (`GGML_TYPE_IQ3_M_BLOCK = 0x4949334D`, 111 bytes / 256 values). Wire type
  **31** remains historical **Q4_0_4_4** (`DType::Other(31)`). CUDA
  host-register is still out of scope. `GgufLayoutMmap::directory_matches`
  compares the full tensor directory (not just count/architecture). CI covers
  mmap↔owned packed dequant, a tensor past the first OS page, and a **sparse
  2 GiB** mapping that never calls `fs::read`. Real on-disk multi-GB pilots
  remain `#[ignore]` behind `ENGRAM_GGUF`.

### Changed

- **API organization ([RM-1792](https://linear.app/rpd-34/issue/RM-1792)):**
  MoE extraction moved behind the model-analysis namespace
  `engram_parser::analysis::moe`, making `checkpoint` the canonical core
  contract for the v0.3 crates.io surface. `MoeExpertWeights`,
  `RawTensor`, `extract_expert`, and `list_experts` now live canonically
  under `analysis::moe`; `engram_parser::moe` and the crate-root
  re-exports remain as compatibility shims over the same single
  implementation, so existing Git consumers keep compiling unchanged.
  `RawTensor` is documented as a GGUF-specific analysis result
  (`DType`/`ggml_type`), not a tensor abstraction — `TensorInfo` +
  `Checkpoint::tensor_bytes` stay the core contract. Safetensors
  `discover_candidates` is unchanged and remains a separate,
  format-specific discovery surface. No behavior changes.
- **Docs/packaging ([RM-1793](https://linear.app/rpd-34/issue/RM-1793) / RM-1782):**
  the crate `description` now leads with the GGUF + Safetensors
  checkpoint/tensor substrate for ANN/SNN research rather than MoE
  extraction; README scope/boundary docs name `grok-ozempic` for
  GOZ1/Grok quantization policy. No API changes.
- **Charter reversal (#10):** safetensors support ships **inside this
  crate** behind an off-by-default `safetensors` cargo feature, not in a
  separate `safetensors-parser` crate. This supersedes the "engram-parser
  charter remains GGUF-only" language previously carried by `README.md`,
  `REVIEW.md`, corinth-canal `docs/MODULE_STATUS.md`, and cortex-tensor#9.
  Default builds keep `[dependencies]` empty; optional dependencies are
  `memmap2` (`mmap`) and the upstream `safetensors` crate
  (`safetensors`). The initial Safetensors extraction is a one-way copy from
  `rmems/corinth-canal` inspiration with no dependency in either direction;
  it is not a permanent duplication policy. Once this crate meets the
  relevant adoption requirements, a future Corinth dependency may replace
  the corresponding local copy. (GGUF is the opposite case:
  corinth-canal#115 already plans a real `engram-parser` dependency, gated on
  #45.) Safetensors is now **code** behind the feature, not a decision-only
  note.

## [0.2.0] - Unpublished development version

### Added

- GGUF tensor **wire-type** layouts (IQ/Q codes + packed `byte_len` only — **no dequant**).
- `file_type` metadata fallback for quantization label when general quant keys are absent.
- Path-gated **T1** real-GGUF pilots (`tests/real_gguf.rs`) with optional
  `ENGRAM_EXPECT_MOE` / `ENGRAM_MOE_SAMPLES`.
- `examples/inspect_gguf` for human inventory of on-disk GGUF files.
- Quality-gate docs in `REVIEW.md` (T0/T1/T2; large MoE RAM budget).
- Local `rust-toolchain.toml` (`channel = "stable"`).
- **GitHub Actions CI** — `fmt`, `clippy`, `build`, and `test` on push/PR to `main`.
- **Boundary documentation** — README scope/ownership section linked to Linear LIM-9.

### Changed

- **Version:** `0.1.0` → **`0.2.0`** during development (canonical GGUF v3 + MoE extract for #7); this version was not published.
- **MSRV:** bumped from 1.87 to **1.97.1** (`Cargo.toml` `rust-version`, CI `msrv` job, Docker `RUST_VERSION`). CI `validate` continues to use latest **stable**.
- **License:** switched from GPL-3.0-or-later to dual MIT/Apache-2.0 for maximum adoption and ecosystem health.
- **Tensor API:** replaced unsafe `as_f32_slice` / `as_u16_bits` with safe `read_f32_values` / `read_u16_values` (allocating `Vec` instead of borrowed slices).
- **`GgufMetadata::quantization()`** returns `String` (owned) so `general.file_type` fallback is derived at call time from the live map. Callers that match on the label should use `.as_str()` or `==`.
- Wire type **31** treated as historical **Q4_0_4_4** (not IQ3_M).

### Fixed

- Wire-type 31 labeling aligned with corinth-canal’s GGUF/`ggml_type` table (metadata only).

## [0.1.0] - 2026-06-01

- Initial release: pure-Rust, zero-dependency GGUF v3 deserializer.
- MoE expert enumeration and per-expert weight extraction (stacked and per-expert conventions).
- Layout-aware dtype handling for F32, F16, BF16, and opaque quant types.
