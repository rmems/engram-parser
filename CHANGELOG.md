# Changelog

All notable changes to this project are documented in this file.

## [Unreleased]

### Added

- **Safetensors feature (`#10`):** off-by-default `safetensors` cargo feature
  for header-only inspection, deterministic manifests, single-file / Hugging
  Face shard-index / directory layouts, and MoE router/expert candidate
  discovery. Public API lives in `engram_parser::safetensors`. Zero extra
  crates: `[dependencies]` stays empty for this feature; the upstream
  `safetensors` crate, `serde_json`, and `corinth-canal` are not used. Does
  not include payload mmap or Hugging Face `config.json` policy.
- **GGUF `ParseLimits` (RM-1358):** documented resource budgets for untrusted
  GGUF headers — KV count, tensor count, string bytes, array work items,
  tensor rank, and metadata bytes. File-declared `u64` sizes convert with
  typed [`HostSizeField`] errors; exhausted budgets return
  [`ParserError::LimitExceeded`] naming the limit. Alignment, tensor offsets,
  element counts, and packed byte sizes use checked arithmetic. Default and
  `mmap` readers share the same policy. Trusted callers override via
  `load_gguf_with_limits` / `parse_bytes_with_limits` /
  `load_gguf_mmap_with_limits` without changing default safety.
- **Optional `mmap` feature (#45 option 1):** `load_gguf_mmap` maps a GGUF with
  `memmap2` 0.9.11 instead of `fs::read` into a `Vec<u8>`. Default builds stay
  zero-dep (`default = []`). Packed CPU dequant for **Q8_0**, **Q5_K**,
  **Q6_K**, and the internal **IQ3_M block** layout
  (`GGML_TYPE_IQ3_M_BLOCK = 0x4949334D`, 111 bytes / 256 values). Wire type
  **31** remains historical **Q4_0_4_4** (`DType::Other(31)`). CUDA
  host-register is still out of scope.

### Changed

- **Charter reversal (#10):** safetensors support ships **inside this
  crate** behind an off-by-default `safetensors` cargo feature, not in a
  separate `safetensors-parser` crate. This supersedes the "engram-parser
  charter remains GGUF-only" language previously carried by `README.md`,
  `REVIEW.md`, corinth-canal `docs/MODULE_STATUS.md`, and cortex-tensor#9.
  Default builds keep `[dependencies]` empty. The `mmap` feature is the
  only optional dependency (`memmap2`); the `safetensors` feature stays
  zero-dep. The initial Safetensors extraction is a one-way copy from
  `rmems/corinth-canal` inspiration with no dependency in either direction;
  it is not a permanent duplication policy. Once this crate meets the
  relevant adoption requirements, a future Corinth dependency may replace
  the corresponding local copy. (GGUF is the opposite case:
  corinth-canal#115 already plans a real `engram-parser` dependency, gated on
  #45.) Safetensors is now **code** behind the feature, not a decision-only
  note.

## [0.2.0] - 2026-08-02

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

- **Version:** `0.1.0` → **`0.2.0`** (canonical GGUF v3 + MoE extract ship for #7).
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
