// SPDX-License-Identifier: MIT OR Apache-2.0

//! Pure-Rust checkpoint and tensor inspection for GGUF and Safetensors.
//!
//! The primary contract is the format-independent
//! [`checkpoint`] module: [`open_checkpoint`] detects the container
//! format and returns a [`Checkpoint`] handle that inventories tensors
//! ([`TensorInfo`], [`TensorShape`], [`TensorDType`],
//! [`TensorLocation`]), reads metadata, and fetches raw payload bytes
//! without the caller branching on GGUF vs. Safetensors.
//!
//! Beneath that contract, [`gguf`] and (feature-gated) `safetensors`
//! are format-specific backends: each exposes its own full parsing
//! surface for callers that need format semantics — GGML wire types,
//! K-quant dequant, Safetensors manifests and MoE candidate discovery.
//!
//! Model-analysis specializations live under [`analysis`]. Today that
//! is [`analysis::moe`]: per-expert raw-weight extraction over a parsed
//! [`GgufLayout`]. MoE is one analysis specialization built above the
//! parsers, not the identity of the crate.
//!
//! # Features
//!
//! - **No dependencies enabled by default**: the default path is pure Rust.
//!   The optional `mmap` feature adds `memmap2`; `safetensors` adds the
//!   upstream `safetensors` crate.
//! - **Parse limits**: [`ParseLimits`] is a documented budget for KV/tensor
//!   counts, string sizes, array work, tensor rank, and metadata bytes.
//!   File-declared `u64` sizes convert with [`HostSizeField`] errors.
//!   Defaults stay generous; trusted callers override explicitly.
//! - **GGUF v3 support**: Full parsing of headers, metadata, and tensor directories
//! - **GGUF wire-type metadata**: labels + packed `byte_len` for known quant
//!   codes (F32/F16/BF16, Q*/IQ*, integers, historical wire 31 = `Q4_0_4_4`).
//!   Packed dequant is implemented for Q8_0, Q5_K, Q6_K, and the internal
//!   IQ3_M block layout. No GGML kernels, no ggml runtime, no CUDA.
//! - **Type labels**: Human-readable names via [`ggml_type_label`] (maps the
//!   on-wire `ggml_type` integer used by GGUF)
//! - **MoE analysis**: Extract expert **raw** weights (byte buffers + shape)
//!   via [`analysis::moe`]; GGUF-specific.
//! - **Metadata helpers**: Architecture-aware convenience methods for common fields
//! - **Optional Safetensors**: enable `--features safetensors` for
//!   inspection, deterministic manifests, MoE candidate discovery, and raw
//!   tensor payload access via the upstream `safetensors` crate. Combine
//!   with `--features mmap` for borrowed mmap-backed payload slices. No
//!   Hugging Face `config.json` policy.
//! - **Format-independent access**: [`Checkpoint`] + [`open_checkpoint`]
//!   inventory tensors, read metadata, and fetch raw payloads from GGUF
//!   or Safetensors without branching on format. See [`checkpoint`].
//!
//! # Example
//!
//! Format-agnostic inventory:
//!
//! ```no_run
//! use engram_parser::{Checkpoint, open_checkpoint};
//!
//! let ckpt = open_checkpoint("model.gguf").unwrap();
//! for t in ckpt.tensors() {
//!     println!("{} {} {:?} {}B", t.name, t.dtype, t.shape.dims(), t.byte_len);
//! }
//! ```
//!
//! MoE expert analysis (GGUF-specific) uses the canonical
//! [`analysis::moe`] path:
//!
//! ```no_run
//! use engram_parser::analysis::moe::{extract_expert, list_experts};
//! use engram_parser::load_gguf;
//!
//! let layout = load_gguf("moe-model.gguf").unwrap();
//! for (block, expert) in list_experts(&layout) {
//!     let weights = extract_expert(&layout, block, expert).unwrap();
//!     println!("blk.{block} expert {expert}: complete={}", weights.is_complete());
//! }
//! ```
//!
//! # Compatibility
//!
//! The legacy `engram_parser::moe` module and the crate-root MoE
//! symbols ([`extract_expert`], [`list_experts`], [`MoeExpertWeights`],
//! [`RawTensor`]) remain as re-exports of [`analysis::moe`] and keep
//! compiling unchanged.

pub mod analysis;
pub mod checkpoint;
pub mod error;
pub mod gguf;
pub mod moe;
#[cfg(feature = "safetensors")]
pub mod safetensors;

// Re-export commonly used types at the crate root for convenience.
#[cfg(feature = "safetensors")]
pub use checkpoint::SafetensorsBackend;
#[cfg(all(feature = "safetensors", feature = "mmap"))]
pub use checkpoint::SafetensorsMmapBackend;
pub use checkpoint::{
    AnyCheckpoint, Checkpoint, CheckpointFormat, CheckpointMetadata, CheckpointSource, DimOrder,
    GgufBackend, MetadataValue, SourceKind, TensorDType, TensorInfo, TensorLocation, TensorShape,
    open_checkpoint,
};
#[cfg(feature = "mmap")]
pub use checkpoint::{GgufMmapBackend, open_checkpoint_mmap};
pub use error::{HostSizeField, ParseLimitKind, ParserError, Result};
pub use gguf::{
    DType,
    // GGML type constants
    GGML_TYPE_BF16,
    GGML_TYPE_F16,
    GGML_TYPE_F32,
    GGML_TYPE_F64,
    GGML_TYPE_I8,
    GGML_TYPE_I16,
    GGML_TYPE_I32,
    GGML_TYPE_I64,
    GGML_TYPE_IQ1_M,
    GGML_TYPE_IQ1_S,
    GGML_TYPE_IQ2_S,
    GGML_TYPE_IQ2_XS,
    GGML_TYPE_IQ2_XXS,
    GGML_TYPE_IQ3_M_BLOCK,
    GGML_TYPE_IQ3_S,
    GGML_TYPE_IQ3_XXS,
    GGML_TYPE_IQ4_NL,
    GGML_TYPE_IQ4_XS,
    GGML_TYPE_Q2_K,
    GGML_TYPE_Q3_K,
    GGML_TYPE_Q4_0,
    GGML_TYPE_Q4_0_4_4,
    GGML_TYPE_Q4_1,
    GGML_TYPE_Q4_K,
    GGML_TYPE_Q5_0,
    GGML_TYPE_Q5_1,
    GGML_TYPE_Q5_K,
    GGML_TYPE_Q6_K,
    GGML_TYPE_Q8_0,
    GGML_TYPE_Q8_1,
    GGML_TYPE_Q8_K,
    // Metadata value type constants
    GGUF_VALUE_TYPE_ARRAY,
    GGUF_VALUE_TYPE_BOOL,
    GGUF_VALUE_TYPE_FLOAT32,
    GGUF_VALUE_TYPE_FLOAT64,
    GGUF_VALUE_TYPE_INT8,
    GGUF_VALUE_TYPE_INT16,
    GGUF_VALUE_TYPE_INT32,
    GGUF_VALUE_TYPE_INT64,
    GGUF_VALUE_TYPE_STRING,
    GGUF_VALUE_TYPE_UINT8,
    GGUF_VALUE_TYPE_UINT16,
    GGUF_VALUE_TYPE_UINT32,
    GGUF_VALUE_TYPE_UINT64,
    GgufLayout,
    GgufMetadata,
    ParseLimits,
    Tensor,
    dequantize_iq3_m,
    dequantize_packed,
    dequantize_q5_k,
    dequantize_q6_k,
    dequantize_q8_0,
    f16_bits_to_f32,
    ggml_type_label,
    load_gguf,
    load_gguf_with_limits,
    packed_row_size,
    parse_bytes,
    parse_bytes_with_limits,
};
#[cfg(feature = "mmap")]
pub use gguf::{
    GgufLayoutMmap, PageAlignedTensorBytes, load_gguf_mmap, load_gguf_mmap_with_limits,
    os_page_size,
};
// MoE analysis result types and extractors, re-exported for
// compatibility; the canonical path is `analysis::moe`.
pub use analysis::moe::{MoeExpertWeights, RawTensor, extract_expert, list_experts};
