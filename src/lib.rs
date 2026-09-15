// SPDX-License-Identifier: MIT OR Apache-2.0

//! Pure-Rust, zero-dependency GGUF parser with MoE support.
//!
//! This crate parses GGUF (GPT-Generated Unified Format) v3 files,
//! extracts metadata and tensor information, and provides utilities
//! for Mixture of Experts (MoE) model analysis.
//!
//! # Features
//!
//! - **Zero dependencies by default**: the default path is pure Rust with an
//!   empty `[dependencies]`. The optional `mmap` feature adds `memmap2`.
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
//! - **MoE support**: Extract expert **raw** weights (byte buffers + shape)
//! - **Metadata helpers**: Architecture-aware convenience methods for common fields
//! - **Optional Safetensors**: enable `--features safetensors` for header-only
//!   inspection, deterministic manifests, and MoE candidate discovery (no
//!   payload mmap, no Hugging Face `config.json` policy)
//!
//! # Example
//!
//! ```no_run
//! use engram_parser::{load_gguf, ggml_type_label};
//!
//! let layout = load_gguf("model.gguf").unwrap();
//! println!("Architecture: {}", layout.metadata.architecture());
//! println!("Quantization: {}", layout.metadata.quantization());
//!
//! if let Some(block_count) = layout.metadata.block_count() {
//!     println!("Blocks: {}", block_count);
//! }
//!
//! for (name, tensor) in &layout.tensors {
//!     println!("{}: {:?} (type: {})",
//!         name, tensor.dims,
//!         ggml_type_label(tensor.ggml_type));
//! }
//! ```

pub mod error;
pub mod gguf;
pub mod moe;
#[cfg(feature = "safetensors")]
pub mod safetensors;

// Re-export commonly used types at the crate root for convenience.
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
pub use moe::{MoeExpertWeights, RawTensor, extract_expert, list_experts};
