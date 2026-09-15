// SPDX-License-Identifier: MIT OR Apache-2.0

//! GGUF file format: header + metadata + tensor directory.
//!
//! Entry point: [`load_gguf`] reads a `.gguf` file into a [`GgufLayout`]
//! containing parsed metadata, a tensor directory, and the underlying
//! byte buffer. Packed K-quant dequant helpers live in [`dequant`].
//! Optional mmap-backed loading is behind the `mmap` feature.

mod cursor;
mod dequant;
mod layout;
#[cfg(feature = "mmap")]
mod map;
mod tensor;

use std::fs;
use std::path::Path;

pub use dequant::{
    dequantize_iq3_m, dequantize_packed, dequantize_q5_k, dequantize_q6_k, dequantize_q8_0,
    packed_row_size,
};
pub use layout::{GgufLayout, GgufMetadata};
#[cfg(feature = "mmap")]
pub use map::{GgufLayoutMmap, PageAlignedTensorBytes, load_gguf_mmap, os_page_size};
pub use tensor::{
    DType, GGML_TYPE_BF16, GGML_TYPE_F16, GGML_TYPE_F32, GGML_TYPE_F64, GGML_TYPE_I8,
    GGML_TYPE_I16, GGML_TYPE_I32, GGML_TYPE_I64, GGML_TYPE_IQ1_M, GGML_TYPE_IQ1_S, GGML_TYPE_IQ2_S,
    GGML_TYPE_IQ2_XS, GGML_TYPE_IQ2_XXS, GGML_TYPE_IQ3_M_BLOCK, GGML_TYPE_IQ3_S, GGML_TYPE_IQ3_XXS,
    GGML_TYPE_IQ4_NL, GGML_TYPE_IQ4_XS, GGML_TYPE_Q2_K, GGML_TYPE_Q3_K, GGML_TYPE_Q4_0,
    GGML_TYPE_Q4_0_4_4, GGML_TYPE_Q4_1, GGML_TYPE_Q4_K, GGML_TYPE_Q5_0, GGML_TYPE_Q5_1,
    GGML_TYPE_Q5_K, GGML_TYPE_Q6_K, GGML_TYPE_Q8_0, GGML_TYPE_Q8_1, GGML_TYPE_Q8_K, Tensor,
    f16_bits_to_f32, ggml_type_label,
};

// Re-export metadata value type constants for public API.
pub use cursor::{
    GGUF_VALUE_TYPE_ARRAY, GGUF_VALUE_TYPE_BOOL, GGUF_VALUE_TYPE_FLOAT32, GGUF_VALUE_TYPE_FLOAT64,
    GGUF_VALUE_TYPE_INT8, GGUF_VALUE_TYPE_INT16, GGUF_VALUE_TYPE_INT32, GGUF_VALUE_TYPE_INT64,
    GGUF_VALUE_TYPE_STRING, GGUF_VALUE_TYPE_UINT8, GGUF_VALUE_TYPE_UINT16, GGUF_VALUE_TYPE_UINT32,
    GGUF_VALUE_TYPE_UINT64,
};

use crate::error::{ParserError, Result};

/// Load a `.gguf` checkpoint from disk and parse its header, KV
/// metadata, and tensor directory.
///
/// The full file contents are read into memory (no mmap on this path;
/// default builds stay zero-dep). For multi-GB checkpoints enable the
/// `mmap` feature and use `load_gguf_mmap`. Tensor payloads remain
/// available as raw byte slices via [`GgufLayout::tensor_bytes`].
pub fn load_gguf<P: AsRef<Path>>(path: P) -> Result<GgufLayout> {
    let path_ref = path.as_ref();
    let path_str = path_ref.display().to_string();
    let bytes = fs::read(path_ref).map_err(|e| ParserError::Io {
        path: path_str.clone(),
        source: e,
    })?;
    parse_bytes(bytes, path_str)
}

/// Parse an already-loaded byte buffer as a GGUF checkpoint. Useful for
/// unit tests and in-memory round-trips.
pub fn parse_bytes(bytes: Vec<u8>, path: String) -> Result<GgufLayout> {
    let (metadata, tensors, alignment, tensor_data_offset) = layout::parse_layout(&bytes, &path)?;
    Ok(GgufLayout {
        path,
        metadata,
        tensors,
        alignment,
        tensor_data_offset,
        bytes,
    })
}
