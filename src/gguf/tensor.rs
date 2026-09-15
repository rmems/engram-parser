// SPDX-License-Identifier: MIT OR Apache-2.0

//! Tensor directory entry + dtype enumeration + GGUF wire-type helpers.
//!
//! A [`Tensor`] is a pure-metadata descriptor: name, shape, dtype, and
//! byte offset within the file. It owns no weight data itself — callers
//! pass it back to [`GgufLayout::tensor_bytes`](super::layout::GgufLayout::tensor_bytes)
//! to obtain the raw `&[u8]` payload.
//!
//! ## GGUF `ggml_type` codes (metadata only)
//!
//! GGUF stores each tensor’s dtype as a `ggml_type` `u32`. The
//! `GGML_TYPE_*` constants mirror that table (same numbers as `ggml.h`)
//! so we can label types and compute **packed byte lengths**. Packed
//! dequant for Q8_0/Q5_K/Q6_K/IQ3_M lives in [`super::dequant`].
//! [`ggml_type_label`] maps any `u32` code to a short diagnostic string.

use crate::error::{ParserError, Result};

// ---------------------------------------------------------------------------
// GGML type constants (mirror `ggml.h` as of 2025-06).
// ---------------------------------------------------------------------------

/// `GGML_TYPE_F32` — 32-bit IEEE-754 float.
pub const GGML_TYPE_F32: u32 = 0;
/// `GGML_TYPE_F16` — 16-bit IEEE-754 half float.
pub const GGML_TYPE_F16: u32 = 1;
/// `GGML_TYPE_Q4_0` — 4-bit quantization (symmetric, block size 32).
pub const GGML_TYPE_Q4_0: u32 = 2;
/// `GGML_TYPE_Q4_1` — 4-bit quantization (with min, block size 32).
pub const GGML_TYPE_Q4_1: u32 = 3;
/// `GGML_TYPE_Q5_0` — 5-bit quantization (symmetric, block size 32).
pub const GGML_TYPE_Q5_0: u32 = 6;
/// `GGML_TYPE_Q5_1` — 5-bit quantization (with min, block size 32).
pub const GGML_TYPE_Q5_1: u32 = 7;
/// `GGML_TYPE_Q8_0` — 8-bit quantization (symmetric, block size 32).
pub const GGML_TYPE_Q8_0: u32 = 8;
/// `GGML_TYPE_Q8_1` — 8-bit quantization (with min, block size 32).
pub const GGML_TYPE_Q8_1: u32 = 9;
/// `GGML_TYPE_Q2_K` — k-quant 2-bit.
pub const GGML_TYPE_Q2_K: u32 = 10;
/// `GGML_TYPE_Q3_K` — k-quant 3-bit.
pub const GGML_TYPE_Q3_K: u32 = 11;
/// `GGML_TYPE_Q4_K` — k-quant 4-bit.
pub const GGML_TYPE_Q4_K: u32 = 12;
/// `GGML_TYPE_Q5_K` — k-quant 5-bit.
pub const GGML_TYPE_Q5_K: u32 = 13;
/// `GGML_TYPE_Q6_K` — k-quant 6-bit.
pub const GGML_TYPE_Q6_K: u32 = 14;
/// `GGML_TYPE_Q8_K` — k-quant 8-bit.
pub const GGML_TYPE_Q8_K: u32 = 15;
/// `GGML_TYPE_IQ2_XXS` — i-quant 2-bit extra-extra-small.
pub const GGML_TYPE_IQ2_XXS: u32 = 16;
/// `GGML_TYPE_IQ2_XS` — i-quant 2-bit extra-small.
pub const GGML_TYPE_IQ2_XS: u32 = 17;
/// `GGML_TYPE_IQ3_XXS` — i-quant 3-bit extra-extra-small.
pub const GGML_TYPE_IQ3_XXS: u32 = 18;
/// `GGML_TYPE_IQ1_S` — i-quant 1-bit small.
pub const GGML_TYPE_IQ1_S: u32 = 19;
/// `GGML_TYPE_IQ4_NL` — i-quant 4-bit non-linear.
pub const GGML_TYPE_IQ4_NL: u32 = 20;
/// `GGML_TYPE_IQ3_S` — i-quant 3-bit small (3.44 bpw).
pub const GGML_TYPE_IQ3_S: u32 = 21;
/// `GGML_TYPE_IQ2_S` — i-quant 2-bit small.
pub const GGML_TYPE_IQ2_S: u32 = 22;
/// `GGML_TYPE_IQ4_XS` — i-quant 4-bit extra-small.
pub const GGML_TYPE_IQ4_XS: u32 = 23;
/// `GGML_TYPE_I8` — 8-bit signed integer.
pub const GGML_TYPE_I8: u32 = 24;
/// `GGML_TYPE_I16` — 16-bit signed integer.
pub const GGML_TYPE_I16: u32 = 25;
/// `GGML_TYPE_I32` — 32-bit signed integer.
pub const GGML_TYPE_I32: u32 = 26;
/// `GGML_TYPE_I64` — 64-bit signed integer.
pub const GGML_TYPE_I64: u32 = 27;
/// `GGML_TYPE_F64` — 64-bit IEEE-754 double float.
pub const GGML_TYPE_F64: u32 = 28;
/// `GGML_TYPE_IQ1_M` — i-quant 1-bit medium.
pub const GGML_TYPE_IQ1_M: u32 = 29;
/// `GGML_TYPE_BF16` — Google Brain bfloat16.
pub const GGML_TYPE_BF16: u32 = 30;
/// GGUF wire type 31: historical `Q4_0_4_4` layout (removed from current ggml).
///
/// Must **not** be treated as an IQ3_M block type. HuggingFace “IQ3_M” is a
/// mixed-quant *preset*, not wire id 31. Corinth-canal documents the same
/// mapping (`GGML_TYPE_Q4_0_4_4 = 31`); its 111-byte IQ3_M path is an
/// **internal** non-wire id only.
pub const GGML_TYPE_Q4_0_4_4: u32 = 31;
/// Internal id for the 111-byte IQ3_M *block layout* decoder only.
///
/// Not a GGUF wire type: HuggingFace “IQ3_M” is a mixed-quant *preset*, not
/// type id 31. Production GGUFs never emit this code; synthetic fixtures and
/// [`crate::dequantize_iq3_m`] use it. ASCII `'II3M'`.
pub const GGML_TYPE_IQ3_M_BLOCK: u32 = 0x4949_334D;

// ---------------------------------------------------------------------------
// Human-readable label helper.
// ---------------------------------------------------------------------------

/// Map a raw GGML `ggml_type` `u32` code to a short human-readable label.
///
/// Returns `"unknown"` for codes not in the known set. This is a pure
/// function with no side effects — safe to call from diagnostics, `Debug`
/// impls, or logging.
///
/// # Examples
///
/// ```
/// use engram_parser::ggml_type_label;
/// assert_eq!(ggml_type_label(0), "F32");
/// assert_eq!(ggml_type_label(31), "Q4_0_4_4");
/// assert_eq!(ggml_type_label(9999), "unknown");
/// ```
pub fn ggml_type_label(ggml_type: u32) -> &'static str {
    match ggml_type {
        GGML_TYPE_F32 => "F32",
        GGML_TYPE_F16 => "F16",
        GGML_TYPE_Q4_0 => "Q4_0",
        GGML_TYPE_Q4_1 => "Q4_1",
        GGML_TYPE_Q5_0 => "Q5_0",
        GGML_TYPE_Q5_1 => "Q5_1",
        GGML_TYPE_Q8_0 => "Q8_0",
        GGML_TYPE_Q8_1 => "Q8_1",
        GGML_TYPE_Q2_K => "Q2_K",
        GGML_TYPE_Q3_K => "Q3_K",
        GGML_TYPE_Q4_K => "Q4_K",
        GGML_TYPE_Q5_K => "Q5_K",
        GGML_TYPE_Q6_K => "Q6_K",
        GGML_TYPE_Q8_K => "Q8_K",
        GGML_TYPE_IQ2_XXS => "IQ2_XXS",
        GGML_TYPE_IQ2_XS => "IQ2_XS",
        GGML_TYPE_IQ3_XXS => "IQ3_XXS",
        GGML_TYPE_IQ1_S => "IQ1_S",
        GGML_TYPE_IQ4_NL => "IQ4_NL",
        GGML_TYPE_IQ3_S => "IQ3_S",
        GGML_TYPE_IQ2_S => "IQ2_S",
        GGML_TYPE_IQ4_XS => "IQ4_XS",
        GGML_TYPE_I8 => "I8",
        GGML_TYPE_I16 => "I16",
        GGML_TYPE_I32 => "I32",
        GGML_TYPE_I64 => "I64",
        GGML_TYPE_F64 => "F64",
        GGML_TYPE_IQ1_M => "IQ1_M",
        GGML_TYPE_BF16 => "BF16",
        GGML_TYPE_Q4_0_4_4 => "Q4_0_4_4",
        GGML_TYPE_IQ3_M_BLOCK => "IQ3_M_BLOCK",
        _ => "unknown",
    }
}

// ---------------------------------------------------------------------------
// DType enum.
// ---------------------------------------------------------------------------

/// GGML tensor dtype codes encountered in GGUF checkpoints.
///
/// Values mirror the `GGML_TYPE_*` constants in `ggml.h`. The parser
/// understands their byte layout (for bounds checking) but performs no
/// arithmetic — raw bytes are returned as-is. `BF16` layout parsing is
/// supported even though no BF16→F32 conversion is provided.
///
/// Types not explicitly enumerated are captured by [`DType::Other`]
/// which preserves the raw code for callers to dispatch on.
#[allow(non_camel_case_types)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DType {
    /// 32-bit little-endian float (`GGML_TYPE_F32 = 0`).
    F32,
    /// 16-bit IEEE-754 half float (`GGML_TYPE_F16 = 1`).
    F16,
    /// `GGML_TYPE_Q4_0` — 4-bit symmetric quantization (block size 32).
    Q4_0,
    /// `GGML_TYPE_Q4_1` — 4-bit quantization with min (block size 32).
    Q4_1,
    /// `GGML_TYPE_Q5_0` — 5-bit symmetric quantization (block size 32).
    Q5_0,
    /// `GGML_TYPE_Q5_1` — 5-bit quantization with min (block size 32).
    Q5_1,
    /// `GGML_TYPE_Q8_0` — 8-bit symmetric quantization (block size 32).
    Q8_0,
    /// `GGML_TYPE_Q8_1` — 8-bit quantization with min (block size 32).
    Q8_1,
    /// `GGML_TYPE_Q2_K` — k-quant 2-bit.
    Q2_K,
    /// `GGML_TYPE_Q3_K` — k-quant 3-bit.
    Q3_K,
    /// `GGML_TYPE_Q4_K` — k-quant 4-bit.
    Q4_K,
    /// `GGML_TYPE_Q5_K` — k-quant 5-bit.
    Q5_K,
    /// `GGML_TYPE_Q6_K` — k-quant 6-bit.
    Q6_K,
    /// `GGML_TYPE_Q8_K` — k-quant 8-bit.
    Q8_K,
    /// `GGML_TYPE_IQ2_XXS` — i-quant 2-bit extra-extra-small (wire layout only).
    IQ2_XXS,
    /// `GGML_TYPE_IQ2_XS` — i-quant 2-bit extra-small.
    IQ2_XS,
    /// `GGML_TYPE_IQ3_XXS` — i-quant 3-bit extra-extra-small.
    IQ3_XXS,
    /// `GGML_TYPE_IQ1_S` — i-quant 1-bit small.
    IQ1_S,
    /// `GGML_TYPE_IQ4_NL` — i-quant 4-bit non-linear (block size 32).
    IQ4_NL,
    /// `GGML_TYPE_IQ3_S` — i-quant 3-bit small.
    IQ3_S,
    /// `GGML_TYPE_IQ2_S` — i-quant 2-bit small.
    IQ2_S,
    /// `GGML_TYPE_IQ4_XS` — i-quant 4-bit extra-small.
    IQ4_XS,
    /// `GGML_TYPE_IQ1_M` — i-quant 1-bit medium.
    IQ1_M,
    /// Google Brain bfloat16 (`GGML_TYPE_BF16 = 30`).
    BF16,
    /// 64-bit IEEE-754 double float (`GGML_TYPE_F64 = 28`).
    F64,
    /// 8-bit signed integer (`GGML_TYPE_I8 = 24`).
    I8,
    /// 16-bit signed integer (`GGML_TYPE_I16 = 25`).
    I16,
    /// 32-bit signed integer (`GGML_TYPE_I32 = 26`).
    I32,
    /// 64-bit signed integer (`GGML_TYPE_I64 = 27`).
    I64,
    /// Internal 111-byte IQ3_M block layout ([`GGML_TYPE_IQ3_M_BLOCK`]).
    /// Not GGUF wire type 31.
    IQ3_M_BLOCK,
    /// Any other GGML dtype not explicitly enumerated above. The raw
    /// `u32` code is preserved so callers can dispatch on it.
    Other(u32),
}

impl DType {
    /// Map a raw `ggml_type` code to a [`DType`] enum.
    pub fn from_ggml_type(code: u32) -> Self {
        match code {
            GGML_TYPE_F32 => Self::F32,
            GGML_TYPE_F16 => Self::F16,
            GGML_TYPE_Q4_0 => Self::Q4_0,
            GGML_TYPE_Q4_1 => Self::Q4_1,
            GGML_TYPE_Q5_0 => Self::Q5_0,
            GGML_TYPE_Q5_1 => Self::Q5_1,
            GGML_TYPE_Q8_0 => Self::Q8_0,
            GGML_TYPE_Q8_1 => Self::Q8_1,
            GGML_TYPE_Q2_K => Self::Q2_K,
            GGML_TYPE_Q3_K => Self::Q3_K,
            GGML_TYPE_Q4_K => Self::Q4_K,
            GGML_TYPE_Q5_K => Self::Q5_K,
            GGML_TYPE_Q6_K => Self::Q6_K,
            GGML_TYPE_Q8_K => Self::Q8_K,
            GGML_TYPE_IQ2_XXS => Self::IQ2_XXS,
            GGML_TYPE_IQ2_XS => Self::IQ2_XS,
            GGML_TYPE_IQ3_XXS => Self::IQ3_XXS,
            GGML_TYPE_IQ1_S => Self::IQ1_S,
            GGML_TYPE_IQ4_NL => Self::IQ4_NL,
            GGML_TYPE_IQ3_S => Self::IQ3_S,
            GGML_TYPE_IQ2_S => Self::IQ2_S,
            GGML_TYPE_IQ4_XS => Self::IQ4_XS,
            GGML_TYPE_IQ1_M => Self::IQ1_M,
            GGML_TYPE_IQ3_M_BLOCK => Self::IQ3_M_BLOCK,
            // Wire 31 is historical Q4_0_4_4: fall through to Other(31) via `other`.
            GGML_TYPE_BF16 => Self::BF16,
            GGML_TYPE_F64 => Self::F64,
            GGML_TYPE_I8 => Self::I8,
            GGML_TYPE_I16 => Self::I16,
            GGML_TYPE_I32 => Self::I32,
            GGML_TYPE_I64 => Self::I64,
            other => Self::Other(other),
        }
    }

    /// The raw `ggml_type` code for this dtype.
    pub fn ggml_type(self) -> u32 {
        match self {
            Self::F32 => GGML_TYPE_F32,
            Self::F16 => GGML_TYPE_F16,
            Self::Q4_0 => GGML_TYPE_Q4_0,
            Self::Q4_1 => GGML_TYPE_Q4_1,
            Self::Q5_0 => GGML_TYPE_Q5_0,
            Self::Q5_1 => GGML_TYPE_Q5_1,
            Self::Q8_0 => GGML_TYPE_Q8_0,
            Self::Q8_1 => GGML_TYPE_Q8_1,
            Self::Q2_K => GGML_TYPE_Q2_K,
            Self::Q3_K => GGML_TYPE_Q3_K,
            Self::Q4_K => GGML_TYPE_Q4_K,
            Self::Q5_K => GGML_TYPE_Q5_K,
            Self::Q6_K => GGML_TYPE_Q6_K,
            Self::Q8_K => GGML_TYPE_Q8_K,
            Self::IQ2_XXS => GGML_TYPE_IQ2_XXS,
            Self::IQ2_XS => GGML_TYPE_IQ2_XS,
            Self::IQ3_XXS => GGML_TYPE_IQ3_XXS,
            Self::IQ1_S => GGML_TYPE_IQ1_S,
            Self::IQ4_NL => GGML_TYPE_IQ4_NL,
            Self::IQ3_S => GGML_TYPE_IQ3_S,
            Self::IQ2_S => GGML_TYPE_IQ2_S,
            Self::IQ4_XS => GGML_TYPE_IQ4_XS,
            Self::IQ1_M => GGML_TYPE_IQ1_M,
            Self::IQ3_M_BLOCK => GGML_TYPE_IQ3_M_BLOCK,
            Self::BF16 => GGML_TYPE_BF16,
            Self::F64 => GGML_TYPE_F64,
            Self::I8 => GGML_TYPE_I8,
            Self::I16 => GGML_TYPE_I16,
            Self::I32 => GGML_TYPE_I32,
            Self::I64 => GGML_TYPE_I64,
            Self::Other(code) => code,
        }
    }

    /// Short human-readable label for this dtype (e.g. `"F32"`, `"Q4_K"`).
    ///
    /// Single source of truth: [`ggml_type_label`] on the wire code.
    pub fn label(self) -> &'static str {
        ggml_type_label(self.ggml_type())
    }

    /// Quantization block length along the innermost GGUF dimension, if any.
    ///
    /// Used to reject shapes whose `dims[0]` cannot form complete blocks.
    /// `None` for dense/integer types and unknown/`Other` codes.
    pub fn quant_block_size(self) -> Option<usize> {
        match self {
            Self::Q4_0
            | Self::Q4_1
            | Self::Q5_0
            | Self::Q5_1
            | Self::Q8_0
            | Self::Q8_1
            | Self::IQ4_NL => Some(32),
            Self::Q2_K
            | Self::Q3_K
            | Self::Q4_K
            | Self::Q5_K
            | Self::Q6_K
            | Self::Q8_K
            | Self::IQ2_XXS
            | Self::IQ2_XS
            | Self::IQ2_S
            | Self::IQ3_XXS
            | Self::IQ3_S
            | Self::IQ1_S
            | Self::IQ1_M
            | Self::IQ4_XS
            | Self::IQ3_M_BLOCK => Some(256),
            Self::F32
            | Self::F16
            | Self::BF16
            | Self::F64
            | Self::I8
            | Self::I16
            | Self::I32
            | Self::I64
            | Self::Other(_) => None,
        }
    }

    /// Size in bytes of `n_elements` values of this dtype, or `None`
    /// for quantized/unknown layouts whose byte-length depends on the
    /// tensor's inner dimension (not a simple `n * sizeof(T)`).
    ///
    /// For quantized dtypes we return the correct blocked byte count
    /// when the total element count is divisible by the block size;
    /// otherwise `None`.
    ///
    /// Block sizes follow the GGUF / llama.cpp wire layouts (`ggml-common.h`):
    /// - Q4_0/Q4_1/Q5_0/Q5_1/Q8_0/Q8_1/IQ4_NL: block size 32
    /// - K-quants and most IQ types: block size 256
    /// - Wire type 31 (`Q4_0_4_4`) is **not** modeled: use [`DType::Other`]
    ///
    /// This is layout sizing only — no dequantization.
    pub fn byte_len_for_elements(self, n_elements: usize) -> Option<usize> {
        match self {
            Self::F32 => Some(n_elements.checked_mul(4)?),
            Self::F16 | Self::BF16 => Some(n_elements.checked_mul(2)?),
            Self::F64 | Self::I64 => Some(n_elements.checked_mul(8)?),
            Self::I32 => Some(n_elements.checked_mul(4)?),
            Self::I16 => Some(n_elements.checked_mul(2)?),
            Self::I8 => Some(n_elements),
            // Q*_0/Q*_1 blocked quants: block size 32.
            //   Q4_0: 18, Q4_1: 20, Q5_0: 22, Q5_1: 24, Q8_0: 34, Q8_1: 36
            Self::Q4_0 => blocked_byte_len(n_elements, 32, 18),
            Self::Q4_1 => blocked_byte_len(n_elements, 32, 20),
            Self::Q5_0 => blocked_byte_len(n_elements, 32, 22),
            Self::Q5_1 => blocked_byte_len(n_elements, 32, 24),
            Self::Q8_0 => blocked_byte_len(n_elements, 32, 34),
            Self::Q8_1 => blocked_byte_len(n_elements, 32, 36),
            // K-quants: block size 256.
            //   Q2_K: 84, Q3_K: 110, Q4_K: 144, Q5_K: 176, Q6_K: 210, Q8_K: 292
            Self::Q2_K => blocked_byte_len(n_elements, 256, 84),
            Self::Q3_K => blocked_byte_len(n_elements, 256, 110),
            Self::Q4_K => blocked_byte_len(n_elements, 256, 144),
            Self::Q5_K => blocked_byte_len(n_elements, 256, 176),
            Self::Q6_K => blocked_byte_len(n_elements, 256, 210),
            Self::Q8_K => blocked_byte_len(n_elements, 256, 292),
            // IQ wire layouts (llama.cpp `block_iq*`, QK_K=256 unless noted):
            //   IQ2_XXS: 66, IQ2_XS: 74, IQ2_S: 82
            //   IQ3_XXS: 98, IQ3_S: 110
            //   IQ1_S: 50, IQ1_M: 56
            //   IQ4_NL: block 32 / 18 B, IQ4_XS: 136
            Self::IQ2_XXS => blocked_byte_len(n_elements, 256, 66),
            Self::IQ2_XS => blocked_byte_len(n_elements, 256, 74),
            Self::IQ2_S => blocked_byte_len(n_elements, 256, 82),
            Self::IQ3_XXS => blocked_byte_len(n_elements, 256, 98),
            Self::IQ3_S => blocked_byte_len(n_elements, 256, 110),
            Self::IQ1_S => blocked_byte_len(n_elements, 256, 50),
            Self::IQ1_M => blocked_byte_len(n_elements, 256, 56),
            Self::IQ4_NL => blocked_byte_len(n_elements, 32, 18),
            Self::IQ4_XS => blocked_byte_len(n_elements, 256, 136),
            // Internal IQ3_M block layout (not wire 31): 111 bytes / 256 values.
            Self::IQ3_M_BLOCK => blocked_byte_len(n_elements, 256, 111),
            // Opaque / unknown (includes wire 31 Q4_0_4_4).
            Self::Other(_) => None,
        }
    }

    /// `true` if this crate models the wire layout of this dtype, i.e. every
    /// variant except [`DType::Other`]. For blocked quants the element count
    /// must also be block-aligned before [`Self::byte_len_for_elements`]
    /// returns `Some`.
    pub fn has_known_byte_layout(self) -> bool {
        !matches!(self, Self::Other(_))
    }

    /// Whether the dtype is a plain (non-quantized) float layout.
    ///
    /// Matches the 0.1 surface: `F32`, `F16`, and `BF16` (not `F64`).
    pub fn is_float(self) -> bool {
        matches!(self, Self::F32 | Self::F16 | Self::BF16)
    }

    /// Byte width of a single dense element, or `None` for block-quantized
    /// / integer / unknown dtypes (same contract as 0.1).
    pub fn element_size(self) -> Option<usize> {
        match self {
            Self::F32 => Some(4),
            Self::F16 | Self::BF16 => Some(2),
            _ => None,
        }
    }
}

/// Compute the total byte length for a blocked quantization format.
///
/// Returns `None` if `n_elements` is not divisible by `block_size`
/// or if the multiplication overflows.
fn blocked_byte_len(n_elements: usize, block_size: usize, bytes_per_block: usize) -> Option<usize> {
    if !n_elements.is_multiple_of(block_size) {
        return None;
    }
    let n_blocks = n_elements / block_size;
    n_blocks.checked_mul(bytes_per_block)
}

// ---------------------------------------------------------------------------
// Tensor directory entry.
// ---------------------------------------------------------------------------

/// A single tensor's directory entry: name, shape, dtype, and offsets.
///
/// This is a metadata-only descriptor — it owns no weight data. Use
/// [`GgufLayout::tensor_bytes`](super::layout::GgufLayout::tensor_bytes)
/// to obtain the raw payload.
#[derive(Debug, Clone)]
pub struct Tensor {
    /// Full tensor name as stored in the GGUF directory.
    pub name: String,
    /// Shape dimensions (GGML innermost-first order).
    pub dims: Vec<usize>,
    /// Parsed dtype enum.
    pub dtype: DType,
    /// Raw `ggml_type` code (preserved for round-tripping).
    pub ggml_type: u32,
    /// Total number of elements (product of `dims`).
    pub n_elements: usize,
    /// Total byte length of the tensor payload.
    pub byte_len: usize,
    /// Offset relative to the tensor data region start.
    pub relative_offset: usize,
    /// Absolute byte offset within the file buffer.
    pub absolute_offset: usize,
}

impl Tensor {
    /// Read the raw tensor bytes as little-endian `f32` values.
    ///
    /// Only valid for [`DType::F32`] tensors; returns an error otherwise.
    pub fn read_f32_values(&self, bytes: &[u8]) -> Result<Vec<f32>> {
        if self.dtype != DType::F32 {
            return Err(ParserError::UnsupportedFormat {
                path: self.name.clone(),
                reason: format!("read_f32_values called on dtype {:?}", self.dtype),
            });
        }
        if bytes.len() != self.n_elements * 4 {
            return Err(ParserError::InvalidLayout {
                path: self.name.clone(),
                reason: format!(
                    "f32 byte-length mismatch: bytes={}, expected={}",
                    bytes.len(),
                    self.n_elements * 4
                ),
            });
        }
        let (chunks, remainder) = bytes.as_chunks::<4>();
        debug_assert!(remainder.is_empty());
        Ok(chunks
            .iter()
            .map(|&chunk| f32::from_le_bytes(chunk))
            .collect())
    }

    /// Decode F16/BF16 tensor bytes into raw `u16` lane values using
    /// little-endian chunk parsing (no numeric conversion).
    pub fn read_u16_values(&self, bytes: &[u8]) -> Result<Vec<u16>> {
        if !matches!(self.dtype, DType::F16 | DType::BF16) {
            return Err(ParserError::UnsupportedFormat {
                path: self.name.clone(),
                reason: format!("read_u16_values called on dtype {:?}", self.dtype),
            });
        }
        if bytes.len() != self.n_elements * 2 {
            return Err(ParserError::InvalidLayout {
                path: self.name.clone(),
                reason: format!(
                    "16-bit byte-length mismatch: bytes={}, expected={}",
                    bytes.len(),
                    self.n_elements * 2
                ),
            });
        }
        let (chunks, remainder) = bytes.as_chunks::<2>();
        debug_assert!(remainder.is_empty());
        Ok(chunks
            .iter()
            .map(|&chunk| u16::from_le_bytes(chunk))
            .collect())
    }

    /// Decode an F16 tensor into a newly-allocated `Vec<f32>`. The only
    /// numeric conversion exposed by this crate — purely a bit-level
    /// reinterpretation of each 16-bit half into a 32-bit float.
    pub fn dequantize_f16(&self, bytes: &[u8]) -> Result<Vec<f32>> {
        if self.dtype != DType::F16 {
            return Err(ParserError::UnsupportedFormat {
                path: self.name.clone(),
                reason: format!("dequantize_f16 called on dtype {:?}", self.dtype),
            });
        }
        if bytes.len() != self.n_elements * 2 {
            return Err(ParserError::InvalidLayout {
                path: self.name.clone(),
                reason: format!(
                    "f16 byte-length mismatch: bytes={}, expected={}",
                    bytes.len(),
                    self.n_elements * 2
                ),
            });
        }
        let (chunks, remainder) = bytes.as_chunks::<2>();
        debug_assert!(remainder.is_empty());
        let out: Vec<f32> = chunks
            .iter()
            .map(|&chunk| f16_bits_to_f32(u16::from_le_bytes(chunk)))
            .collect();
        Ok(out)
    }
}

/// Convert a 16-bit IEEE-754 half-precision float (as raw bits) into a
/// 32-bit float. Pure bit manipulation — no external math library.
pub fn f16_bits_to_f32(bits: u16) -> f32 {
    let sign = ((bits as u32) & 0x8000) << 16;
    let exp = ((bits as u32) & 0x7C00) >> 10;
    let mant = ((bits as u32) & 0x03FF) << 13;
    f32::from_bits(sign | f16_payload_bits(exp, mant))
}

fn f16_payload_bits(exp: u32, mant: u32) -> u32 {
    match exp {
        0 => mant,
        31 => 0x7F800000 | mant,
        biased => ((biased + 127 - 15) << 23) | mant,
    }
}

// ---------------------------------------------------------------------------
// Unit tests.
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dtype_round_trips_through_ggml_type() {
        let variants = [
            DType::F32,
            DType::F16,
            DType::Q4_0,
            DType::Q4_1,
            DType::Q5_0,
            DType::Q5_1,
            DType::Q8_0,
            DType::Q8_1,
            DType::Q2_K,
            DType::Q3_K,
            DType::Q4_K,
            DType::Q5_K,
            DType::Q6_K,
            DType::Q8_K,
            DType::IQ2_XXS,
            DType::IQ2_XS,
            DType::IQ3_XXS,
            DType::IQ1_S,
            DType::IQ4_NL,
            DType::IQ3_S,
            DType::IQ2_S,
            DType::IQ4_XS,
            DType::IQ1_M,
            DType::IQ3_M_BLOCK,
            DType::BF16,
            DType::F64,
            DType::I8,
            DType::I16,
            DType::I32,
            DType::I64,
        ];
        for dt in variants {
            let code = dt.ggml_type();
            let back = DType::from_ggml_type(code);
            assert_eq!(dt, back, "round-trip failed for {dt:?} (code={code})");
        }
    }

    #[test]
    fn unknown_code_becomes_other() {
        let dt = DType::from_ggml_type(9999);
        let checks = [
            (dt == DType::Other(9999), "variant"),
            (dt.ggml_type() == 9999, "ggml_type"),
            (dt.byte_len_for_elements(100).is_none(), "byte_len"),
            (!dt.has_known_byte_layout(), "layout"),
        ];
        for (ok, label) in checks {
            assert!(ok, "unknown code 9999: {label}");
        }
    }

    #[test]
    fn ggml_type_label_known_codes() {
        let cases = [
            (GGML_TYPE_F32, "F32"),
            (GGML_TYPE_F16, "F16"),
            (GGML_TYPE_Q4_0, "Q4_0"),
            (GGML_TYPE_Q8_0, "Q8_0"),
            (GGML_TYPE_Q2_K, "Q2_K"),
            (GGML_TYPE_Q6_K, "Q6_K"),
            (GGML_TYPE_IQ3_S, "IQ3_S"),
            (GGML_TYPE_Q4_0_4_4, "Q4_0_4_4"),
            (GGML_TYPE_IQ3_M_BLOCK, "IQ3_M_BLOCK"),
            (GGML_TYPE_BF16, "BF16"),
            (GGML_TYPE_F64, "F64"),
            (GGML_TYPE_I8, "I8"),
            (GGML_TYPE_I16, "I16"),
            (GGML_TYPE_I32, "I32"),
            (GGML_TYPE_I64, "I64"),
            (GGML_TYPE_IQ1_M, "IQ1_M"),
            (GGML_TYPE_IQ1_S, "IQ1_S"),
            (GGML_TYPE_IQ2_XXS, "IQ2_XXS"),
            (GGML_TYPE_IQ2_XS, "IQ2_XS"),
            (GGML_TYPE_IQ2_S, "IQ2_S"),
            (GGML_TYPE_IQ3_XXS, "IQ3_XXS"),
            (GGML_TYPE_IQ4_NL, "IQ4_NL"),
            (GGML_TYPE_IQ4_XS, "IQ4_XS"),
            (GGML_TYPE_Q4_1, "Q4_1"),
            (GGML_TYPE_Q5_0, "Q5_0"),
            (GGML_TYPE_Q5_1, "Q5_1"),
            (GGML_TYPE_Q8_1, "Q8_1"),
            (GGML_TYPE_Q3_K, "Q3_K"),
            (GGML_TYPE_Q4_K, "Q4_K"),
            (GGML_TYPE_Q5_K, "Q5_K"),
            (GGML_TYPE_Q8_K, "Q8_K"),
        ];
        for (code, expected) in cases {
            assert_eq!(ggml_type_label(code), expected, "label for code {code}");
        }
    }

    #[test]
    fn ggml_type_label_unknown() {
        assert_eq!(ggml_type_label(9999), "unknown");
        assert_eq!(ggml_type_label(u32::MAX), "unknown");
    }

    #[test]
    fn dtype_label_matches_ggml_type_label() {
        let variants = [
            DType::F32,
            DType::F16,
            DType::Q4_0,
            DType::Q8_0,
            DType::IQ3_S,
            DType::BF16,
            DType::F64,
            DType::I8,
            DType::I16,
            DType::I32,
            DType::I64,
        ];
        for dt in variants {
            assert_eq!(
                dt.label(),
                ggml_type_label(dt.ggml_type()),
                "label mismatch for {dt:?}"
            );
        }
    }

    #[test]
    fn dtype_label_other_delegates() {
        let dt = DType::Other(9999);
        assert_eq!(dt.label(), "unknown");

        // An Other wrapping a known code should return the known label.
        let dt2 = DType::Other(GGML_TYPE_IQ4_NL);
        assert_eq!(dt2.label(), "IQ4_NL");
    }

    #[test]
    fn byte_len_for_simple_types() {
        let cases = [
            (DType::F32, 100, Some(400)),
            (DType::F16, 100, Some(200)),
            (DType::BF16, 100, Some(200)),
            (DType::F64, 10, Some(80)),
            (DType::I8, 10, Some(10)),
            (DType::I16, 10, Some(20)),
            (DType::I32, 10, Some(40)),
            (DType::I64, 10, Some(80)),
        ];
        for (dt, n, expected) in cases {
            assert_eq!(dt.byte_len_for_elements(n), expected, "{dt:?} x {n}");
        }
    }

    #[test]
    fn byte_len_for_blocked_quants() {
        let cases = [
            // Q4_0: block_size=32, 18 bytes per block
            (DType::Q4_0, 32, Some(18)),
            (DType::Q4_0, 64, Some(36)),
            (DType::Q4_0, 33, None),
            // Q8_0: block_size=32, 34 bytes per block
            (DType::Q8_0, 32, Some(34)),
            // Q4_K: block_size=256, 144 bytes per block
            (DType::Q4_K, 256, Some(144)),
            (DType::Q4_K, 128, None),
            // Q6_K: block_size=256, 210 bytes per block
            (DType::Q6_K, 256, Some(210)),
            // Q8_K: block_size=256, 292 bytes per block
            (DType::Q8_K, 256, Some(292)),
        ];
        for (dt, n, expected) in cases {
            assert_eq!(dt.byte_len_for_elements(n), expected, "{dt:?} x {n}");
        }
    }

    #[test]
    fn byte_len_for_iq_quants() {
        // Wire layouts from llama.cpp ggml-common.h (QK_K=256).
        let cases = [
            (DType::IQ2_XXS, 256, Some(66)),
            (DType::IQ2_XS, 256, Some(74)),
            (DType::IQ2_S, 256, Some(82)),
            (DType::IQ3_XXS, 256, Some(98)),
            (DType::IQ3_S, 256, Some(110)),
            (DType::IQ3_S, 512, Some(220)),
            (DType::IQ3_S, 100, None),
            (DType::IQ1_S, 256, Some(50)),
            (DType::IQ1_M, 256, Some(56)),
            (DType::IQ4_NL, 32, Some(18)),
            (DType::IQ4_NL, 64, Some(36)),
            (DType::IQ4_NL, 33, None),
            (DType::IQ4_XS, 256, Some(136)),
            (DType::IQ3_M_BLOCK, 256, Some(111)),
            (DType::IQ3_M_BLOCK, 512, Some(222)),
            (DType::IQ3_M_BLOCK, 100, None),
        ];
        for (dt, n, expected) in cases {
            assert_eq!(dt.byte_len_for_elements(n), expected, "{dt:?} x {n}");
        }
    }

    #[test]
    fn has_known_byte_layout_check() {
        let cases = [
            (DType::F32, true),
            (DType::Q8_0, true),
            (DType::Q4_K, true),
            (DType::IQ3_S, true),
            (DType::IQ2_XXS, true),
            (DType::IQ4_NL, true),
            (DType::IQ3_M_BLOCK, true),
            (DType::Other(31), false),
            (DType::Other(99), false),
        ];
        for (dt, expected) in cases {
            assert_eq!(dt.has_known_byte_layout(), expected, "{dt:?}");
        }
    }

    #[test]
    fn ggml_type_constants_match_values() {
        let cases = [
            (GGML_TYPE_F32, 0),
            (GGML_TYPE_F16, 1),
            (GGML_TYPE_Q4_0, 2),
            (GGML_TYPE_Q4_1, 3),
            (GGML_TYPE_Q5_0, 6),
            (GGML_TYPE_Q5_1, 7),
            (GGML_TYPE_Q8_0, 8),
            (GGML_TYPE_Q8_1, 9),
            (GGML_TYPE_Q2_K, 10),
            (GGML_TYPE_Q3_K, 11),
            (GGML_TYPE_Q4_K, 12),
            (GGML_TYPE_Q5_K, 13),
            (GGML_TYPE_Q6_K, 14),
            (GGML_TYPE_Q8_K, 15),
            (GGML_TYPE_IQ2_XXS, 16),
            (GGML_TYPE_IQ2_XS, 17),
            (GGML_TYPE_IQ3_XXS, 18),
            (GGML_TYPE_IQ1_S, 19),
            (GGML_TYPE_IQ4_NL, 20),
            (GGML_TYPE_IQ3_S, 21),
            (GGML_TYPE_IQ2_S, 22),
            (GGML_TYPE_IQ4_XS, 23),
            (GGML_TYPE_I8, 24),
            (GGML_TYPE_I16, 25),
            (GGML_TYPE_I32, 26),
            (GGML_TYPE_I64, 27),
            (GGML_TYPE_F64, 28),
            (GGML_TYPE_IQ1_M, 29),
            (GGML_TYPE_BF16, 30),
            (GGML_TYPE_Q4_0_4_4, 31),
            (GGML_TYPE_IQ3_M_BLOCK, 0x4949_334D),
        ];
        for (constant, expected) in cases {
            assert_eq!(constant, expected, "constant value mismatch");
        }
    }

    #[test]
    fn wire_type_31_is_q4_0_4_4_not_iq3_m() {
        let dt = DType::from_ggml_type(31);
        let label = ggml_type_label(31);
        assert!(
            GGML_TYPE_Q4_0_4_4 == 31
                && label == "Q4_0_4_4"
                && label != "IQ3_M"
                && dt == DType::Other(31)
                && dt.byte_len_for_elements(256).is_none()
                && !dt.has_known_byte_layout(),
            "wire 31 semantics: label={label}, dt={dt:?}"
        );
        let iq3 = DType::from_ggml_type(GGML_TYPE_IQ3_M_BLOCK);
        assert!(
            iq3 == DType::IQ3_M_BLOCK
                && ggml_type_label(GGML_TYPE_IQ3_M_BLOCK) == "IQ3_M_BLOCK"
                && iq3.byte_len_for_elements(256) == Some(111)
                && iq3.has_known_byte_layout(),
            "IQ3_M_BLOCK must stay distinct from wire 31: {iq3:?}"
        );
    }

    #[test]
    fn f16_to_f32_known_values() {
        let finite = [(0x0000, 0.0), (0x3C00, 1.0), (0xBC00, -1.0)];
        for (bits, expected) in finite {
            assert_eq!(f16_bits_to_f32(bits), expected, "bits {bits:#06X}");
        }

        for (bits, positive) in [(0x7C00, true), (0xFC00, false)] {
            let v = f16_bits_to_f32(bits);
            assert!(
                v.is_infinite() && v.is_sign_positive() == positive,
                "bits {bits:#06X}: {v}"
            );
        }
    }
}
