// SPDX-License-Identifier: MIT OR Apache-2.0

//! llama.cpp layout audit for RM-2012: IQ3_M is not a tensor block type.

#![allow(deprecated)]

use engram_parser::{
    DType, GGML_TYPE_IQ3_S, dequantize_iq3_m, dequantize_packed, ggml_type_label,
    IQ3_M_DECODER_REMOVED,
};

/// Verified against ggml-org/llama.cpp `ggml-common.h` (`block_iq3_s`): 2 + 64 + 8 + 32 + 4 = 110.
const LLAMA_CPP_BLOCK_IQ3_S_BYTES: usize = 110;

/// Verified: `block_q3_K` is also 110 bytes (hmask + qs + scales + d); not 111.
const LLAMA_CPP_BLOCK_Q3_K_BYTES: usize = 110;

#[test]
fn iq3_s_wire_type_matches_llama_cpp_block_size() {
    assert_eq!(ggml_type_label(GGML_TYPE_IQ3_S), "IQ3_S");
    assert_eq!(
        DType::IQ3_S.byte_len_for_elements(256),
        Some(LLAMA_CPP_BLOCK_IQ3_S_BYTES)
    );
}

#[test]
fn q3_k_block_size_matches_llama_cpp() {
    assert_eq!(
        DType::Q3_K.byte_len_for_elements(256),
        Some(LLAMA_CPP_BLOCK_Q3_K_BYTES)
    );
}

#[test]
fn internal_iq3_m_block_has_no_byte_model_or_dequant() {
    assert!(DType::IQ3_M_BLOCK.byte_len_for_elements(256).is_none());
    assert!(!DType::IQ3_M_BLOCK.has_known_byte_layout());
    assert!(dequantize_iq3_m(&[0u8; 111], &[256]).is_err());
    assert!(dequantize_packed(DType::IQ3_M_BLOCK, &[0u8; 111], &[256]).is_err());
    assert!(IQ3_M_DECODER_REMOVED.contains("block_iq3_s"));
    assert!(IQ3_M_DECODER_REMOVED.contains("110"));
}
