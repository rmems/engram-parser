// SPDX-License-Identifier: MIT OR Apache-2.0

//! Packed-block dequantization for Q4_0, Q4_1, Q2_K, Q3_K, Q4_K, Q8_0, Q5_K, Q6_K, and IQ3_M.
//!
//! These helpers operate on already-sliced packed bytes (owned `Vec<u8>` or
//! mmap-backed `&[u8]`). They perform no CUDA / host-register work.
//!
//! HuggingFace “IQ3_M” is a mixed-quant *preset*, not GGUF wire type 31.
//! Wire 31 remains historical [`super::tensor::GGML_TYPE_Q4_0_4_4`]. The
//! 111-byte IQ3_M block decoder uses the internal non-wire id
//! [`super::tensor::GGML_TYPE_IQ3_M_BLOCK`].

use super::tensor::{
    DType, GGML_TYPE_IQ3_M_BLOCK, GGML_TYPE_Q2_K, GGML_TYPE_Q3_K, GGML_TYPE_Q4_0, GGML_TYPE_Q4_1,
    GGML_TYPE_Q4_K, GGML_TYPE_Q5_K, GGML_TYPE_Q6_K, GGML_TYPE_Q8_0, dequantize_bf16_bytes,
    f16_bits_to_f32,
};
use crate::error::{ParserError, Result};

const Q8_0_BLOCK: usize = 32;
const Q8_0_BYTES: usize = 34;
const Q4_0_BLOCK: usize = 32;
const Q4_0_BYTES: usize = 18;
const Q4_1_BYTES: usize = 20;
const Q2_K_BYTES: usize = 84;
const Q3_K_BYTES: usize = 110;
const Q4_K_BYTES: usize = 144;
const K_BLOCK: usize = 256;
const Q5_K_BYTES: usize = 176;
const Q6_K_BYTES: usize = 210;
const IQ3_M_BYTES: usize = 111;

/// Packed byte length of one row for a blocked quant dtype.
pub fn packed_row_size(ggml_type: u32, width: usize) -> Result<usize> {
    match ggml_type {
        GGML_TYPE_Q4_0 => row_size_for_block(width, Q4_0_BLOCK, Q4_0_BYTES, "Q4_0"),
        GGML_TYPE_Q4_1 => row_size_for_block(width, Q4_0_BLOCK, Q4_1_BYTES, "Q4_1"),
        GGML_TYPE_Q2_K => row_size_for_block(width, K_BLOCK, Q2_K_BYTES, "Q2_K"),
        GGML_TYPE_Q3_K => row_size_for_block(width, K_BLOCK, Q3_K_BYTES, "Q3_K"),
        GGML_TYPE_Q4_K => row_size_for_block(width, K_BLOCK, Q4_K_BYTES, "Q4_K"),
        GGML_TYPE_Q8_0 => row_size_for_block(width, Q8_0_BLOCK, Q8_0_BYTES, "Q8_0"),
        GGML_TYPE_Q5_K => row_size_for_block(width, K_BLOCK, Q5_K_BYTES, "Q5_K"),
        GGML_TYPE_Q6_K => row_size_for_block(width, K_BLOCK, Q6_K_BYTES, "Q6_K"),
        GGML_TYPE_IQ3_M_BLOCK => row_size_for_block(width, K_BLOCK, IQ3_M_BYTES, "IQ3_M"),
        other => Err(unsupported(format!(
            "row-size lookup is not implemented for ggml_type={other}"
        ))),
    }
}

/// Dequantize packed Q4_0 bytes to `f32`.
pub fn dequantize_q4_0(packed: &[u8], dims: &[usize]) -> Result<Vec<f32>> {
    dequantize_rows(
        packed,
        dims,
        Q4_0_BLOCK,
        Q4_0_BYTES,
        "Q4_0",
        dequantize_row_q4_0,
    )
}

/// Dequantize packed Q4_1 bytes to `f32`.
pub fn dequantize_q4_1(packed: &[u8], dims: &[usize]) -> Result<Vec<f32>> {
    dequantize_rows(
        packed,
        dims,
        Q4_0_BLOCK,
        Q4_1_BYTES,
        "Q4_1",
        dequantize_row_q4_1,
    )
}

/// Dequantize packed Q2_K bytes to `f32`.
pub fn dequantize_q2_k(packed: &[u8], dims: &[usize]) -> Result<Vec<f32>> {
    dequantize_rows(
        packed,
        dims,
        K_BLOCK,
        Q2_K_BYTES,
        "Q2_K",
        dequantize_row_q2_k,
    )
}

/// Dequantize packed Q3_K bytes to `f32`.
pub fn dequantize_q3_k(packed: &[u8], dims: &[usize]) -> Result<Vec<f32>> {
    dequantize_rows(
        packed,
        dims,
        K_BLOCK,
        Q3_K_BYTES,
        "Q3_K",
        dequantize_row_q3_k,
    )
}

/// Dequantize packed Q4_K bytes to `f32`.
pub fn dequantize_q4_k(packed: &[u8], dims: &[usize]) -> Result<Vec<f32>> {
    dequantize_rows(
        packed,
        dims,
        K_BLOCK,
        Q4_K_BYTES,
        "Q4_K",
        dequantize_row_q4_k,
    )
}

/// Dequantize packed Q8_0 bytes (block size 32) to `f32`.
///
/// `dims` is GGUF innermost-first; `dims[0]` is the row width.
pub fn dequantize_q8_0(packed: &[u8], dims: &[usize]) -> Result<Vec<f32>> {
    dequantize_rows(
        packed,
        dims,
        Q8_0_BLOCK,
        Q8_0_BYTES,
        "Q8_0",
        dequantize_row_q8_0,
    )
}

/// Dequantize packed Q5_K bytes (block size 256) to `f32`.
pub fn dequantize_q5_k(packed: &[u8], dims: &[usize]) -> Result<Vec<f32>> {
    dequantize_rows(
        packed,
        dims,
        K_BLOCK,
        Q5_K_BYTES,
        "Q5_K",
        dequantize_row_q5_k,
    )
}

/// Dequantize packed Q6_K bytes (block size 256) to `f32`.
pub fn dequantize_q6_k(packed: &[u8], dims: &[usize]) -> Result<Vec<f32>> {
    dequantize_rows(
        packed,
        dims,
        K_BLOCK,
        Q6_K_BYTES,
        "Q6_K",
        dequantize_row_q6_k,
    )
}

/// Dequantize packed IQ3_M *block-layout* bytes (111 bytes / 256 values) to `f32`.
///
/// This is **not** GGUF wire type 31 (`Q4_0_4_4`). See [`GGML_TYPE_IQ3_M_BLOCK`].
pub fn dequantize_iq3_m(packed: &[u8], dims: &[usize]) -> Result<Vec<f32>> {
    dequantize_rows(
        packed,
        dims,
        K_BLOCK,
        IQ3_M_BYTES,
        "IQ3_M",
        dequantize_row_iq3_m,
    )
}

/// Dispatch packed dequant for the dtypes this crate implements.
///
/// Unknown / unimplemented dtypes fail closed. Adding a [`DType`] variant
/// without updating this match is a compile error.
pub fn dequantize_packed(dtype: DType, packed: &[u8], dims: &[usize]) -> Result<Vec<f32>> {
    match dtype {
        DType::Q4_0 => dequantize_q4_0(packed, dims),
        DType::Q4_1 => dequantize_q4_1(packed, dims),
        DType::Q2_K => dequantize_q2_k(packed, dims),
        DType::Q3_K => dequantize_q3_k(packed, dims),
        DType::Q4_K => dequantize_q4_k(packed, dims),
        DType::Q8_0 => dequantize_q8_0(packed, dims),
        DType::Q5_K => dequantize_q5_k(packed, dims),
        DType::Q6_K => dequantize_q6_k(packed, dims),
        DType::IQ3_M_BLOCK => dequantize_iq3_m(packed, dims),
        DType::BF16 => dequantize_bf16_dims(packed, dims),
        other => Err(fail_closed_dequant(other)),
    }
}

fn dequantize_bf16_dims(packed: &[u8], dims: &[usize]) -> Result<Vec<f32>> {
    let (width, n_rows) = width_and_rows(dims)?;
    let n = width
        .checked_mul(n_rows)
        .and_then(|n| n.checked_mul(2))
        .ok_or_else(|| invalid("BF16 element count overflow"))?;
    if packed.len() != n {
        return Err(invalid(format!(
            "BF16 length mismatch: expected {n} bytes for dims {dims:?}, got {}",
            packed.len()
        )));
    }
    dequantize_bf16_bytes(packed)
}

fn fail_closed_dequant(dtype: DType) -> ParserError {
    // Exhaustive: a new `DType` variant that is not handled above must be
    // named here (or in the `dequantize_packed` match) or the crate will not
    // compile. The `never` binding documents the fail-closed default.
    let unknown: DType = dtype;
    match unknown {
        DType::Q4_0
        | DType::Q4_1
        | DType::Q2_K
        | DType::Q3_K
        | DType::Q4_K
        | DType::Q8_0
        | DType::Q5_K
        | DType::Q6_K
        | DType::IQ3_M_BLOCK
        | DType::BF16 => {
            unreachable!("handled by dequantize_packed")
        }
        DType::F32
        | DType::F16
        | DType::Q5_0
        | DType::Q5_1
        | DType::Q8_1
        | DType::Q8_K
        | DType::IQ2_XXS
        | DType::IQ2_XS
        | DType::IQ3_XXS
        | DType::IQ1_S
        | DType::IQ4_NL
        | DType::IQ3_S
        | DType::IQ2_S
        | DType::IQ4_XS
        | DType::IQ1_M
        | DType::F64
        | DType::I8
        | DType::I16
        | DType::I32
        | DType::I64
        | DType::Other(_) => unsupported(format!(
            "no packed dequant for dtype {} (ggml_type={})",
            unknown.label(),
            unknown.ggml_type()
        )),
    }
}

pub(crate) fn dequantize_row_q4_0(row: &[u8], width: usize) -> Result<Vec<f32>> {
    require_multiple(width, Q4_0_BLOCK, "Q4_0")?;
    expect_row_len(row, width, Q4_0_BLOCK, Q4_0_BYTES, "Q4_0")?;
    let mut out = Vec::with_capacity(width);
    let (blocks, _) = row.as_chunks::<Q4_0_BYTES>();
    for block in blocks {
        let d = f16_bits_to_f32(u16::from_le_bytes([block[0], block[1]]));
        let qs = &block[2..];
        // ggml writes 16 low nibbles, then 16 high nibbles.
        out.extend(qs.iter().map(|&q| ((q & 0x0F) as i32 - 8) as f32 * d));
        out.extend(qs.iter().map(|&q| ((q >> 4) as i32 - 8) as f32 * d));
    }
    Ok(out)
}

pub(crate) fn dequantize_row_q4_1(row: &[u8], width: usize) -> Result<Vec<f32>> {
    require_multiple(width, Q4_0_BLOCK, "Q4_1")?;
    expect_row_len(row, width, Q4_0_BLOCK, Q4_1_BYTES, "Q4_1")?;
    let mut out = Vec::with_capacity(width);
    let (blocks, _) = row.as_chunks::<Q4_1_BYTES>();
    for block in blocks {
        let d = f16_bits_to_f32(u16::from_le_bytes([block[0], block[1]]));
        let m = f16_bits_to_f32(u16::from_le_bytes([block[2], block[3]]));
        let qs = &block[4..];
        out.extend(qs.iter().map(|&q| (q & 0x0F) as f32 * d + m));
        out.extend(qs.iter().map(|&q| (q >> 4) as f32 * d + m));
    }
    Ok(out)
}

pub(crate) fn dequantize_row_q2_k(row: &[u8], width: usize) -> Result<Vec<f32>> {
    require_multiple(width, K_BLOCK, "Q2_K")?;
    expect_row_len(row, width, K_BLOCK, Q2_K_BYTES, "Q2_K")?;
    let mut out = Vec::with_capacity(width);
    let (blocks, _) = row.as_chunks::<Q2_K_BYTES>();
    for block in blocks {
        // ggml `block_q2_K`: scales(16) + qs(64) + d(2) + dmin(2).
        let scales = &block[0..16];
        let qs = &block[16..80];
        let d = f16_bits_to_f32(u16::from_le_bytes([block[80], block[81]]));
        let dmin = f16_bits_to_f32(u16::from_le_bytes([block[82], block[83]]));
        let mut is = 0usize;
        for q in qs.as_chunks::<32>().0.iter() {
            for shift in [0u32, 2, 4, 6] {
                for half in q.as_chunks::<16>().0.iter() {
                    let sc = scales[is];
                    is += 1;
                    let dl = d * (sc & 0x0F) as f32;
                    let ml = dmin * (sc >> 4) as f32;
                    out.extend(half.iter().map(|&b| dl * ((b >> shift) & 3) as f32 - ml));
                }
            }
        }
    }
    Ok(out)
}

pub(crate) fn dequantize_row_q3_k(row: &[u8], width: usize) -> Result<Vec<f32>> {
    require_multiple(width, K_BLOCK, "Q3_K")?;
    expect_row_len(row, width, K_BLOCK, Q3_K_BYTES, "Q3_K")?;
    let mut out = Vec::with_capacity(width);
    let (blocks, _) = row.as_chunks::<Q3_K_BYTES>();
    for block in blocks {
        // ggml `block_q3_K`: hmask(32) + qs(64) + scales(12) + d(2).
        let hmask = &block[0..32];
        let qs = &block[32..96];
        let d = f16_bits_to_f32(u16::from_le_bytes([block[108], block[109]]));
        let scales = unpack_q3_k_scales(&block[96..108]);
        let mut is = 0usize;
        let mut m = 1u8;
        for q in qs.as_chunks::<32>().0.iter() {
            for shift in [0u32, 2, 4, 6] {
                for (half, hm) in q
                    .as_chunks::<16>()
                    .0
                    .iter()
                    .zip(hmask.as_chunks::<16>().0.iter())
                {
                    let dl = d * (scales[is] as i32 - 32) as f32;
                    is += 1;
                    out.extend(half.iter().zip(hm).map(|(&b, &h)| {
                        let low = ((b >> shift) & 3) as i32;
                        let q3 = low - if h & m != 0 { 0 } else { 4 };
                        dl * q3 as f32
                    }));
                }
                m <<= 1;
            }
        }
    }
    Ok(out)
}

/// Unpack ggml's 12-byte packed Q3_K scales into 16 raw 6-bit values (bias 32).
fn unpack_q3_k_scales(packed: &[u8]) -> [u8; 16] {
    const K1: u32 = 0x0303_0303;
    const K2: u32 = 0x0f0f_0f0f;
    let word =
        |i: usize| u32::from_le_bytes([packed[i], packed[i + 1], packed[i + 2], packed[i + 3]]);
    let (a0, a1, tmp) = (word(0), word(4), word(8));
    let aux = [
        (a0 & K2) | ((tmp & K1) << 4),
        (a1 & K2) | (((tmp >> 2) & K1) << 4),
        ((a0 >> 4) & K2) | (((tmp >> 4) & K1) << 4),
        ((a1 >> 4) & K2) | (((tmp >> 6) & K1) << 4),
    ];
    let mut sc = [0u8; 16];
    for (i, w) in aux.iter().enumerate() {
        sc[i * 4..i * 4 + 4].copy_from_slice(&w.to_le_bytes());
    }
    sc
}

pub(crate) fn dequantize_row_q4_k(row: &[u8], width: usize) -> Result<Vec<f32>> {
    require_multiple(width, K_BLOCK, "Q4_K")?;
    expect_row_len(row, width, K_BLOCK, Q4_K_BYTES, "Q4_K")?;
    let mut out = Vec::with_capacity(width);
    let (blocks, _) = row.as_chunks::<Q4_K_BYTES>();
    for block in blocks {
        let d = f16_bits_to_f32(u16::from_le_bytes([block[0], block[1]]));
        let dmin = f16_bits_to_f32(u16::from_le_bytes([block[2], block[3]]));
        let scales = &block[4..16];
        let qs = &block[16..];
        for (j, q) in qs.as_chunks::<32>().0.iter().enumerate() {
            let (sc1, m1) = scale_min_k4(2 * j, scales);
            let (sc2, m2) = scale_min_k4(2 * j + 1, scales);
            let (d1, mn1) = (d * sc1 as f32, dmin * m1 as f32);
            let (d2, mn2) = (d * sc2 as f32, dmin * m2 as f32);
            out.extend(q.iter().map(|&b| d1 * (b & 0x0F) as f32 - mn1));
            out.extend(q.iter().map(|&b| d2 * (b >> 4) as f32 - mn2));
        }
    }
    Ok(out)
}

pub(crate) fn dequantize_row_q8_0(row: &[u8], width: usize) -> Result<Vec<f32>> {
    require_multiple(width, Q8_0_BLOCK, "Q8_0")?;
    expect_row_len(row, width, Q8_0_BLOCK, Q8_0_BYTES, "Q8_0")?;
    let mut out = Vec::with_capacity(width);
    let (blocks, _) = row.as_chunks::<Q8_0_BYTES>();
    for block in blocks {
        let d = f16_bits_to_f32(u16::from_le_bytes([block[0], block[1]]));
        for &quant in &block[2..Q8_0_BYTES] {
            out.push((quant as i8) as f32 * d);
        }
    }
    Ok(out)
}

pub(crate) fn dequantize_row_q5_k(row: &[u8], width: usize) -> Result<Vec<f32>> {
    require_multiple(width, K_BLOCK, "Q5_K")?;
    expect_row_len(row, width, K_BLOCK, Q5_K_BYTES, "Q5_K")?;
    let mut out = Vec::with_capacity(width);
    let (blocks, _) = row.as_chunks::<Q5_K_BYTES>();
    for block in blocks {
        let d = f16_bits_to_f32(u16::from_le_bytes([block[0], block[1]]));
        let dmin = f16_bits_to_f32(u16::from_le_bytes([block[2], block[3]]));
        let scales = &block[4..16];
        let qh = &block[16..48];
        let ql = &block[48..Q5_K_BYTES];

        let mut is = 0usize;
        let mut u1 = 1u8;
        let mut u2 = 2u8;

        let (ql_chunks, _) = ql.as_chunks::<32>();
        for ql_chunk in ql_chunks {
            let (sc1, m1) = scale_min_k4(is, scales);
            let (sc2, m2) = scale_min_k4(is + 1, scales);
            let d1 = d * sc1 as f32;
            let mn1 = dmin * m1 as f32;
            let d2 = d * sc2 as f32;
            let mn2 = dmin * m2 as f32;

            // ggml writes 32 low nibbles, then 32 high nibbles (`y[l]`, `y[l+32]`).
            let mut group = [0f32; 64];
            for (lane, &q) in ql_chunk.iter().enumerate() {
                let qh_byte = qh[lane];
                let hi1 = if qh_byte & u1 != 0 { 16 } else { 0 };
                let hi2 = if qh_byte & u2 != 0 { 16 } else { 0 };
                group[lane] = d1 * ((q & 0x0F) + hi1) as f32 - mn1;
                group[lane + 32] = d2 * ((q >> 4) + hi2) as f32 - mn2;
            }
            out.extend_from_slice(&group);

            is += 2;
            u1 <<= 2;
            u2 <<= 2;
        }
    }
    Ok(out)
}

pub(crate) fn dequantize_row_q6_k(row: &[u8], width: usize) -> Result<Vec<f32>> {
    require_multiple(width, K_BLOCK, "Q6_K")?;
    expect_row_len(row, width, K_BLOCK, Q6_K_BYTES, "Q6_K")?;
    let mut out = Vec::with_capacity(width);
    let (blocks, _) = row.as_chunks::<Q6_K_BYTES>();
    for block in blocks {
        // ggml `block_q6_K`: ql(128) + qh(64) + scales(16) + d(2).
        let ql = &block[0..128];
        let qh = &block[128..192];
        let scales = &block[192..208];
        let d = f16_bits_to_f32(u16::from_le_bytes([block[208], block[209]]));
        let mut decoded = [0f32; K_BLOCK];
        for pass in 0..2usize {
            let y = &mut decoded[pass * 128..];
            let base = pass * 64;
            let qh_base = pass * 32;
            let sc_pass_base = pass * 8;
            for l in 0..32 {
                let is = l / 16;
                let q1 = ((ql[base + l] & 0x0F) | ((qh[qh_base + l] & 3) << 4)) as i8 - 32;
                let q2 =
                    ((ql[base + 32 + l] & 0x0F) | (((qh[qh_base + l] >> 2) & 3) << 4)) as i8 - 32;
                let q3 = ((ql[base + l] >> 4) | (((qh[qh_base + l] >> 4) & 3) << 4)) as i8 - 32;
                let q4 =
                    ((ql[base + 32 + l] >> 4) | (((qh[qh_base + l] >> 6) & 3) << 4)) as i8 - 32;
                y[l] = d * (scales[sc_pass_base + is] as i8) as f32 * q1 as f32;
                y[l + 32] = d * (scales[sc_pass_base + is + 2] as i8) as f32 * q2 as f32;
                y[l + 64] = d * (scales[sc_pass_base + is + 4] as i8) as f32 * q3 as f32;
                y[l + 96] = d * (scales[sc_pass_base + is + 6] as i8) as f32 * q4 as f32;
            }
        }
        out.extend_from_slice(&decoded);
    }
    Ok(out)
}

pub(crate) fn dequantize_row_iq3_m(row: &[u8], width: usize) -> Result<Vec<f32>> {
    require_multiple(width, K_BLOCK, "IQ3_M")?;
    expect_row_len(row, width, K_BLOCK, IQ3_M_BYTES, "IQ3_M")?;
    let mut out = Vec::with_capacity(width);
    let (blocks, _) = row.as_chunks::<IQ3_M_BYTES>();
    for block in blocks {
        let d = f16_bits_to_f32(u16::from_le_bytes([block[0], block[1]]));
        let hmask = &block[2..34];
        let qs = &block[34..98];
        let scales = &block[98..110];
        let scales_h = block[110];
        let sc = unpack_iq3_m_scales(scales, scales_h);
        for i in 0..256 {
            let qs_byte = qs[i / 4];
            let qs_shift = (i % 4) * 2;
            let low_2 = (qs_byte >> qs_shift) & 0x03;
            let hmask_byte = hmask[i / 8];
            let high_bit = (hmask_byte >> (i % 8)) & 0x01;
            let q = low_2 | (high_bit << 2);
            let q_signed = q as i8 - 4;
            let scale = sc[i / 16] as f32;
            out.push(d * scale * q_signed as f32);
        }
    }
    Ok(out)
}

fn unpack_iq3_m_scales(scales: &[u8], scales_h: u8) -> [u8; 16] {
    let mut sc = [0u8; 16];
    let mut bit_pos = 0usize;
    for sc_val in &mut sc {
        let byte_idx = bit_pos / 8;
        let bit_shift = bit_pos % 8;
        let mut val = if byte_idx < 12 {
            (scales[byte_idx] >> bit_shift) & 0x3F
        } else {
            0
        };
        if bit_shift > 2 && byte_idx + 1 < 12 {
            let rem = 6 - (8 - bit_shift);
            val |= (scales[byte_idx + 1] & ((1 << rem) - 1)) << (8 - bit_shift);
        }
        *sc_val = val;
        bit_pos += 6;
    }
    let scales_h_u32 = u32::from(scales_h);
    for (i, sc_val) in sc.iter_mut().enumerate() {
        let high_bits = ((scales_h_u32 >> (i * 2)) & 0x03) as u8;
        *sc_val |= high_bits << 6;
    }
    sc
}

fn scale_min_k4(index: usize, scales: &[u8]) -> (u8, u8) {
    if index < 4 {
        (scales[index] & 63, scales[index + 4] & 63)
    } else {
        (
            (scales[index + 4] & 0x0F) | ((scales[index - 4] >> 6) << 4),
            (scales[index + 4] >> 4) | ((scales[index] >> 6) << 4),
        )
    }
}

fn dequantize_rows(
    packed: &[u8],
    dims: &[usize],
    block: usize,
    bytes_per_block: usize,
    label: &str,
    dequant_row: fn(&[u8], usize) -> Result<Vec<f32>>,
) -> Result<Vec<f32>> {
    let (width, n_rows) = width_and_rows(dims)?;
    require_multiple(width, block, label)?;
    let row_bytes = (width / block)
        .checked_mul(bytes_per_block)
        .ok_or_else(|| invalid(format!("{label} row byte-length overflow")))?;
    let expected = n_rows
        .checked_mul(row_bytes)
        .ok_or_else(|| invalid(format!("{label} packed byte-length overflow")))?;
    if packed.len() != expected {
        return Err(invalid(format!(
            "{label} packed length mismatch: expected {expected} bytes for dims {dims:?}, got {}",
            packed.len()
        )));
    }
    let mut out = Vec::with_capacity(
        width
            .checked_mul(n_rows)
            .ok_or_else(|| invalid(format!("{label} element count overflow")))?,
    );
    for row in packed.chunks_exact(row_bytes) {
        out.extend_from_slice(&dequant_row(row, width)?);
    }
    Ok(out)
}

fn width_and_rows(dims: &[usize]) -> Result<(usize, usize)> {
    let Some((&width, rest)) = dims.split_first() else {
        return Err(unsupported("dequant dims must be non-empty"));
    };
    let n_rows = rest
        .iter()
        .try_fold(1usize, |acc, &d| acc.checked_mul(d))
        .ok_or_else(|| invalid("dequant dims overflow"))?;
    Ok((width, n_rows))
}

fn row_size_for_block(
    width: usize,
    block: usize,
    bytes_per_block: usize,
    label: &str,
) -> Result<usize> {
    require_multiple(width, block, label)?;
    (width / block)
        .checked_mul(bytes_per_block)
        .ok_or_else(|| invalid(format!("{label} row byte-length overflow")))
}

fn require_multiple(width: usize, block: usize, label: &str) -> Result<()> {
    if width == 0 {
        return Err(unsupported(format!("{label} width must be positive")));
    }
    if width.is_multiple_of(block) {
        Ok(())
    } else {
        Err(unsupported(format!(
            "{label} width {width} is not divisible by {block}"
        )))
    }
}

fn expect_row_len(
    row: &[u8],
    width: usize,
    block: usize,
    bytes_per_block: usize,
    label: &str,
) -> Result<()> {
    let expected = (width / block) * bytes_per_block;
    if row.len() == expected {
        Ok(())
    } else {
        Err(invalid(format!(
            "{label} row length mismatch: expected {expected} bytes for width {width}, got {}",
            row.len()
        )))
    }
}

fn unsupported(reason: impl Into<String>) -> ParserError {
    ParserError::UnsupportedFormat {
        path: "<dequantize>".into(),
        reason: reason.into(),
    }
}

fn invalid(reason: impl Into<String>) -> ParserError {
    ParserError::InvalidLayout {
        path: "<dequantize>".into(),
        reason: reason.into(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn f16_one() -> [u8; 2] {
        0x3C00u16.to_le_bytes()
    }

    fn f16_two() -> [u8; 2] {
        0x4000u16.to_le_bytes()
    }

    #[test]
    fn q8_0_handmade_block() {
        // scale = 2.0, quants: 0, 1, -1, then zeros.
        let mut block = vec![0u8; Q8_0_BYTES];
        block[0..2].copy_from_slice(&f16_two());
        block[2] = 0;
        block[3] = 1;
        block[4] = (-1i8) as u8;
        let out = dequantize_q8_0(&block, &[32]).expect("q8_0");
        assert_eq!(out.len(), 32);
        assert_eq!(out[0], 0.0);
        assert_eq!(out[1], 2.0);
        assert_eq!(out[2], -2.0);
        assert!(out[3..].iter().all(|&v| v == 0.0));
    }

    #[test]
    fn q8_0_denormal_scale() {
        // Packed-scale denormal path: f16 scale bits 0x0001 = 2^-24.
        // Dequantized output must equal quant_i8 * 2^-24 (RM-1461).
        let scale = 2f32.powi(-24);
        let mut block = vec![0u8; Q8_0_BYTES];
        block[0..2].copy_from_slice(&0x0001u16.to_le_bytes());
        block[2] = 0;
        block[3] = 1;
        block[4] = (-1i8) as u8;
        block[5] = 127;
        block[6] = (-128i8) as u8;
        let out = dequantize_q8_0(&block, &[32]).expect("q8_0 denormal scale");
        assert_eq!(out.len(), 32);
        for (i, &quant) in [0i8, 1, -1, 127, -128].iter().enumerate() {
            assert_eq!(out[i], quant as f32 * scale, "quant {quant} at index {i}");
        }
        assert!(out[5..].iter().all(|&v| v == 0.0));
    }

    #[test]
    fn q8_0_unit_scale() {
        let mut block = vec![0u8; Q8_0_BYTES];
        block[0..2].copy_from_slice(&f16_one());
        for q in &mut block[2..] {
            *q = 1;
        }
        let out = dequantize_q8_0(&block, &[32]).expect("q8_0 ones");
        assert!(out.iter().all(|&v| v == 1.0), "{out:?}");
    }

    #[test]
    fn q5_k_handmade_ones() {
        // Synthetic block: d=1, dmin=0, scales=0x01, ql=0x11, qh=0 → 1.0.
        let mut block = vec![0u8; Q5_K_BYTES];
        block[0..2].copy_from_slice(&f16_one());
        for b in &mut block[4..16] {
            *b = 0x01;
        }
        for b in &mut block[48..] {
            *b = 0x11;
        }
        let out = dequantize_q5_k(&block, &[256]).expect("q5_k");
        assert_eq!(out.len(), 256);
        assert!(out.iter().all(|&v| v == 1.0), "got {:?}", &out[..8]);
    }

    #[test]
    fn q6_k_handmade_minus_32() {
        // ggml layout: ql(128) + qh(64) + scales(16) + d(2). ql/qh=0 → (0-32)*1 = -32.
        let mut block = vec![0u8; Q6_K_BYTES];
        for b in &mut block[192..208] {
            *b = 1;
        }
        block[208..210].copy_from_slice(&f16_one());
        let out = dequantize_q6_k(&block, &[256]).expect("q6_k");
        assert_eq!(out.len(), 256);
        assert!(out.iter().all(|&v| v == -32.0), "got {:?}", &out[..8]);
    }

    #[test]
    fn q5_k_ggml_group_order() {
        // Only ql[0] low nibble is 1. ggml emits y[0]=1, y[32]=0 — not interleaved.
        let mut block = vec![0u8; Q5_K_BYTES];
        block[0..2].copy_from_slice(&f16_one());
        for b in &mut block[4..16] {
            *b = 0x01;
        }
        block[48] = 0x01;
        let out = dequantize_q5_k(&block, &[256]).expect("q5_k order");
        assert_eq!(out[0], 1.0);
        assert_eq!(out[32], 0.0);
        assert!(out[1..32].iter().all(|&v| v == 0.0));
    }

    #[test]
    fn q6_k_ggml_group_order() {
        // ql[0]=1 → q1=-31 at y[0]; y[32]/y[64]/y[96] stay -32.
        let mut block = vec![0u8; Q6_K_BYTES];
        block[0] = 1;
        for b in &mut block[192..208] {
            *b = 1;
        }
        block[208..210].copy_from_slice(&f16_one());
        let out = dequantize_q6_k(&block, &[256]).expect("q6_k order");
        assert_eq!(out[0], -31.0);
        assert_eq!(out[32], -32.0);
        assert_eq!(out[64], -32.0);
        assert_eq!(out[96], -32.0);
    }

    #[test]
    fn iq3_m_zero_scales_vector() {
        let mut block = vec![0u8; IQ3_M_BYTES];
        block[0..2].copy_from_slice(&f16_one());
        for b in &mut block[34..98] {
            *b = 0x55;
        }
        let out = dequantize_iq3_m(&block, &[256]).expect("iq3_m");
        assert_eq!(out.len(), 256);
        assert!(out.iter().all(|&v| v == 0.0), "expected zeros, got {out:?}");
    }

    #[test]
    fn iq3_m_nonzero_first_scale() {
        let mut block = vec![0u8; IQ3_M_BYTES];
        block[0..2].copy_from_slice(&f16_one());
        for b in &mut block[34..98] {
            *b = 0x55; // low-2 = 1 → q_signed = -3
        }
        block[98] = 0x01; // first 6-bit scale = 1
        let out = dequantize_iq3_m(&block, &[256]).expect("iq3_m scaled");
        assert!(out[..16].iter().all(|&v| v == -3.0), "{:?}", &out[..16]);
        assert!(out[16..].iter().all(|&v| v == 0.0));
    }

    // Golden vectors below are hand-derived from the ggml `dequantize_row_*`
    // reference loops (ggml-quants.c), exercising nibble/bit order, scale
    // packing, and block strides.

    #[test]
    fn q4_0_golden_nibble_order() {
        // d=2: qs[0]=0x9A -> y[0]=(10-8)*2=4, y[16]=(9-8)*2=2; qs[1]=0 -> -16/-16.
        let mut block = vec![0x88u8; Q4_0_BYTES];
        block[0..2].copy_from_slice(&f16_two());
        block[2] = 0x9A;
        block[3] = 0x00;
        let out = dequantize_q4_0(&block, &[32]).expect("q4_0");
        assert_eq!(out.len(), 32);
        assert_eq!((out[0], out[16]), (4.0, 2.0));
        assert_eq!((out[1], out[17]), (-16.0, -16.0));
        assert!(
            out.iter()
                .enumerate()
                .all(|(i, &v)| matches!(i, 0 | 1 | 16 | 17) || v == 0.0)
        );
    }

    #[test]
    fn q4_1_golden_nibble_order() {
        // d=2, m=1: qs[0]=0x53 -> y[0]=3*2+1=7, y[16]=5*2+1=11; rest = m.
        let mut block = vec![0u8; Q4_1_BYTES];
        block[0..2].copy_from_slice(&f16_two());
        block[2..4].copy_from_slice(&f16_one());
        block[4] = 0x53;
        let out = dequantize_q4_1(&block, &[32]).expect("q4_1");
        assert_eq!((out[0], out[16]), (7.0, 11.0));
        assert_eq!(out[1], 1.0);
        assert_eq!(out[17], 1.0);
    }

    #[test]
    fn q4_k_golden_scales_and_nibbles() {
        // d=1, dmin=1. Group 0: sc=2, min=3. Group 1: sc=0, min=0.
        // Group 4 (index >= 4 branch): scales[8]=0x21 -> sc=1, min=2.
        let mut block = vec![0u8; Q4_K_BYTES];
        block[0..2].copy_from_slice(&f16_one());
        block[2..4].copy_from_slice(&f16_one());
        block[4] = 2; // scales[0]
        block[8] = 3; // scales[4]
        block[12] = 0x21; // scales[8]
        block[16] = 0x25; // qs[0]: low 5 -> y[0], high 2 -> y[32]
        block[16 + 64] = 0x03; // qs[64]: low 3 -> y[128]
        let out = dequantize_q4_k(&block, &[256]).expect("q4_k");
        assert_eq!(out.len(), 256);
        assert_eq!(out[0], 2.0 * 5.0 - 3.0);
        assert_eq!(out[1], -3.0);
        assert_eq!(out[32], 0.0);
        assert_eq!(out[128], 1.0 * 3.0 - 2.0);
        assert_eq!(out[129], -2.0);
    }

    #[test]
    fn q2_k_golden_scales_and_shifts() {
        // Layout: scales(16) qs(64) d(2) dmin(2); d=dmin=1.
        let mut block = vec![0u8; Q2_K_BYTES];
        block[80..82].copy_from_slice(&f16_one());
        block[82..84].copy_from_slice(&f16_one());
        block[0] = 0x32; // dl=2, ml=3
        block[2] = 0x01; // dl=1, ml=0
        block[4] = 0x10; // dl=0, ml=1
        block[6] = 0x04; // dl=4, ml=0
        block[8] = 0x21; // second 128-chunk: dl=1, ml=2
        block[16] = 0b11_10_01_00; // shifts 0,2,4,6 -> 0,1,2,3
        block[16 + 32] = 0x01;
        let out = dequantize_q2_k(&block, &[256]).expect("q2_k");
        assert_eq!(out.len(), 256);
        assert_eq!(out[0], -3.0);
        assert_eq!(out[1], -3.0);
        assert_eq!(out[16], 0.0);
        assert_eq!(out[32], 1.0);
        assert_eq!(out[64], -1.0);
        assert_eq!(out[96], 12.0);
        assert_eq!(out[128], 1.0 * 1.0 - 2.0);
    }

    #[test]
    fn q3_k_golden_scales_hmask_and_shifts() {
        // Layout: hmask(32) qs(64) scales(12) d(2); d=1.
        let mut block = vec![0u8; Q3_K_BYTES];
        block[108..110].copy_from_slice(&f16_one());
        for b in &mut block[0..32] {
            *b = 0xFF;
        }
        // Packed scales: idx0 raw = 1 | (1<<4) = 17 (scale -15); idx8 raw = 1 (scale -31);
        // everything else raw 0 (scale -32).
        block[96] = 0x11;
        block[96 + 8] = 0x01;
        block[0] = 0x00; // hmask[0]: all high bits clear -> q-4 for lane 0
        block[16] = 0xFE; // hmask[16]: bit 0 clear
        block[2] = 0xEF; // hmask[2]: bit 4 clear (second 128-chunk)
        block[32] = 0x03; // qs[0]: shift0 -> 3
        let out = dequantize_q3_k(&block, &[256]).expect("q3_k");
        assert_eq!(out.len(), 256);
        assert_eq!(out[0], -15.0 * (3.0 - 4.0));
        assert_eq!(out[1], 0.0);
        assert_eq!(out[16], -32.0 * (0.0 - 4.0));
        assert_eq!(out[32], -32.0 * (0.0 - 4.0));
        assert_eq!(out[33], 0.0);
        assert_eq!(out[130], -31.0 * (0.0 - 4.0));
        assert_eq!(out[129], 0.0);
    }

    #[test]
    fn new_quants_dispatch_and_row_sizes() {
        for (dtype, bytes, width) in [
            (DType::Q4_0, Q4_0_BYTES, 32),
            (DType::Q4_1, Q4_1_BYTES, 32),
            (DType::Q2_K, Q2_K_BYTES, 256),
            (DType::Q3_K, Q3_K_BYTES, 256),
            (DType::Q4_K, Q4_K_BYTES, 256),
        ] {
            let ty = dtype.ggml_type();
            assert_eq!(packed_row_size(ty, width).unwrap(), bytes);
            assert_eq!(packed_row_size(ty, width * 2).unwrap(), bytes * 2);
            assert!(packed_row_size(ty, width + 1).is_err());
            assert!(packed_row_size(ty, 0).is_err());
            let zeros = vec![0u8; bytes];
            let out = dequantize_packed(dtype, &zeros, &[width]).expect("dispatch");
            assert_eq!(out.len(), width);
            assert!(out.iter().all(|v| v.is_finite()));
            // Short, long, and non-divisible rows fail closed.
            assert!(dequantize_packed(dtype, &zeros[..bytes - 1], &[width]).is_err());
            let long = vec![0u8; bytes + 1];
            assert!(dequantize_packed(dtype, &long, &[width]).is_err());
            assert!(dequantize_packed(dtype, &zeros, &[width + 1]).is_err());
            assert!(dequantize_packed(dtype, &[], &[]).is_err());
        }
    }

    #[test]
    fn packed_row_size_iq3_m_is_not_wire_31() {
        assert_eq!(packed_row_size(GGML_TYPE_IQ3_M_BLOCK, 256).unwrap(), 111);
        assert_eq!(packed_row_size(GGML_TYPE_IQ3_M_BLOCK, 512).unwrap(), 222);
        assert!(packed_row_size(GGML_TYPE_IQ3_M_BLOCK, 255).is_err());
        assert!(packed_row_size(31, 256).is_err());
    }

    #[test]
    fn dequant_error_paths() {
        assert!(dequantize_q8_0(&[0u8; 34], &[31]).is_err());
        assert!(dequantize_q8_0(&[0u8; 33], &[32]).is_err());
        assert!(dequantize_row_q8_0(&[0u8; 33], 32).is_err());
        assert!(dequantize_row_q5_k(&[0u8; 175], 256).is_err());
        assert!(dequantize_iq3_m(&[0u8; 111], &[255]).is_err());
        assert!(dequantize_iq3_m(&[0u8; 110], &[256]).is_err());
        assert!(dequantize_q5_k(&[0u8; 176], &[]).is_err());
        assert!(dequantize_q8_0(&[], &[0]).is_err());
        assert!(dequantize_packed(DType::F32, &[0u8; 4], &[1]).is_err());
        assert!(dequantize_packed(DType::Other(31), &[0u8; 18], &[32]).is_err());
        assert!(dequantize_packed(DType::Q5_K, &[0u8; 176], &[256]).is_ok());
        assert!(dequantize_packed(DType::Q6_K, &[0u8; 210], &[256]).is_ok());
        assert!(dequantize_packed(DType::IQ3_M_BLOCK, &[0u8; 111], &[256]).is_ok());
    }

    #[test]
    fn dequantize_packed_dispatches_q8_0() {
        let mut block = vec![0u8; Q8_0_BYTES];
        block[0..2].copy_from_slice(&f16_one());
        let out = dequantize_packed(DType::Q8_0, &block, &[32]).unwrap();
        assert_eq!(out.len(), 32);
    }

    #[test]
    fn dequantize_packed_dispatches_bf16() {
        let bytes = [0x80, 0x3F, 0x00, 0xC0];
        assert_eq!(
            dequantize_packed(DType::BF16, &bytes, &[2]).unwrap(),
            vec![1.0, -2.0]
        );
        assert_eq!(
            dequantize_packed(DType::BF16, &bytes, &[1, 2]).unwrap(),
            vec![1.0, -2.0]
        );
        assert!(dequantize_packed(DType::BF16, &bytes, &[3]).is_err());
        assert!(dequantize_packed(DType::BF16, &bytes[..3], &[2]).is_err());
    }
}
