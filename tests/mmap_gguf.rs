// SPDX-License-Identifier: MIT OR Apache-2.0

//! mmap-backed GGUF reader vs owned `load_gguf` / `parse_bytes`.
//!
//! Covers tiny-file parity, packed K-quant dequant from the mapping,
//! a tensor that starts past the first OS page, and a sparse multi-GiB
//! file that is mapped without `fs::read`.

#![cfg(feature = "mmap")]

mod common;
use common::*;

use engram_parser::{
    DType, dequantize_iq3_m, dequantize_packed, dequantize_q5_k, dequantize_q6_k, dequantize_q8_0,
    load_gguf, load_gguf_mmap, os_page_size, parse_bytes,
};
use std::fs;
use std::fs::OpenOptions;
use std::os::unix::fs::MetadataExt;
use std::path::{Path, PathBuf};
use std::process;
use std::time::{SystemTime, UNIX_EPOCH};

/// 2 GiB sparse payload — large enough to be a multi-GB checkpoint stand-in
/// without allocating RSS (ext4 hole).
const SPARSE_PAYLOAD_BYTES: u64 = 2 * 1024 * 1024 * 1024;

fn write_temp_gguf(bytes: &[u8]) -> PathBuf {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("clock")
        .as_nanos();
    let path =
        std::env::temp_dir().join(format!("engram-parser-mmap-{}-{nanos}.gguf", process::id()));
    fs::write(&path, bytes).expect("write temp gguf");
    path
}

fn f16_one() -> [u8; 2] {
    0x3C00u16.to_le_bytes()
}

fn q8_0_ones_block() -> Vec<u8> {
    let mut block = vec![0u8; 34];
    block[0..2].copy_from_slice(&f16_one());
    for q in &mut block[2..] {
        *q = 1;
    }
    block
}

fn q5_k_ones_block() -> Vec<u8> {
    let mut block = vec![0u8; 176];
    block[0..2].copy_from_slice(&f16_one());
    for b in &mut block[4..16] {
        *b = 0x01;
    }
    for b in &mut block[48..] {
        *b = 0x11;
    }
    block
}

fn q6_k_minus_32_block() -> Vec<u8> {
    let mut block = vec![0u8; 210];
    for b in &mut block[192..208] {
        *b = 1;
    }
    block[208..210].copy_from_slice(&f16_one());
    block
}

fn iq3_m_scaled_block() -> Vec<u8> {
    let mut block = vec![0u8; 111];
    block[0..2].copy_from_slice(&f16_one());
    for b in &mut block[34..98] {
        *b = 0x55;
    }
    block[98] = 0x01;
    block
}

/// Header-only GGUF whose F32 payload is a sparse hole of `n_elements * 4` bytes.
fn write_sparse_f32_gguf(path: &Path, n_elements: u64) -> u64 {
    let mut out = Vec::new();
    out.extend_from_slice(&GGUF_MAGIC);
    push_u32(&mut out, GGUF_VERSION);
    push_u64(&mut out, 1);
    push_u64(&mut out, 2);
    push_kv_u32(&mut out, "general.alignment", ALIGNMENT);
    push_kv_string(&mut out, "general.architecture", "olmoe");
    push_string(&mut out, "sparse.weight");
    push_u32(&mut out, 1);
    push_u64(&mut out, n_elements);
    push_u32(&mut out, GGML_F32);
    push_u64(&mut out, 0);
    while !out.len().is_multiple_of(ALIGNMENT as usize) {
        out.push(0);
    }
    let header_len = out.len() as u64;
    let payload = n_elements
        .checked_mul(4)
        .expect("sparse payload byte-length overflow");
    let file_len = header_len
        .checked_add(payload)
        .expect("sparse file length overflow");
    fs::write(path, &out).expect("write sparse header");
    let file = OpenOptions::new()
        .write(true)
        .open(path)
        .expect("open sparse gguf");
    file.set_len(file_len).expect("ftruncate sparse gguf");
    file_len
}

#[test]
fn mmap_matches_owned_parse_on_tiny_gguf() {
    let kv = [
        ("general.alignment", KvValue::U32(ALIGNMENT)),
        ("general.architecture", KvValue::Str("olmoe")),
        ("olmoe.expert_count", KvValue::U32(4)),
    ];
    let payload = f32_vec_to_le_bytes(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
    let tensors = [TensorSpec {
        name: "token_embd.weight",
        dims: vec![4, 2],
        ggml_type: GGML_F32,
        payload,
    }];
    let bytes = build_gguf(&kv, &tensors);
    let path = write_temp_gguf(&bytes);
    let _guard = TempGuard(path.clone());

    let owned_mem = parse_bytes(bytes.clone(), "mem://mmap-parity".into()).expect("parse_bytes");
    let owned_fs = load_gguf(&path).expect("load_gguf");
    let mapped = load_gguf_mmap(&path).expect("load_gguf_mmap");

    let name = "token_embd.weight";
    let t_mem = &owned_mem.tensors[name];
    let t_fs = &owned_fs.tensors[name];
    let t_map = &mapped.tensors[name];

    let b_mem = owned_mem.tensor_bytes(t_mem).expect("mem bytes");
    let b_fs = owned_fs.tensor_bytes(t_fs).expect("fs bytes");
    let b_map = mapped.tensor_bytes(t_map).expect("mmap bytes");
    let pages = mapped.tensor_page_aligned_bytes(t_map).expect("page range");

    assert_all(&[
        (
            mapped.directory_matches(&owned_fs),
            "directory vs load_gguf",
        ),
        (
            mapped.directory_matches(&owned_mem),
            "directory vs parse_bytes",
        ),
        (t_mem.dims == t_map.dims, "dims"),
        (t_mem.ggml_type == t_map.ggml_type, "ggml_type"),
        (t_mem.byte_len == t_map.byte_len, "byte_len"),
        (t_mem.absolute_offset == t_map.absolute_offset, "offset"),
        (b_mem == b_fs, "owned mem vs fs"),
        (b_mem == b_map, "owned vs mmap payload"),
        (pages.tensor_bytes() == b_map, "page-aligned payload"),
        (pages.page_size > 0, "page size"),
        (
            (pages.pages.as_ptr() as usize).is_multiple_of(pages.page_size)
                || t_map.absolute_offset < pages.page_size,
            "page start aligned or first-page file",
        ),
    ]);
}

#[test]
fn mmap_directory_matches_rejects_different_tensor_names() {
    let kv = [
        ("general.alignment", KvValue::U32(ALIGNMENT)),
        ("general.architecture", KvValue::Str("olmoe")),
    ];
    let payload = f32_vec_to_le_bytes(&[1.0, 2.0, 3.0, 4.0]);
    let a_bytes = build_gguf(
        &kv,
        &[TensorSpec {
            name: "token_embd.weight",
            dims: vec![4],
            ggml_type: GGML_F32,
            payload: payload.clone(),
        }],
    );
    let b_bytes = build_gguf(
        &kv,
        &[TensorSpec {
            name: "output.weight",
            dims: vec![4],
            ggml_type: GGML_F32,
            payload,
        }],
    );
    let path = write_temp_gguf(&a_bytes);
    let _guard = TempGuard(path.clone());
    let mapped = load_gguf_mmap(&path).expect("mmap a");
    let owned_b = parse_bytes(b_bytes, "mem://other".into()).expect("parse b");
    assert!(
        !mapped.directory_matches(&owned_b),
        "same count/arch but different tensor names must not match"
    );
}

#[test]
fn mmap_dequant_matches_owned_for_k_quants() {
    let page = os_page_size().max(4096);
    let pad_elems = page;
    let pad = vec![0u8; pad_elems * 4];
    let q8 = q8_0_ones_block();
    let q5 = q5_k_ones_block();
    let q6 = q6_k_minus_32_block();
    let iq3 = iq3_m_scaled_block();
    let kv = [
        ("general.alignment", KvValue::U32(ALIGNMENT)),
        ("general.architecture", KvValue::Str("olmoe")),
    ];
    let tensors = [
        TensorSpec {
            name: "pad.weight",
            dims: vec![pad_elems],
            ggml_type: GGML_F32,
            payload: pad,
        },
        TensorSpec {
            name: "q8.weight",
            dims: vec![32],
            ggml_type: GGML_Q8_0,
            payload: q8.clone(),
        },
        TensorSpec {
            name: "q5.weight",
            dims: vec![256],
            ggml_type: GGML_Q5_K,
            payload: q5.clone(),
        },
        TensorSpec {
            name: "q6.weight",
            dims: vec![256],
            ggml_type: GGML_Q6_K,
            payload: q6.clone(),
        },
        TensorSpec {
            name: "iq3.weight",
            dims: vec![256],
            ggml_type: GGML_IQ3_M_BLOCK,
            payload: iq3.clone(),
        },
    ];
    let bytes = build_gguf(&kv, &tensors);
    let path = write_temp_gguf(&bytes);
    let _guard = TempGuard(path.clone());

    let owned = load_gguf(&path).expect("load_gguf");
    let mapped = load_gguf_mmap(&path).expect("load_gguf_mmap");
    assert!(
        mapped.directory_matches(&owned),
        "mmap directory must match owned load"
    );

    let q8_t = mapped.tensor("q8.weight").expect("q8 tensor");
    assert!(
        q8_t.absolute_offset >= page,
        "q8 payload should start past the first OS page (offset {}, page {page})",
        q8_t.absolute_offset
    );
    let q8_pages = mapped
        .tensor_page_aligned_bytes(q8_t)
        .expect("q8 page range");
    assert!(
        (q8_pages.pages.as_ptr() as usize).is_multiple_of(q8_pages.page_size),
        "page-aligned range must start on an OS page"
    );

    let q8_map = mapped.tensor_bytes(q8_t).expect("q8 mmap");
    let q5_map = mapped
        .tensor_bytes(mapped.tensor("q5.weight").unwrap())
        .expect("q5 mmap");
    let q6_map = mapped
        .tensor_bytes(mapped.tensor("q6.weight").unwrap())
        .expect("q6 mmap");
    let iq3_map = mapped
        .tensor_bytes(mapped.tensor("iq3.weight").unwrap())
        .expect("iq3 mmap");

    let q8_owned = owned
        .tensor_bytes(owned.tensor("q8.weight").unwrap())
        .expect("q8 owned");
    let q5_owned = owned
        .tensor_bytes(owned.tensor("q5.weight").unwrap())
        .expect("q5 owned");
    let q6_owned = owned
        .tensor_bytes(owned.tensor("q6.weight").unwrap())
        .expect("q6 owned");
    let iq3_owned = owned
        .tensor_bytes(owned.tensor("iq3.weight").unwrap())
        .expect("iq3 owned");

    assert_eq!(q8_map, q8_owned);
    assert_eq!(q5_map, q5_owned);
    assert_eq!(q6_map, q6_owned);
    assert_eq!(iq3_map, iq3_owned);
    assert_eq!(q8_pages.tensor_bytes(), q8_map);

    let q8_f = dequantize_q8_0(q8_map, &[32]).expect("q8 dequant");
    let q5_f = dequantize_q5_k(q5_map, &[256]).expect("q5 dequant");
    let q6_f = dequantize_q6_k(q6_map, &[256]).expect("q6 dequant");
    let iq3_f = dequantize_iq3_m(iq3_map, &[256]).expect("iq3 dequant");

    assert_all(&[
        (q8_f.iter().all(|&v| v == 1.0), "q8 ones"),
        (q5_f.iter().all(|&v| v == 1.0), "q5 ones"),
        (q6_f.iter().all(|&v| v == -32.0), "q6 -32"),
        (iq3_f[..16].iter().all(|&v| v == -3.0), "iq3 first scale"),
        (iq3_f[16..].iter().all(|&v| v == 0.0), "iq3 rest zero"),
        (
            dequantize_packed(DType::Q8_0, q8_map, &[32]).unwrap() == q8_f,
            "packed q8",
        ),
        (
            dequantize_packed(DType::Q5_K, q5_map, &[256]).unwrap() == q5_f,
            "packed q5",
        ),
        (
            dequantize_packed(DType::Q6_K, q6_map, &[256]).unwrap() == q6_f,
            "packed q6",
        ),
        (
            dequantize_packed(DType::IQ3_M_BLOCK, iq3_map, &[256]).unwrap() == iq3_f,
            "packed iq3",
        ),
    ]);
}

#[test]
fn mmap_sparse_multi_gib_does_not_require_owned_read() {
    let n_elements = SPARSE_PAYLOAD_BYTES / 4;
    let path = {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock")
            .as_nanos();
        std::env::temp_dir().join(format!(
            "engram-parser-mmap-sparse-{}-{nanos}.gguf",
            process::id()
        ))
    };
    let file_len = write_sparse_f32_gguf(&path, n_elements);
    let _guard = TempGuard(path.clone());

    let allocated = fs::metadata(&path).expect("stat sparse").blocks() * 512;
    if allocated >= 64 * 1024 * 1024 {
        eprintln!(
            "skip sparse multi-GiB mmap: filesystem allocated {allocated} bytes (not a hole)"
        );
        return;
    }

    let mapped = load_gguf_mmap(&path).expect("mmap sparse multi-GiB GGUF");
    assert_eq!(mapped.len() as u64, file_len, "mapped length");
    let tensor = mapped.tensor("sparse.weight").expect("sparse tensor");
    assert_eq!(tensor.byte_len as u64, SPARSE_PAYLOAD_BYTES);
    assert_eq!(tensor.n_elements as u64, n_elements);
    let prefix = mapped.tensor_bytes(tensor).expect("sparse payload view");
    assert_eq!(prefix.len() as u64, SPARSE_PAYLOAD_BYTES);
    assert_eq!(&prefix[..16], &[0u8; 16], "sparse hole reads as zeros");
    let pages = mapped
        .tensor_page_aligned_bytes(tensor)
        .expect("sparse page range");
    assert_eq!(pages.tensor_bytes().len() as u64, SPARSE_PAYLOAD_BYTES);
    assert!(pages.page_size > 0);
}

struct TempGuard(PathBuf);

impl Drop for TempGuard {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.0);
    }
}
