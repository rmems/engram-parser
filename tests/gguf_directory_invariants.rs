// SPDX-License-Identifier: MIT OR Apache-2.0

//! RM-1460 tensor-directory invariants: fail-closed on duplicate names,
//! overlapping payloads, and out-of-bounds (past-EOF / truncated) ranges,
//! while still accepting valid multi-tensor files with alignment gaps.
//!
//! `build_gguf` (tests/common) auto-computes valid, non-overlapping, aligned,
//! in-bounds offsets, so it cannot express malformed directories. These tests
//! hand-build the header + tensor directory from the low-level helpers to
//! place tensors at arbitrary/overlapping/out-of-bounds relative offsets.

mod common;
use common::*;

use engram_parser::{ParserError, parse_bytes};

/// One tensor-directory entry: name, dims, wire dtype, and relative offset
/// (relative to the aligned tensor-data region start).
struct DirEntry {
    name: &'static str,
    dims: Vec<usize>,
    ggml_type: u32,
    relative_offset: u64,
}

/// Hand-build a GGUF with an explicit tensor directory and a chosen total
/// payload length, bypassing `build_gguf`'s automatic valid-offset packing.
///
/// Emits magic + version + tensor_count + kv_count(0), each directory entry
/// (`push_string(name)`, `push_u32(n_dims)`, per-dim `push_u64`,
/// `push_u32(ggml_type)`, `push_u64(relative_offset)`), aligns to
/// `ALIGNMENT`, then appends exactly `payload_len` zero bytes.
fn build_dir_gguf(entries: &[DirEntry], payload_len: usize) -> Vec<u8> {
    let mut out = gguf_header(entries.len() as u64, 0);
    for entry in entries {
        push_string(&mut out, entry.name);
        push_u32(&mut out, entry.dims.len() as u32);
        for &d in &entry.dims {
            push_u64(&mut out, d as u64);
        }
        push_u32(&mut out, entry.ggml_type);
        push_u64(&mut out, entry.relative_offset);
    }
    while !out.len().is_multiple_of(ALIGNMENT as usize) {
        out.push(0);
    }
    out.resize(out.len() + payload_len, 0);
    out
}

fn parse_invalid_reason(bytes: Vec<u8>, label: &str) -> String {
    match parse_bytes(bytes, format!("mem://{label}")).expect_err(label) {
        ParserError::InvalidLayout { reason, .. } => reason,
        other => panic!("{label}: expected InvalidLayout, got {other}"),
    }
}

#[test]
fn duplicate_tensor_names_fail_closed() {
    // Two entries share the name "dup.weight". A HashMap would silently
    // last-win; the directory contract must reject the collision.
    let entries = [
        DirEntry {
            name: "dup.weight",
            dims: vec![1],
            ggml_type: GGML_F32,
            relative_offset: 0,
        },
        DirEntry {
            name: "dup.weight",
            dims: vec![1],
            ggml_type: GGML_F32,
            relative_offset: 32,
        },
    ];
    let reason = parse_invalid_reason(build_dir_gguf(&entries, 64), "dup");
    assert_all(&[
        (reason.contains("duplicate"), "names-duplicate-word"),
        (reason.contains("dup.weight"), "names-the-key"),
    ]);
}

#[test]
fn overlapping_payloads_fail_closed() {
    // An F32 dims [16] tensor (64 bytes, range [0,64)) and an F32 dims [2]
    // tensor (8 bytes) at relative offset 32 (range [32,40)). Both offsets are
    // multiples of the default alignment 32, so the failure is specifically
    // overlap, NOT misalignment. 64 payload bytes are provided so it is also
    // NOT past-EOF.
    let entries = [
        DirEntry {
            name: "a",
            dims: vec![16],
            ggml_type: GGML_F32,
            relative_offset: 0,
        },
        DirEntry {
            name: "b",
            dims: vec![2],
            ggml_type: GGML_F32,
            relative_offset: 32,
        },
    ];
    let reason = parse_invalid_reason(build_dir_gguf(&entries, 64), "overlap");
    assert_all(&[
        (reason.contains("overlap"), "overlap-word"),
        (reason.contains('a'), "names-a"),
        (reason.contains('b'), "names-b"),
    ]);
}

#[test]
fn past_eof_tensor_fails_closed() {
    // A single F32 dims [4] tensor needs 16 bytes at relative offset 0, but
    // only 8 payload bytes are present: the range extends past EOF.
    let entries = [DirEntry {
        name: "trunc.weight",
        dims: vec![4],
        ggml_type: GGML_F32,
        relative_offset: 0,
    }];
    let reason = parse_invalid_reason(build_dir_gguf(&entries, 8), "past-eof");
    assert_all(&[
        (reason.contains("beyond file"), "beyond-file"),
        (reason.contains("trunc.weight"), "names-tensor"),
    ]);
}

#[test]
fn truncated_at_last_byte_fails_at_parse_time() {
    // An otherwise-valid header + directory for one F32 dims [4] tensor
    // (16 bytes) whose file is exactly one byte short of the declared payload.
    // Parse itself must fail (proving inventory/inspect paths never see a
    // "healthy" directory with a truncated payload).
    let entries = [DirEntry {
        name: "last.byte",
        dims: vec![4],
        ggml_type: GGML_F32,
        relative_offset: 0,
    }];
    let mut bytes = build_dir_gguf(&entries, 16);
    bytes.pop(); // drop the final payload byte -> 15 of 16 bytes present
    let reason = parse_invalid_reason(bytes, "truncated");
    assert_all(&[
        (reason.contains("beyond file"), "beyond-file"),
        (reason.contains("last.byte"), "names-tensor"),
    ]);
}

#[test]
fn valid_gapped_multi_tensor_parses() {
    // Two F32 dims [2] tensors (8 bytes each) at relative offsets 0 and 32:
    // a deliberate alignment GAP (bytes [8,32) unused) that legal packing
    // produces. This must parse successfully (guards against over-rejection).
    let entries = [
        DirEntry {
            name: "first",
            dims: vec![2],
            ggml_type: GGML_F32,
            relative_offset: 0,
        },
        DirEntry {
            name: "second",
            dims: vec![2],
            ggml_type: GGML_F32,
            relative_offset: 32,
        },
    ];
    // Payload must cover through the end of `second`: 32 + 8 = 40 bytes.
    let layout = parse_bytes(build_dir_gguf(&entries, 40), "mem://gapped".into())
        .expect("valid gapped multi-tensor file must parse");

    let first = layout.tensor("first").expect("first tensor");
    let second = layout.tensor("second").expect("second tensor");
    let base = layout.tensor_data_offset;
    let first_bytes = layout.tensor_bytes(first).expect("first payload");
    let second_bytes = layout.tensor_bytes(second).expect("second payload");

    assert_all(&[
        (layout.tensors.len() == 2, "two-tensors"),
        (first.byte_len == 8, "first-byte-len"),
        (second.byte_len == 8, "second-byte-len"),
        (first.absolute_offset == base, "first-absolute-offset"),
        (
            second.absolute_offset == base + 32,
            "second-absolute-offset",
        ),
        (first_bytes.len() == 8, "first-bytes-len"),
        (second_bytes.len() == 8, "second-bytes-len"),
    ]);
}

#[test]
fn zero_length_tensor_is_not_overlap() {
    // A zero-length tensor (dims [0] => byte_len 0) shares relative offset 0
    // with a real 8-byte tensor. An empty range cannot overlap, so this must
    // parse per the documented policy in finalize_tensor_offsets.
    let entries = [
        DirEntry {
            name: "empty",
            dims: vec![0],
            ggml_type: GGML_F32,
            relative_offset: 0,
        },
        DirEntry {
            name: "real",
            dims: vec![2],
            ggml_type: GGML_F32,
            relative_offset: 0,
        },
    ];
    let layout = parse_bytes(build_dir_gguf(&entries, 8), "mem://zerolen".into())
        .expect("zero-length tensor must not be treated as an overlap");
    let empty = layout.tensor("empty").expect("empty tensor");
    let real = layout.tensor("real").expect("real tensor");
    assert_all(&[
        (empty.byte_len == 0, "empty-byte-len-zero"),
        (real.byte_len == 8, "real-byte-len"),
        (
            layout
                .tensor_bytes(empty)
                .expect("empty payload")
                .is_empty(),
            "empty-payload-empty",
        ),
    ]);
}

/// Like `build_dir_gguf`, but declares an explicit `general.alignment` KV and
/// pads the header/directory to that alignment before the payload.
fn build_dir_gguf_aligned(entries: &[DirEntry], alignment: u32, payload_len: usize) -> Vec<u8> {
    let mut out = gguf_header(entries.len() as u64, 1);
    push_kv_u32(&mut out, "general.alignment", alignment);
    for entry in entries {
        push_string(&mut out, entry.name);
        push_u32(&mut out, entry.dims.len() as u32);
        for &d in &entry.dims {
            push_u64(&mut out, d as u64);
        }
        push_u32(&mut out, entry.ggml_type);
        push_u64(&mut out, entry.relative_offset);
    }
    while !out.len().is_multiple_of(alignment as usize) {
        out.push(0);
    }
    out.resize(out.len() + payload_len, 0);
    out
}

/// One 8-byte F32 dims [2] tensor at `relative_offset`, with enough payload to
/// cover it (so any rejection is alignment, never past-EOF).
fn one_tensor_at(relative_offset: u64) -> [DirEntry; 1] {
    [DirEntry {
        name: "misaligned.weight",
        dims: vec![2],
        ggml_type: GGML_F32,
        relative_offset,
    }]
}

fn assert_misaligned(reason: &str, offset: u64, alignment: u32) {
    assert_all(&[
        (reason.contains("misaligned.weight"), "names-tensor"),
        (reason.contains(&offset.to_string()), "names-offset"),
        (
            reason.contains(&format!("general.alignment {alignment}")),
            "names-alignment",
        ),
        (reason.contains("multiple"), "says-multiple"),
    ]);
}

#[test]
fn misaligned_offset_at_default_alignment_fails_closed() {
    // No `general.alignment` KV, so the default 32 applies. Relative offset 16
    // is in-bounds and non-overlapping but not a multiple of 32; llama.cpp's
    // HANDCRAFTED_TENSORS_BAD_ALIGN case rejects the same shape.
    let offset = 16;
    let bytes = build_dir_gguf(&one_tensor_at(offset), offset as usize + 8);
    let reason = parse_invalid_reason(bytes, "misaligned-default");
    assert_misaligned(&reason, offset, 32);
}

#[test]
fn misaligned_offset_at_default_alignment_by_one_byte_fails_closed() {
    let offset = 33;
    let bytes = build_dir_gguf(&one_tensor_at(offset), offset as usize + 8);
    let reason = parse_invalid_reason(bytes, "misaligned-plus-one");
    assert_misaligned(&reason, offset, 32);
}

#[test]
fn misaligned_offset_at_custom_alignment_64_fails_closed() {
    // Offset 32 is aligned at the default 32 but NOT at the declared 64, so
    // the declared `general.alignment` (not the default) must drive the check.
    let offset = 32;
    let bytes = build_dir_gguf_aligned(&one_tensor_at(offset), 64, offset as usize + 8);
    let reason = parse_invalid_reason(bytes, "misaligned-64");
    assert_misaligned(&reason, offset, 64);
}

#[test]
fn offset_aligned_at_64_parses_but_fails_at_other_alignments() {
    // 192 = 3 * 64: a multiple of 64 and 32, but not of 128.
    let offset = 192;
    let entries = one_tensor_at(offset);
    let payload_len = offset as usize + 8;

    let layout = parse_bytes(
        build_dir_gguf_aligned(&entries, 64, payload_len),
        "mem://aligned-64".into(),
    )
    .expect("offset aligned at general.alignment 64 must parse");
    let tensor = layout.tensor("misaligned.weight").expect("tensor");
    assert_all(&[
        (layout.alignment == 64, "alignment-64"),
        (
            layout.tensor_data_offset.is_multiple_of(64),
            "data-region-aligned",
        ),
        (tensor.relative_offset == 192, "relative-offset"),
        (
            tensor.absolute_offset == layout.tensor_data_offset + 192,
            "absolute-offset",
        ),
        (
            layout.tensor_bytes(tensor).expect("payload").len() == 8,
            "payload-len",
        ),
    ]);

    // The identical directory under general.alignment 128 is misaligned.
    let reason = parse_invalid_reason(
        build_dir_gguf_aligned(&entries, 128, payload_len),
        "misaligned-128",
    );
    assert_misaligned(&reason, offset, 128);
}

#[test]
fn offset_aligned_at_16_parses_but_fails_at_default_32() {
    // Offset 16 is valid under a declared alignment of 16 and invalid under
    // the default 32 (see the default-alignment rejection test above).
    let offset = 16;
    let layout = parse_bytes(
        build_dir_gguf_aligned(&one_tensor_at(offset), 16, offset as usize + 8),
        "mem://aligned-16".into(),
    )
    .expect("offset aligned at general.alignment 16 must parse");
    assert_eq!(layout.alignment, 16);
}

#[cfg(feature = "mmap")]
mod mmap_parity {
    use super::*;
    use engram_parser::load_gguf_mmap;
    use std::fs;
    use std::path::PathBuf;
    use std::process;
    use std::time::{SystemTime, UNIX_EPOCH};

    struct TempGguf(PathBuf);
    impl Drop for TempGguf {
        fn drop(&mut self) {
            let _ = fs::remove_file(&self.0);
        }
    }

    fn write_temp_gguf(bytes: &[u8]) -> (PathBuf, TempGguf) {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock")
            .as_nanos();
        let path = std::env::temp_dir().join(format!(
            "engram-parser-dirinv-{}-{nanos}.gguf",
            process::id()
        ));
        fs::write(&path, bytes).expect("write temp gguf");
        (path.clone(), TempGguf(path))
    }

    #[test]
    fn mmap_rejects_past_eof_like_owned() {
        // Same past-EOF fixture as the owned path: F32 dims [4] (16 bytes) at
        // relative offset 0 with only 8 payload bytes. load_gguf_mmap shares
        // parse_layout, so it must fail with the same InvalidLayout.
        let entries = [DirEntry {
            name: "trunc.weight",
            dims: vec![4],
            ggml_type: GGML_F32,
            relative_offset: 0,
        }];
        let bytes = build_dir_gguf(&entries, 8);
        let (path, _guard) = write_temp_gguf(&bytes);
        match load_gguf_mmap(&path).expect_err("mmap past-eof") {
            ParserError::InvalidLayout { reason, .. } => {
                assert!(
                    reason.contains("beyond") && reason.contains("trunc.weight"),
                    "mmap past-eof message: {reason}"
                );
            }
            other => panic!("expected InvalidLayout, got {other}"),
        }
    }
}
