// SPDX-License-Identifier: MIT OR Apache-2.0

//! mmap-backed GGUF reader vs owned `load_gguf` / `parse_bytes` on a tiny file.

#![cfg(feature = "mmap")]

mod common;
use common::*;

use engram_parser::{load_gguf, load_gguf_mmap, parse_bytes};
use std::fs;
use std::path::PathBuf;
use std::process;
use std::time::{SystemTime, UNIX_EPOCH};

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

struct TempGuard(PathBuf);

impl Drop for TempGuard {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.0);
    }
}
