// SPDX-License-Identifier: MIT OR Apache-2.0
#![cfg(all(feature = "safetensors", feature = "mmap"))]

//! Mmap-backed borrowed payload access, including a sparse large shard.

use engram_parser::safetensors::{open_safetensors_checkpoint, open_safetensors_checkpoint_mmap};
use std::fs::{self, File};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

struct TestDir(PathBuf);

impl Drop for TestDir {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

fn temp_dir(name: &str) -> TestDir {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let path = std::env::temp_dir().join(format!(
        "engram-st-mmap-{name}-{}-{nanos}",
        std::process::id()
    ));
    fs::create_dir_all(&path).unwrap();
    TestDir(path)
}

fn write_safetensors(path: &Path, header: &str, payload: &[u8]) {
    let mut file = File::create(path).unwrap();
    file.write_all(&(header.len() as u64).to_le_bytes())
        .unwrap();
    file.write_all(header.as_bytes()).unwrap();
    file.write_all(payload).unwrap();
}

#[test]
fn mmap_returns_borrowed_bytes_matching_owned_reads() {
    let dir = temp_dir("borrowed");
    let path = dir.0.join("model.safetensors");
    let payload: Vec<u8> = (0u8..32).collect();
    write_safetensors(
        &path,
        r#"{"w":{"dtype":"F32","shape":[8],"data_offsets":[0,32]},"b":{"dtype":"U8","shape":[4],"data_offsets":[32,36]}}"#,
        &[payload.clone(), vec![40, 41, 42, 43]].concat(),
    );

    let mapped = open_safetensors_checkpoint_mmap(&path).unwrap();
    let owned = open_safetensors_checkpoint(&path).unwrap();
    assert_eq!(mapped.tensor_bytes("w").unwrap(), payload.as_slice());
    assert_eq!(mapped.tensor_bytes("b").unwrap(), &[40, 41, 42, 43]);
    assert_eq!(
        mapped.tensor_bytes("w").unwrap(),
        owned.tensor_bytes("w").unwrap().as_slice()
    );
}

#[test]
fn mmap_sharded_checkpoint_reads_both_shards() {
    let dir = temp_dir("sharded");
    write_safetensors(
        &dir.0.join("model-00001-of-00002.safetensors"),
        r#"{"a.weight":{"dtype":"U8","shape":[4],"data_offsets":[0,4]}}"#,
        &[1, 2, 3, 4],
    );
    write_safetensors(
        &dir.0.join("model-00002-of-00002.safetensors"),
        r#"{"b.weight":{"dtype":"U8","shape":[4],"data_offsets":[0,4]}}"#,
        &[5, 6, 7, 8],
    );
    fs::write(
        dir.0.join("model.safetensors.index.json"),
        r#"{"metadata":{"total_size":8},"weight_map":{"a.weight":"model-00001-of-00002.safetensors","b.weight":"model-00002-of-00002.safetensors"}}"#,
    )
    .unwrap();

    let mapped =
        open_safetensors_checkpoint_mmap(dir.0.join("model.safetensors.index.json")).unwrap();
    assert_eq!(mapped.tensor_bytes("a.weight").unwrap(), &[1, 2, 3, 4]);
    assert_eq!(mapped.tensor_bytes("b.weight").unwrap(), &[5, 6, 7, 8]);
}

#[test]
fn mmap_sparse_large_shard_maps_without_full_read() {
    // 1 GiB tensor payload in a sparse file: the checkpoint maps the file
    // and borrows the payload range without a Vec copy of the shard.
    let dir = temp_dir("sparse");
    let path = dir.0.join("model.safetensors");
    let elements = 256 * 1024 * 1024usize; // 1 GiB of F32
    let header =
        format!(r#"{{"w":{{"dtype":"F32","shape":[{elements}],"data_offsets":[0,1073741824]}}}}"#);
    {
        let mut file = File::create(&path).unwrap();
        file.write_all(&(header.len() as u64).to_le_bytes())
            .unwrap();
        file.write_all(header.as_bytes()).unwrap();
        file.set_len(8 + header.len() as u64 + 1_073_741_824)
            .unwrap();
    }

    let mapped = open_safetensors_checkpoint_mmap(&path).unwrap();
    let bytes = mapped.tensor_bytes("w").unwrap();
    assert_eq!(bytes.len(), 1_073_741_824);
    assert_eq!(bytes[0], 0);
}

#[test]
fn mmap_open_rejects_invalid_checkpoint() {
    let dir = temp_dir("invalid");
    let path = dir.0.join("model.safetensors");
    // Upstream rejects: declared offsets do not cover the data section.
    write_safetensors(
        &path,
        r#"{"w":{"dtype":"U8","shape":[4],"data_offsets":[0,4]}}"#,
        &[0; 8],
    );
    assert!(open_safetensors_checkpoint_mmap(&path).is_err());
}

#[test]
fn mmap_missing_tensor_is_a_structured_error() {
    let dir = temp_dir("missing");
    let path = dir.0.join("model.safetensors");
    write_safetensors(
        &path,
        r#"{"w":{"dtype":"U8","shape":[2],"data_offsets":[0,2]}}"#,
        &[1, 2],
    );
    let mapped = open_safetensors_checkpoint_mmap(&path).unwrap();
    assert!(mapped.tensor_bytes("nope").is_err());
}
