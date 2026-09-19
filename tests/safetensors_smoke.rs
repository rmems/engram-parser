// SPDX-License-Identifier: MIT OR Apache-2.0
#![cfg(feature = "safetensors")]

//! Tiny on-disk fixtures: single-file, HF shard index, and directory layouts.

use engram_parser::safetensors::{inspect_safetensors_checkpoint, write_safetensors_manifest};
use std::path::PathBuf;

fn fixture_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/safetensors")
}

#[test]
fn inspects_committed_single_file_fixture() {
    let path = fixture_root().join("single/model.safetensors");
    let manifest = inspect_safetensors_checkpoint(&path).expect("single-file fixture");
    assert_eq!(manifest.checkpoint.input_kind, "single_file");
    assert_eq!(manifest.checkpoint.shard_count, 1);
    assert_eq!(manifest.checkpoint.tensor_count, 1);
    assert_eq!(manifest.tensors[0].name, "a.weight");
    assert_eq!(manifest.tensors[0].dtype, "F16");
    assert_eq!(manifest.tensors[0].shape, vec![1]);
    assert_eq!(manifest.tensors[0].byte_size, 2);
    assert_eq!(manifest.tensors[0].source_shard, "model.safetensors");
}

#[test]
fn inspects_committed_shard_index_fixture() {
    let path = fixture_root().join("sharded/model.safetensors.index.json");
    let manifest = inspect_safetensors_checkpoint(&path).expect("shard-index fixture");
    assert_eq!(manifest.checkpoint.input_kind, "hf_index");
    assert_eq!(
        manifest.checkpoint.index_file.as_deref(),
        Some("model.safetensors.index.json")
    );
    assert_eq!(manifest.checkpoint.shard_count, 2);
    assert_eq!(manifest.checkpoint.tensor_count, 2);
    assert_eq!(manifest.tensors[0].name, "a.weight");
    assert_eq!(manifest.tensors[1].name, "b.weight");
    assert_eq!(
        manifest.tensors[0].source_shard,
        "model-00001-of-00002.safetensors"
    );
    assert_eq!(
        manifest.tensors[1].source_shard,
        "model-00002-of-00002.safetensors"
    );
}

#[test]
fn inspects_committed_directory_fixture() {
    let path = fixture_root().join("directory");
    let manifest = inspect_safetensors_checkpoint(&path).expect("directory fixture");
    assert_eq!(manifest.checkpoint.input_kind, "directory");
    assert_eq!(manifest.checkpoint.index_file, None);
    assert_eq!(manifest.checkpoint.shard_count, 2);
    assert_eq!(manifest.tensors[0].name, "a.weight");
    assert_eq!(manifest.tensors[1].name, "b.weight");
}

#[test]
fn fixture_manifest_write_is_byte_stable() {
    let path = fixture_root().join("single/model.safetensors");
    let out = std::env::temp_dir().join(format!(
        "engram-fixture-manifest-{}-{}.json",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    write_safetensors_manifest(&path, &out).unwrap();
    let first = std::fs::read_to_string(&out).unwrap();
    write_safetensors_manifest(&path, &out).unwrap();
    let second = std::fs::read_to_string(&out).unwrap();
    let _ = std::fs::remove_file(&out);
    assert_eq!(first, second);
    assert!(first.starts_with("{\n  \"candidates\":"));
}

#[test]
fn sharded_fixture_manifest_is_byte_stable() {
    let path = fixture_root().join("sharded");
    let first = inspect_safetensors_checkpoint(&path)
        .expect("sharded fixture")
        .to_pretty_json();
    let second = inspect_safetensors_checkpoint(&path)
        .expect("sharded fixture")
        .to_pretty_json();
    assert_eq!(first, second);
}
