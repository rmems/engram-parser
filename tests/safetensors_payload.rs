// SPDX-License-Identifier: MIT OR Apache-2.0
#![cfg(feature = "safetensors")]

//! Raw payload access: single-file, shard-index, and directory reads,
//! plus malformed/bounds rejection.

use engram_parser::ParserError;
use engram_parser::safetensors::open_safetensors_checkpoint;
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
        "engram-st-payload-{name}-{}-{nanos}",
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

fn fixture_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/safetensors")
}

#[test]
fn reads_single_file_payload() {
    let path = fixture_root().join("single/model.safetensors");
    let checkpoint = open_safetensors_checkpoint(&path).expect("open single-file");

    let tensor = checkpoint.tensor("a.weight").expect("tensor lookup");
    assert_eq!(tensor.dtype, "F16");
    assert_eq!(tensor.shape, vec![1]);
    assert_eq!(tensor.byte_size, 2);
    assert_eq!(checkpoint.tensor_bytes("a.weight").unwrap(), vec![0, 0]);
}

#[test]
fn reads_real_payload_bytes() {
    let dir = temp_dir("real-payload");
    let path = dir.0.join("model.safetensors");
    let payload: Vec<u8> = (0u8..16).collect();
    write_safetensors(
        &path,
        r#"{"w":{"dtype":"F32","shape":[2,2],"data_offsets":[0,16]}}"#,
        &payload,
    );

    let checkpoint = open_safetensors_checkpoint(&path).unwrap();
    assert_eq!(checkpoint.tensor_bytes("w").unwrap(), payload);
}

#[test]
fn reads_sharded_checkpoint_across_index() {
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

    let checkpoint =
        open_safetensors_checkpoint(dir.0.join("model.safetensors.index.json")).unwrap();
    assert_eq!(
        checkpoint.tensor_bytes("a.weight").unwrap(),
        vec![1, 2, 3, 4]
    );
    assert_eq!(
        checkpoint.tensor_bytes("b.weight").unwrap(),
        vec![5, 6, 7, 8]
    );
}

#[test]
fn reads_directory_layout() {
    let dir = temp_dir("directory");
    write_safetensors(
        &dir.0.join("a.safetensors"),
        r#"{"a.weight":{"dtype":"U8","shape":[2],"data_offsets":[0,2]}}"#,
        &[9, 9],
    );
    write_safetensors(
        &dir.0.join("b.safetensors"),
        r#"{"b.weight":{"dtype":"U8","shape":[2],"data_offsets":[0,2]}}"#,
        &[7, 7],
    );

    let checkpoint = open_safetensors_checkpoint(&dir.0).unwrap();
    assert_eq!(checkpoint.tensor_bytes("a.weight").unwrap(), vec![9, 9]);
    assert_eq!(checkpoint.tensor_bytes("b.weight").unwrap(), vec![7, 7]);
}

#[test]
fn discovery_results_resolve_to_payload() {
    let dir = temp_dir("moe-resolve");
    write_safetensors(
        &dir.0.join("model.safetensors"),
        r#"{"model.layers.0.mlp.experts.0.w1.weight":{"dtype":"U8","shape":[3],"data_offsets":[0,3]},"model.layers.0.mlp.gate.weight":{"dtype":"U8","shape":[2],"data_offsets":[3,5]}}"#,
        &[1, 2, 3, 4, 5],
    );

    let checkpoint = open_safetensors_checkpoint(dir.0.join("model.safetensors")).unwrap();
    let manifest = checkpoint.manifest();

    // Every record and every discovery candidate resolves to raw bytes.
    for record in &manifest.tensors {
        let bytes = checkpoint.resolve_tensor_bytes(record).unwrap();
        assert_eq!(bytes.len(), record.byte_size);
    }
    for name in manifest
        .candidates
        .router_tensors
        .iter()
        .chain(manifest.candidates.expert_tensors.iter())
    {
        assert!(checkpoint.tensor_bytes(name).is_ok(), "unresolved: {name}");
    }
}

#[test]
fn missing_tensor_is_a_structured_error() {
    let path = fixture_root().join("single/model.safetensors");
    let checkpoint = open_safetensors_checkpoint(&path).unwrap();
    match checkpoint.tensor_bytes("nope").unwrap_err() {
        ParserError::MissingTensor { name, .. } => assert_eq!(name, "nope"),
        other => panic!("expected MissingTensor, got {other}"),
    }
}

#[test]
fn malformed_header_is_rejected() {
    let dir = temp_dir("bad-dtype");
    let path = dir.0.join("model.safetensors");
    write_safetensors(
        &path,
        r#"{"w":{"dtype":"NOPE","shape":[1],"data_offsets":[0,4]}}"#,
        &[0; 4],
    );
    assert!(open_safetensors_checkpoint(&path).is_err());
}

#[test]
fn byte_size_mismatch_is_rejected() {
    let dir = temp_dir("size-mismatch");
    let path = dir.0.join("model.safetensors");
    write_safetensors(
        &path,
        r#"{"w":{"dtype":"F32","shape":[2],"data_offsets":[0,12]}}"#,
        &[0; 12],
    );
    assert!(open_safetensors_checkpoint(&path).is_err());
}

#[test]
fn reversed_offsets_are_rejected() {
    let dir = temp_dir("reversed");
    let path = dir.0.join("model.safetensors");
    write_safetensors(
        &path,
        r#"{"w":{"dtype":"U8","shape":[4],"data_offsets":[4,0]}}"#,
        &[0; 4],
    );
    assert!(open_safetensors_checkpoint(&path).is_err());
}

#[test]
fn reads_stay_pinned_when_shard_path_is_replaced() {
    let dir = temp_dir("replaced-shard");
    let shard = dir.0.join("model.safetensors");
    write_safetensors(
        &shard,
        r#"{"w":{"dtype":"U8","shape":[4],"data_offsets":[0,4]}}"#,
        &[1, 2, 3, 4],
    );

    let checkpoint = open_safetensors_checkpoint(&shard).unwrap();

    // Atomically replace the shard after open: reads must keep returning
    // the bytes of the file that was actually inspected.
    let replacement = dir.0.join("replacement.safetensors");
    write_safetensors(
        &replacement,
        r#"{"w":{"dtype":"U8","shape":[4],"data_offsets":[0,4]}}"#,
        &[9, 9, 9, 9],
    );
    fs::rename(&replacement, &shard).unwrap();

    assert_eq!(checkpoint.tensor_bytes("w").unwrap(), vec![1, 2, 3, 4]);
}

#[cfg(unix)]
#[test]
fn non_utf8_shard_filename_resolves() {
    use std::ffi::OsStr;
    use std::os::unix::ffi::OsStrExt;

    let dir = temp_dir("non-utf8");
    let name = OsStr::from_bytes(b"model-\xff.safetensors");
    write_safetensors(
        &dir.0.join(name),
        r#"{"w":{"dtype":"U8","shape":[2],"data_offsets":[0,2]}}"#,
        &[6, 7],
    );

    let checkpoint = open_safetensors_checkpoint(&dir.0).unwrap();
    assert_eq!(checkpoint.tensor_bytes("w").unwrap(), vec![6, 7]);
}

#[test]
fn offsets_beyond_eof_are_rejected() {
    let dir = temp_dir("beyond-eof");
    let path = dir.0.join("model.safetensors");
    write_safetensors(
        &path,
        r#"{"w":{"dtype":"U8","shape":[4],"data_offsets":[0,8]}}"#,
        &[0; 4],
    );
    assert!(open_safetensors_checkpoint(&path).is_err());
}
