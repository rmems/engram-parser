// SPDX-License-Identifier: MIT OR Apache-2.0
#![cfg(feature = "safetensors")]

//! A compact cross-format fixture matrix for the shared `Checkpoint` API.
//!
//! The fixtures deliberately use the same scalar tensors and payloads in
//! GGUF and Safetensors.  They test the abstraction boundary rather than the
//! unrelated container headers or physical offsets of the two formats.

mod common;

use std::borrow::Cow;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use common::{GGML_F32, TensorSpec, build_gguf, f32_vec_to_le_bytes};
use engram_parser::{Checkpoint, DimOrder, GgufBackend, SourceKind, TensorDType, open_checkpoint};

const MATRIX_F32: [f32; 6] = [1.0, -2.0, 3.5, 4.25, 0.0, 8.0];
const MATRIX_F16: [u8; 4] = [0x00, 0x3c, 0x00, 0xc0];
const GGML_F16: u32 = 1;

struct TestDir(PathBuf);

impl Drop for TestDir {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

fn temp_dir(name: &str) -> TestDir {
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let path = std::env::temp_dir().join(format!(
        "engram-parity-{name}-{}-{nonce}",
        std::process::id()
    ));
    fs::create_dir_all(&path).unwrap();
    TestDir(path)
}

fn write_safetensors(path: &Path, header: &str, payload: &[u8]) {
    let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
    bytes.extend_from_slice(header.as_bytes());
    bytes.extend_from_slice(payload);
    fs::write(path, bytes).unwrap();
}

fn gguf_bytes() -> Vec<u8> {
    // Input order intentionally differs from lexical order: the shared
    // catalog, not either container directory, supplies deterministic order.
    build_gguf(
        &[],
        &[
            TensorSpec {
                name: "z.f16",
                dims: vec![2],
                ggml_type: GGML_F16,
                payload: MATRIX_F16.to_vec(),
            },
            TensorSpec {
                name: "a.matrix",
                // GGUF stores these as innermost-first; the normalized shape
                // is the Safetensors/NumPy shape [2, 3].
                dims: vec![3, 2],
                ggml_type: GGML_F32,
                payload: f32_vec_to_le_bytes(&MATRIX_F32),
            },
        ],
    )
}

fn write_sharded_safetensors(dir: &Path) -> PathBuf {
    write_safetensors(
        &dir.join("model-00002-of-00002.safetensors"),
        r#"{"z.f16":{"dtype":"F16","shape":[2],"data_offsets":[0,4]}}"#,
        &MATRIX_F16,
    );
    write_safetensors(
        &dir.join("model-00001-of-00002.safetensors"),
        r#"{"a.matrix":{"dtype":"F32","shape":[2,3],"data_offsets":[0,24]}}"#,
        &f32_vec_to_le_bytes(&MATRIX_F32),
    );
    let index = dir.join("model.safetensors.index.json");
    fs::write(
        &index,
        r#"{"metadata":{"total_size":28},"weight_map":{"z.f16":"model-00002-of-00002.safetensors","a.matrix":"model-00001-of-00002.safetensors"}}"#,
    )
    .unwrap();
    index
}

fn inventory(checkpoint: &dyn Checkpoint) -> Vec<(String, TensorDType, Vec<usize>, usize)> {
    checkpoint
        .tensors()
        .iter()
        .map(|tensor| {
            (
                tensor.name.clone(),
                tensor.dtype.clone(),
                tensor.shape.dims().to_vec(),
                tensor.byte_len,
            )
        })
        .collect()
}

#[test]
fn parity_matrix_normalizes_inventory_payloads_and_ordering() {
    let dir = temp_dir("inventory");
    let gguf_path = dir.0.join("model.gguf");
    fs::write(&gguf_path, gguf_bytes()).unwrap();
    let index = write_sharded_safetensors(&dir.0);

    let gguf = open_checkpoint(&gguf_path).unwrap();
    let safetensors = open_checkpoint(&index).unwrap();

    let expected = vec![
        ("a.matrix".to_owned(), TensorDType::F32, vec![2, 3], 24),
        ("z.f16".to_owned(), TensorDType::F16, vec![2], 4),
    ];
    assert_eq!(inventory(&gguf), expected);
    assert_eq!(inventory(&safetensors), expected);
    assert_eq!(gguf.tensors(), gguf.tensors(), "GGUF order is stable");
    assert_eq!(
        safetensors.tensors(),
        safetensors.tensors(),
        "Safetensors order is stable"
    );

    for name in ["a.matrix", "z.f16"] {
        assert_eq!(
            gguf.tensor_bytes(name).unwrap(),
            safetensors.tensor_bytes(name).unwrap()
        );
    }
    assert_eq!(
        gguf.tensor("a.matrix").unwrap().shape.native_order(),
        DimOrder::InnermostFirst
    );
    assert_eq!(
        safetensors.tensor("a.matrix").unwrap().shape.native_order(),
        DimOrder::OutermostFirst
    );
}

#[test]
fn parity_matrix_records_sharded_safetensors_ownership() {
    let dir = temp_dir("ownership");
    let index = write_sharded_safetensors(&dir.0);
    let checkpoint = open_checkpoint(&index).unwrap();

    assert_eq!(checkpoint.source().kind, SourceKind::ShardIndex);
    assert_eq!(
        checkpoint.source().files,
        vec![
            "model-00001-of-00002.safetensors".to_owned(),
            "model-00002-of-00002.safetensors".to_owned(),
        ]
    );
    assert_eq!(
        checkpoint.tensor("a.matrix").unwrap().location.source,
        "model-00001-of-00002.safetensors"
    );
    assert_eq!(
        checkpoint.tensor("z.f16").unwrap().location.source,
        "model-00002-of-00002.safetensors"
    );
}

#[test]
fn parity_matrix_rejects_malformed_payload_bounds() {
    let dir = temp_dir("bounds");
    let mut truncated_gguf = gguf_bytes();
    truncated_gguf.truncate(truncated_gguf.len() - 1);
    assert!(GgufBackend::from_bytes(truncated_gguf, "truncated.gguf").is_err());

    let safetensors = dir.0.join("truncated.safetensors");
    write_safetensors(
        &safetensors,
        r#"{"a.matrix":{"dtype":"F32","shape":[2,3],"data_offsets":[0,24]}}"#,
        &f32_vec_to_le_bytes(&MATRIX_F32)[..23],
    );
    assert!(open_checkpoint(&safetensors).is_err());
}

#[cfg(feature = "mmap")]
#[test]
fn parity_matrix_gguf_mmap_and_owned_are_identical_borrowed_views() {
    use engram_parser::open_checkpoint_mmap;

    let dir = temp_dir("mmap");
    let path = dir.0.join("model.gguf");
    fs::write(&path, gguf_bytes()).unwrap();
    let owned = open_checkpoint(&path).unwrap();
    let mapped = open_checkpoint_mmap(&path).unwrap();

    assert_eq!(owned.tensors(), mapped.tensors());
    for name in ["a.matrix", "z.f16"] {
        assert!(matches!(
            owned.tensor_bytes(name).unwrap(),
            Cow::Borrowed(_)
        ));
        assert!(matches!(
            mapped.tensor_bytes(name).unwrap(),
            Cow::Borrowed(_)
        ));
        assert_eq!(
            owned.tensor_bytes(name).unwrap(),
            mapped.tensor_bytes(name).unwrap()
        );
    }
}
