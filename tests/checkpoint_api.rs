// SPDX-License-Identifier: MIT OR Apache-2.0

//! Format-independent `Checkpoint` contract (RM-1784 / #87).
//!
//! One assertion helper runs over `&dyn Checkpoint` for every backend;
//! format-specific assertions check shape normalization, packed dtype
//! preservation, locations, metadata mapping, and runtime detection.

mod common;

use std::borrow::Cow;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use common::{GGML_F32, GGML_Q8_0, KvValue, TensorSpec, build_gguf, f32_vec_to_le_bytes};
use engram_parser::{
    Checkpoint, CheckpointFormat, DimOrder, GgufBackend, MetadataValue, ParserError, SourceKind,
    TensorDType, list_experts, open_checkpoint,
};

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
    let path =
        std::env::temp_dir().join(format!("engram-ckpt-{name}-{}-{nanos}", std::process::id()));
    fs::create_dir_all(&path).unwrap();
    TestDir(path)
}

/// Contract every backend must satisfy.
fn assert_contract(ckpt: &dyn Checkpoint) {
    let tensors = ckpt.tensors();
    assert!(!tensors.is_empty(), "inventory must not be empty");
    assert!(
        tensors.windows(2).all(|w| w[0].name < w[1].name),
        "inventory must be sorted by unique name"
    );
    assert_eq!(tensors, ckpt.tensors(), "inventory must be deterministic");
    for info in tensors {
        assert_eq!(ckpt.tensor(&info.name), Some(info), "lookup {}", info.name);
        let bytes = ckpt.tensor_bytes(&info.name).expect("payload read");
        assert_eq!(bytes.len(), info.byte_len, "payload length {}", info.name);
        if let Some(width) = info.dtype.element_size() {
            let elements = info.shape.element_count().unwrap();
            assert_eq!(info.byte_len, elements * width, "scalar size {}", info.name);
        }
        assert!(
            ckpt.source().files.contains(&info.location.source),
            "location source {} listed in files",
            info.location.source
        );
    }
    assert!(ckpt.tensor("__missing__").is_none());
    match ckpt.tensor_bytes("__missing__").unwrap_err() {
        ParserError::MissingTensor { name, .. } => assert_eq!(name, "__missing__"),
        other => panic!("expected MissingTensor, got {other}"),
    }
}

const EMBD: [f32; 6] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];

/// A tiny GGUF with one dense F32 tensor and one packed Q8_0 tensor.
fn gguf_fixture() -> (Vec<u8>, Vec<u8>) {
    // Q8_0: 64 elements = 2 blocks x 34 bytes.
    let q8: Vec<u8> = (0..68u8).collect();
    let bytes = build_gguf(
        &[
            ("general.architecture", KvValue::Str("llama")),
            ("llama.block_count", KvValue::U32(2)),
            ("llama.rope.freq_base", KvValue::F32(10000.0)),
        ],
        &[
            TensorSpec {
                name: "token_embd.weight",
                // GGUF innermost-first: 3-wide rows, 2 rows.
                dims: vec![3, 2],
                ggml_type: GGML_F32,
                payload: f32_vec_to_le_bytes(&EMBD),
            },
            TensorSpec {
                name: "blk.0.ffn_down.weight",
                dims: vec![32, 2],
                ggml_type: GGML_Q8_0,
                payload: q8.clone(),
            },
        ],
    );
    (bytes, q8)
}

fn write_gguf(dir: &Path, file: &str) -> PathBuf {
    let path = dir.join(file);
    fs::write(&path, gguf_fixture().0).unwrap();
    path
}

#[test]
fn gguf_in_memory_backend_meets_contract() {
    let (bytes, q8) = gguf_fixture();
    let ckpt = GgufBackend::from_bytes(bytes, "mem://fixture").unwrap();
    assert_contract(&ckpt);

    assert_eq!(ckpt.format(), CheckpointFormat::Gguf);
    assert_eq!(ckpt.source().kind, SourceKind::InMemory);

    let embd = ckpt.tensor("token_embd.weight").unwrap();
    assert_eq!(embd.dtype, TensorDType::F32);
    assert_eq!(embd.shape.dims(), &[2, 3], "outermost-first");
    assert_eq!(embd.shape.native_dims(), vec![3, 2]);
    assert_eq!(embd.shape.native_order(), DimOrder::InnermostFirst);
    assert_eq!(
        ckpt.tensor_bytes("token_embd.weight").unwrap().as_ref(),
        f32_vec_to_le_bytes(&EMBD).as_slice()
    );

    let packed = ckpt.tensor("blk.0.ffn_down.weight").unwrap();
    assert_eq!(packed.dtype, TensorDType::GgufPacked { ggml_type: 8 });
    assert!(packed.dtype.is_packed());
    assert_eq!(packed.native_dtype, "Q8_0");
    assert_eq!(packed.byte_len, 68);
    let raw = ckpt.tensor_bytes("blk.0.ffn_down.weight").unwrap();
    assert!(matches!(raw, Cow::Borrowed(_)));
    assert_eq!(raw.as_ref(), q8.as_slice(), "packed blocks untouched");

    let tensor = ckpt.layout().tensor("blk.0.ffn_down.weight").unwrap();
    assert_eq!(packed.location.data_offset, tensor.relative_offset);
    assert_eq!(packed.location.file_offset, Some(tensor.absolute_offset));

    let meta = ckpt.metadata();
    assert_eq!(
        meta.get("general.architecture")
            .and_then(MetadataValue::as_str),
        Some("llama")
    );
    assert_eq!(meta.get("llama.block_count"), Some(&MetadataValue::UInt(2)));
    assert_eq!(
        meta.get("llama.rope.freq_base"),
        Some(&MetadataValue::F32(10000.0))
    );
}

#[test]
fn gguf_format_specific_apis_stay_reachable() {
    let (bytes, _) = gguf_fixture();
    let ckpt = GgufBackend::from_bytes(bytes, "mem://moe").unwrap();
    assert_eq!(ckpt.layout().metadata.block_count(), Some(2));
    assert!(list_experts(ckpt.layout()).is_empty());
    let layout = ckpt.into_layout();
    assert_eq!(layout.find_tensors_with_suffix(".weight").len(), 2);
}

#[test]
fn open_checkpoint_detects_gguf_by_magic() {
    let dir = temp_dir("gguf-magic");
    // No `.gguf` extension: detection must use the magic bytes.
    let path = write_gguf(&dir.0, "weights.bin");
    let ckpt = open_checkpoint(&path).unwrap();
    assert_contract(&ckpt);
    assert_eq!(ckpt.format(), CheckpointFormat::Gguf);
    assert_eq!(ckpt.source().kind, SourceKind::SingleFile);
    assert_eq!(ckpt.source().root, dir.0);
    assert_eq!(ckpt.source().files, vec!["weights.bin".to_string()]);
    assert!(ckpt.as_gguf().is_some());
}

#[test]
fn open_checkpoint_rejects_unknown_and_bad_gguf() {
    let dir = temp_dir("unknown");
    let txt = dir.0.join("notes.txt");
    fs::write(&txt, b"hello").unwrap();
    assert!(matches!(
        open_checkpoint(&txt).unwrap_err(),
        ParserError::UnsupportedFormat { .. }
    ));

    // `.gguf` without the magic reaches the GGUF parser, which rejects it.
    let bad = dir.0.join("bad.gguf");
    fs::write(&bad, b"NOPE").unwrap();
    assert!(open_checkpoint(&bad).is_err());

    assert!(matches!(
        open_checkpoint(dir.0.join("absent.gguf")).unwrap_err(),
        ParserError::Io { .. }
    ));
}

#[cfg(not(feature = "safetensors"))]
#[test]
fn safetensors_input_without_feature_is_explicit() {
    let dir = temp_dir("st-disabled");
    let file = dir.0.join("model.safetensors");
    fs::write(&file, b"{}").unwrap();
    for path in [file.as_path(), dir.0.as_path()] {
        match open_checkpoint(path).unwrap_err() {
            ParserError::FeatureDisabled { feature, .. } => assert_eq!(feature, "safetensors"),
            other => panic!("expected FeatureDisabled, got {other}"),
        }
    }
}

#[cfg(feature = "mmap")]
#[test]
fn gguf_mmap_matches_owned() {
    use engram_parser::{GgufMmapBackend, open_checkpoint_mmap};

    let dir = temp_dir("gguf-mmap");
    let path = write_gguf(&dir.0, "model.gguf");
    let owned = GgufBackend::open(&path).unwrap();
    let mapped = open_checkpoint_mmap(&path).unwrap();
    assert_contract(&mapped);
    assert!(mapped.as_gguf_mmap().is_some());
    assert_eq!(owned.source(), mapped.source());
    assert_eq!(owned.tensors(), mapped.tensors());
    assert_eq!(owned.metadata(), mapped.metadata());
    for info in owned.tensors() {
        let raw = mapped.tensor_bytes(&info.name).unwrap();
        assert!(matches!(raw, Cow::Borrowed(_)));
        assert_eq!(owned.tensor_bytes(&info.name).unwrap(), raw);
    }
    let direct = GgufMmapBackend::open(&path).unwrap();
    assert_eq!(direct.tensors(), owned.tensors());
}

#[cfg(feature = "safetensors")]
mod safetensors {
    use super::*;
    use engram_parser::SafetensorsBackend;

    fn write_st(path: &Path, header: &str, payload: &[u8]) {
        let mut out = (header.len() as u64).to_le_bytes().to_vec();
        out.extend_from_slice(header.as_bytes());
        out.extend_from_slice(payload);
        fs::write(path, out).unwrap();
    }

    const SHARD_A: &str = r#"{"a.weight":{"dtype":"F32","shape":[2,3],"data_offsets":[0,24]},"a.bias":{"dtype":"U8","shape":[4],"data_offsets":[24,28]}}"#;
    const SHARD_B: &str = r#"{"b.weight":{"dtype":"BF16","shape":[2],"data_offsets":[0,4]}}"#;

    fn sharded(dir: &Path) -> PathBuf {
        let mut a = f32_vec_to_le_bytes(&EMBD);
        a.extend_from_slice(&[9, 8, 7, 6]);
        write_st(&dir.join("model-00001-of-00002.safetensors"), SHARD_A, &a);
        write_st(
            &dir.join("model-00002-of-00002.safetensors"),
            SHARD_B,
            &[1, 2, 3, 4],
        );
        let index = dir.join("model.safetensors.index.json");
        fs::write(
            &index,
            r#"{"metadata":{"total_size":32},"weight_map":{"a.weight":"model-00001-of-00002.safetensors","a.bias":"model-00001-of-00002.safetensors","b.weight":"model-00002-of-00002.safetensors"}}"#,
        )
        .unwrap();
        index
    }

    #[test]
    fn sharded_index_meets_contract() {
        let dir = temp_dir("st-index");
        let index = sharded(&dir.0);
        let ckpt = open_checkpoint(&index).unwrap();
        assert_contract(&ckpt);
        assert!(ckpt.as_safetensors().is_some());

        let source = ckpt.source();
        assert_eq!(source.format, CheckpointFormat::Safetensors);
        assert_eq!(source.kind, SourceKind::ShardIndex);
        assert_eq!(
            source.index_file.as_deref(),
            Some("model.safetensors.index.json")
        );
        assert_eq!(
            source.files,
            vec![
                "model-00001-of-00002.safetensors".to_string(),
                "model-00002-of-00002.safetensors".to_string(),
            ]
        );

        let bias = ckpt.tensor("a.bias").unwrap();
        assert_eq!(bias.dtype, TensorDType::U8);
        assert_eq!(bias.location.source, "model-00001-of-00002.safetensors");
        assert_eq!(bias.location.data_offset, 24);
        assert_eq!(bias.location.file_offset, Some(8 + SHARD_A.len() + 24));
        let raw = ckpt.tensor_bytes("a.bias").unwrap();
        assert!(matches!(raw, Cow::Owned(_)));
        assert_eq!(raw.as_ref(), &[9, 8, 7, 6]);

        let b = ckpt.tensor("b.weight").unwrap();
        assert_eq!(b.dtype, TensorDType::BF16);
        assert_eq!(b.location.file_offset, Some(8 + SHARD_B.len()));
        assert_eq!(
            ckpt.metadata()
                .get("index:total_size")
                .and_then(MetadataValue::as_str),
            Some("32")
        );
    }

    #[test]
    fn directory_and_single_file_detection() {
        let dir = temp_dir("st-dir");
        let index = sharded(&dir.0);
        // Discovery prefers an index inside a directory; without one the
        // directory itself is the shard set.
        assert_eq!(
            open_checkpoint(&dir.0).unwrap().source().kind,
            SourceKind::ShardIndex
        );
        fs::remove_file(index).unwrap();
        let from_dir = open_checkpoint(&dir.0).unwrap();
        assert_contract(&from_dir);
        assert_eq!(from_dir.source().kind, SourceKind::Directory);
        assert_eq!(from_dir.source().root, dir.0);
        assert_eq!(from_dir.source().files.len(), 2);

        let single = dir.0.join("model-00002-of-00002.safetensors");
        let ckpt = SafetensorsBackend::open(&single).unwrap();
        assert_contract(&ckpt);
        assert_eq!(ckpt.source().kind, SourceKind::SingleFile);
        assert_eq!(ckpt.checkpoint().manifest().tensors.len(), 1);
    }

    #[test]
    fn same_logical_tensor_normalizes_identically_across_formats() {
        let dir = temp_dir("st-vs-gguf");
        let st_path = dir.0.join("model.safetensors");
        write_st(
            &st_path,
            r#"{"token_embd.weight":{"dtype":"F32","shape":[2,3],"data_offsets":[0,24]}}"#,
            &f32_vec_to_le_bytes(&EMBD),
        );
        let gguf_path = write_gguf(&dir.0, "model.gguf");

        let st = open_checkpoint(&st_path).unwrap();
        let gguf = open_checkpoint(&gguf_path).unwrap();
        let (a, b) = (
            st.tensor("token_embd.weight").unwrap(),
            gguf.tensor("token_embd.weight").unwrap(),
        );
        assert_eq!(a.dtype, b.dtype);
        assert_eq!(a.shape.dims(), b.shape.dims());
        assert_eq!(a.byte_len, b.byte_len);
        assert_eq!(
            st.tensor_bytes("token_embd.weight").unwrap(),
            gguf.tensor_bytes("token_embd.weight").unwrap()
        );
        assert_eq!(a.shape.native_order(), DimOrder::OutermostFirst);
        assert_eq!(b.shape.native_order(), DimOrder::InnermostFirst);
    }

    #[cfg(feature = "mmap")]
    #[test]
    fn safetensors_mmap_matches_owned() {
        use engram_parser::open_checkpoint_mmap;

        let dir = temp_dir("st-mmap");
        let index = sharded(&dir.0);
        let owned = open_checkpoint(&index).unwrap();
        let mapped = open_checkpoint_mmap(&index).unwrap();
        assert_contract(&mapped);
        assert!(mapped.as_safetensors_mmap().is_some());
        assert_eq!(owned.source(), mapped.source());
        assert_eq!(owned.tensors(), mapped.tensors());
        assert_eq!(owned.metadata(), mapped.metadata());
        for info in owned.tensors() {
            let raw = mapped.tensor_bytes(&info.name).unwrap();
            assert!(matches!(raw, Cow::Borrowed(_)));
            assert_eq!(owned.tensor_bytes(&info.name).unwrap(), raw);
        }
    }
}
