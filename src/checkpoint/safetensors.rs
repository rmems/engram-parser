// SPDX-License-Identifier: MIT OR Apache-2.0
//! Safetensors backends for [`Checkpoint`], built on [`crate::safetensors`].
//!
//! Discovery, shard path policy, header validation, and the manifest all
//! come from the existing `safetensors` module; these wrappers only
//! normalize its records into the shared vocabulary. The manifest,
//! candidate discovery, and the underlying checkpoint handles stay
//! reachable through `checkpoint()` / `into_checkpoint()`.

use std::borrow::Cow;
use std::fs;
use std::path::{Path, PathBuf};

use super::{
    Catalog, Checkpoint, CheckpointFormat, CheckpointMetadata, CheckpointSource, MetadataValue,
    SourceKind, TensorDType, TensorInfo, TensorLocation, TensorShape, delegate_catalog,
};
use crate::error::{ParserError, Result};
use crate::safetensors::{
    SafetensorsCheckpoint, SafetensorsManifest, SafetensorsTensorRecord,
    open_safetensors_checkpoint,
};

/// [`Checkpoint`] over a [`SafetensorsCheckpoint`] (bounded per-tensor reads).
///
/// [`Checkpoint::tensor_bytes`] returns [`Cow::Owned`] bytes read from
/// the owning shard; memory use is bounded by the requested tensor.
#[derive(Debug)]
pub struct SafetensorsBackend {
    checkpoint: SafetensorsCheckpoint,
    catalog: Catalog,
}

impl SafetensorsBackend {
    /// Open a `.safetensors` file, `*.safetensors.index.json`, or shard directory.
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let checkpoint = open_safetensors_checkpoint(path)?;
        let catalog = build_catalog(path, checkpoint.manifest(), |shard| {
            checkpoint.shard_data_begin(shard)
        })?;
        Ok(Self {
            checkpoint,
            catalog,
        })
    }

    /// Format-specific handle (manifest, candidates, record lookup).
    pub fn checkpoint(&self) -> &SafetensorsCheckpoint {
        &self.checkpoint
    }

    /// Unwrap into the format-specific handle.
    pub fn into_checkpoint(self) -> SafetensorsCheckpoint {
        self.checkpoint
    }
}

impl Checkpoint for SafetensorsBackend {
    delegate_catalog!();

    fn tensor_bytes(&self, name: &str) -> Result<Cow<'_, [u8]>> {
        let info = self.catalog.require(name)?;
        let bytes = self.checkpoint.tensor_bytes(name)?;
        self.catalog.check_len(info, bytes.len())?;
        Ok(Cow::Owned(bytes))
    }
}

/// [`Checkpoint`] over a memory-mapped
/// [`crate::safetensors::SafetensorsCheckpointMmap`] (borrowed payloads).
///
/// Callers must not modify shard files while the backend is alive:
/// in-place writes are silently visible, truncation within a payload's
/// last page yields zeros for the tail, and truncating so a reader
/// accesses a page wholly past EOF raises `SIGBUS` (see "Concurrent
/// modification" on [`crate::safetensors::open_safetensors_checkpoint_mmap`]).
#[cfg(feature = "mmap")]
#[derive(Debug)]
pub struct SafetensorsMmapBackend {
    checkpoint: crate::safetensors::SafetensorsCheckpointMmap,
    catalog: Catalog,
}

#[cfg(feature = "mmap")]
impl SafetensorsMmapBackend {
    /// Map a `.safetensors` file, `*.safetensors.index.json`, or shard directory.
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let checkpoint = crate::safetensors::open_safetensors_checkpoint_mmap(path)?;
        let catalog = build_catalog(path, checkpoint.manifest(), |shard| {
            checkpoint.shard_data_begin(shard)
        })?;
        Ok(Self {
            checkpoint,
            catalog,
        })
    }

    /// Format-specific mapped handle.
    pub fn checkpoint(&self) -> &crate::safetensors::SafetensorsCheckpointMmap {
        &self.checkpoint
    }

    /// Unwrap into the format-specific mapped handle.
    pub fn into_checkpoint(self) -> crate::safetensors::SafetensorsCheckpointMmap {
        self.checkpoint
    }
}

#[cfg(feature = "mmap")]
impl Checkpoint for SafetensorsMmapBackend {
    delegate_catalog!();

    fn tensor_bytes(&self, name: &str) -> Result<Cow<'_, [u8]>> {
        let info = self.catalog.require(name)?;
        let bytes = self.checkpoint.tensor_bytes(name)?;
        self.catalog.check_len(info, bytes.len())?;
        Ok(Cow::Borrowed(bytes))
    }
}

/// Shared by owned and mmap backends so both produce identical catalogs.
fn build_catalog(
    path: &Path,
    manifest: &SafetensorsManifest,
    data_begin: impl Fn(&str) -> Option<u64>,
) -> Result<Catalog> {
    let mut files: Vec<String> = manifest
        .tensors
        .iter()
        .map(|t| t.source_shard.clone())
        .collect();
    files.sort();
    files.dedup();

    let source = CheckpointSource {
        format: CheckpointFormat::Safetensors,
        kind: source_kind(path, &manifest.checkpoint.input_kind)?,
        path: path.to_path_buf(),
        root: checkpoint_root(path)?,
        index_file: manifest.checkpoint.index_file.clone(),
        files,
    };
    let tensors = manifest
        .tensors
        .iter()
        .map(|record| {
            let file_offset = data_begin(&record.source_shard)
                .and_then(|begin| usize::try_from(begin).ok())
                .and_then(|begin| begin.checked_add(record.data_offsets[0]));
            tensor_info(record, file_offset)
        })
        .collect();
    let metadata: CheckpointMetadata = manifest
        .checkpoint
        .metadata
        .iter()
        .map(|(k, v)| (k.clone(), MetadataValue::String(v.clone())))
        .collect();
    Ok(Catalog::new(source, tensors, metadata))
}

fn tensor_info(record: &SafetensorsTensorRecord, file_offset: Option<usize>) -> TensorInfo {
    TensorInfo::new(
        record.name.clone(),
        normalize_dtype(&record.dtype),
        TensorShape::outermost_first(record.shape.clone()),
        record.byte_size,
        TensorLocation::new(
            record.source_shard.clone(),
            record.data_offsets[0],
            file_offset,
        ),
    )
    .with_native_dtype(record.dtype.clone())
}

fn normalize_dtype(label: &str) -> TensorDType {
    match label {
        "F64" => TensorDType::F64,
        "F32" => TensorDType::F32,
        "F16" => TensorDType::F16,
        "BF16" => TensorDType::BF16,
        "F8_E4M3" => TensorDType::F8E4M3,
        "F8_E5M2" => TensorDType::F8E5M2,
        "I64" => TensorDType::I64,
        "I32" => TensorDType::I32,
        "I16" => TensorDType::I16,
        "I8" => TensorDType::I8,
        "U64" => TensorDType::U64,
        "U32" => TensorDType::U32,
        "U16" => TensorDType::U16,
        "U8" => TensorDType::U8,
        "BOOL" => TensorDType::Bool,
        "I4" | "INT4" => TensorDType::I4,
        "U4" => TensorDType::U4,
        other => TensorDType::Other(other.to_owned()),
    }
}

fn source_kind(path: &Path, input_kind: &str) -> Result<SourceKind> {
    match input_kind {
        "single_file" => Ok(SourceKind::SingleFile),
        "hf_index" => Ok(SourceKind::ShardIndex),
        "directory" => Ok(SourceKind::Directory),
        other => Err(ParserError::UnsupportedFormat {
            path: path.display().to_string(),
            reason: format!("unknown Safetensors input kind '{other}'"),
        }),
    }
}

/// Same root rule as Safetensors shard resolution: a directory is its own
/// root; a file resolves against its parent (or `.`).
fn checkpoint_root(path: &Path) -> Result<PathBuf> {
    let metadata = fs::metadata(path).map_err(|source| ParserError::Io {
        path: path.display().to_string(),
        source,
    })?;
    if metadata.is_dir() {
        return Ok(path.to_path_buf());
    }
    Ok(path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."))
        .to_path_buf())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dtype_labels_normalize() {
        assert_eq!(normalize_dtype("BF16"), TensorDType::BF16);
        assert_eq!(normalize_dtype("INT4"), TensorDType::I4);
        assert_eq!(normalize_dtype("BOOL"), TensorDType::Bool);
        assert_eq!(normalize_dtype("C64"), TensorDType::Other("C64".into()));
    }

    #[test]
    fn unknown_input_kind_is_rejected() {
        assert!(source_kind(Path::new("x"), "zip").is_err());
        assert_eq!(
            source_kind(Path::new("x"), "hf_index").unwrap(),
            SourceKind::ShardIndex
        );
    }
}
