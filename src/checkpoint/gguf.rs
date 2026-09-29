// SPDX-License-Identifier: MIT OR Apache-2.0
//! GGUF backends for [`Checkpoint`], built on [`crate::gguf`].
//!
//! The wrappers own the format-specific layout plus a precomputed
//! [`Catalog`]. [`crate::GgufLayout`] itself is unchanged; `layout()` /
//! `into_layout()` keep MoE helpers, suffix search, dequant, and (with
//! `mmap`) page-aligned views reachable.

use std::borrow::Cow;
use std::collections::HashMap;
use std::path::{Path, PathBuf};

use super::{
    Catalog, Checkpoint, CheckpointFormat, CheckpointMetadata, CheckpointSource, MetadataValue,
    SourceKind, TensorDType, TensorInfo, TensorLocation, TensorShape, delegate_catalog,
};
use crate::error::Result;
use crate::gguf::{
    DType, GgufLayout, GgufMetadata, ParseLimits, Tensor, ggml_type_label, load_gguf_with_limits,
    parse_bytes_with_limits,
};

/// [`Checkpoint`] over an owned [`GgufLayout`] (whole file in memory).
#[derive(Debug)]
pub struct GgufBackend {
    layout: GgufLayout,
    catalog: Catalog,
}

impl GgufBackend {
    /// Read and parse a `.gguf` file with [`ParseLimits::default`].
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        Self::open_with_limits(path, ParseLimits::default())
    }

    /// Read and parse a `.gguf` file with explicit limits.
    pub fn open_with_limits(path: impl AsRef<Path>, limits: ParseLimits) -> Result<Self> {
        let layout = load_gguf_with_limits(path, limits)?;
        Ok(Self::with_kind(layout, SourceKind::SingleFile))
    }

    /// Parse an in-memory GGUF buffer. `label` is used as the source path.
    pub fn from_bytes(bytes: Vec<u8>, label: impl Into<String>) -> Result<Self> {
        let layout = parse_bytes_with_limits(bytes, label.into(), ParseLimits::default())?;
        Ok(Self::with_kind(layout, SourceKind::InMemory))
    }

    /// Wrap an already-parsed layout. The source kind is
    /// [`SourceKind::SingleFile`] when `layout.path` names an existing
    /// file, otherwise [`SourceKind::InMemory`].
    pub fn from_layout(layout: GgufLayout) -> Self {
        let kind = if Path::new(&layout.path).is_file() {
            SourceKind::SingleFile
        } else {
            SourceKind::InMemory
        };
        Self::with_kind(layout, kind)
    }

    fn with_kind(layout: GgufLayout, kind: SourceKind) -> Self {
        let catalog = build_catalog(&layout.path, kind, &layout.metadata, &layout.tensors);
        Self { layout, catalog }
    }

    /// Format-specific layout (MoE helpers, suffix search, dequant).
    pub fn layout(&self) -> &GgufLayout {
        &self.layout
    }

    /// Unwrap into the format-specific layout.
    pub fn into_layout(self) -> GgufLayout {
        self.layout
    }
}

impl Checkpoint for GgufBackend {
    delegate_catalog!();

    fn tensor_bytes(&self, name: &str) -> Result<Cow<'_, [u8]>> {
        let info = self.catalog.require(name)?;
        let bytes = self.layout.tensor_bytes(self.layout.tensor(name)?)?;
        self.catalog.check_len(info, bytes.len())?;
        Ok(Cow::Borrowed(bytes))
    }
}

/// [`Checkpoint`] over a memory-mapped [`crate::GgufLayoutMmap`].
///
/// Callers must not truncate the file while the backend is alive
/// (standard mmap invariant).
#[cfg(feature = "mmap")]
#[derive(Debug)]
pub struct GgufMmapBackend {
    layout: crate::gguf::GgufLayoutMmap,
    catalog: Catalog,
}

#[cfg(feature = "mmap")]
impl GgufMmapBackend {
    /// Map and parse a `.gguf` file with [`ParseLimits::default`].
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        Self::open_with_limits(path, ParseLimits::default())
    }

    /// Map and parse a `.gguf` file with explicit limits.
    pub fn open_with_limits(path: impl AsRef<Path>, limits: ParseLimits) -> Result<Self> {
        Ok(Self::from_layout(crate::gguf::load_gguf_mmap_with_limits(
            path, limits,
        )?))
    }

    /// Wrap an already-mapped layout.
    pub fn from_layout(layout: crate::gguf::GgufLayoutMmap) -> Self {
        let catalog = build_catalog(
            &layout.path,
            SourceKind::SingleFile,
            &layout.metadata,
            &layout.tensors,
        );
        Self { layout, catalog }
    }

    /// Format-specific mapped layout (page-aligned views, directory checks).
    pub fn layout(&self) -> &crate::gguf::GgufLayoutMmap {
        &self.layout
    }

    /// Unwrap into the format-specific mapped layout.
    pub fn into_layout(self) -> crate::gguf::GgufLayoutMmap {
        self.layout
    }
}

#[cfg(feature = "mmap")]
impl Checkpoint for GgufMmapBackend {
    delegate_catalog!();

    fn tensor_bytes(&self, name: &str) -> Result<Cow<'_, [u8]>> {
        let info = self.catalog.require(name)?;
        let bytes = self.layout.tensor_bytes(self.layout.tensor(name)?)?;
        self.catalog.check_len(info, bytes.len())?;
        Ok(Cow::Borrowed(bytes))
    }
}

/// Shared by owned and mmap backends so both produce identical catalogs.
fn build_catalog(
    path: &str,
    kind: SourceKind,
    metadata: &GgufMetadata,
    tensors: &HashMap<String, Tensor>,
) -> Catalog {
    let source = gguf_source(path, kind);
    let file = source.files[0].clone();
    let infos = tensors
        .values()
        .map(|tensor| tensor_info(tensor, &file))
        .collect();
    Catalog::new(source, infos, metadata_values(metadata))
}

fn gguf_source(path: &str, kind: SourceKind) -> CheckpointSource {
    let as_path = PathBuf::from(path);
    let (root, file) = match (kind, as_path.file_name()) {
        // Same root rule as Safetensors: a bare filename resolves against `.`.
        (SourceKind::SingleFile, Some(name)) => (
            as_path
                .parent()
                .filter(|parent| !parent.as_os_str().is_empty())
                .unwrap_or_else(|| Path::new("."))
                .to_path_buf(),
            name.to_string_lossy().into_owned(),
        ),
        _ => (PathBuf::new(), path.to_owned()),
    };
    CheckpointSource {
        format: CheckpointFormat::Gguf,
        kind,
        path: as_path,
        root,
        index_file: None,
        files: vec![file],
    }
}

fn tensor_info(tensor: &Tensor, file: &str) -> TensorInfo {
    TensorInfo::new(
        tensor.name.clone(),
        normalize_dtype(tensor.dtype, tensor.ggml_type),
        TensorShape::from_innermost_first(&tensor.dims),
        tensor.byte_len,
        TensorLocation::new(file, tensor.relative_offset, Some(tensor.absolute_offset)),
    )
    .with_native_dtype(ggml_type_label(tensor.ggml_type))
}

/// Exact scalar equivalents normalize; every GGML block layout stays packed.
fn normalize_dtype(dtype: DType, ggml_type: u32) -> TensorDType {
    match dtype {
        DType::F32 => TensorDType::F32,
        DType::F16 => TensorDType::F16,
        DType::BF16 => TensorDType::BF16,
        DType::F64 => TensorDType::F64,
        DType::I8 => TensorDType::I8,
        DType::I16 => TensorDType::I16,
        DType::I32 => TensorDType::I32,
        DType::I64 => TensorDType::I64,
        _ => TensorDType::GgufPacked { ggml_type },
    }
}

/// GGUF typed maps → owned values. Signed entries win over the unsigned
/// shadow copy the parser also keeps. Arrays are not captured upstream,
/// and `general.alignment` is consumed as a layout field (see
/// [`CheckpointMetadata`] docs), so neither appears here.
fn metadata_values(metadata: &GgufMetadata) -> CheckpointMetadata {
    let mut out = CheckpointMetadata::new();
    for (key, value) in &metadata.strings {
        out.insert(key.clone(), MetadataValue::String(value.clone()));
    }
    for (key, &value) in &metadata.numerics {
        let value = match metadata.signed_numerics.get(key) {
            Some(&signed) => MetadataValue::Int(signed),
            None => MetadataValue::UInt(value),
        };
        out.insert(key.clone(), value);
    }
    for (key, &value) in &metadata.floats_32 {
        out.insert(key.clone(), MetadataValue::F32(value));
    }
    for (key, &value) in &metadata.floats_64 {
        out.insert(key.clone(), MetadataValue::F64(value));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn signed_metadata_is_not_reported_as_unsigned() {
        let mut meta = GgufMetadata::default();
        meta.numerics.insert("neg".into(), (-3i64) as u64);
        meta.signed_numerics.insert("neg".into(), -3);
        meta.numerics.insert("big".into(), u64::MAX);
        meta.floats_64.insert("eps".into(), 1e-6);
        let values = metadata_values(&meta);
        assert_eq!(values.get("neg"), Some(&MetadataValue::Int(-3)));
        assert_eq!(values.get("big"), Some(&MetadataValue::UInt(u64::MAX)));
        assert_eq!(values.get("eps"), Some(&MetadataValue::F64(1e-6)));
    }

    #[test]
    fn quantized_and_internal_types_stay_packed() {
        assert_eq!(
            normalize_dtype(DType::Q4_K, 12),
            TensorDType::GgufPacked { ggml_type: 12 }
        );
        assert_eq!(
            normalize_dtype(DType::IQ3_M_BLOCK, 0x4949_334D),
            TensorDType::GgufPacked {
                ggml_type: 0x4949_334D
            }
        );
        assert_eq!(normalize_dtype(DType::BF16, 30), TensorDType::BF16);
    }

    #[test]
    fn in_memory_source_uses_label() {
        let source = gguf_source("mem://unit", SourceKind::InMemory);
        assert_eq!(source.files, vec!["mem://unit".to_string()]);
        assert_eq!(source.root, PathBuf::new());
        let file = gguf_source("/tmp/x/model.gguf", SourceKind::SingleFile);
        assert_eq!(file.files, vec!["model.gguf".to_string()]);
        assert_eq!(file.root, PathBuf::from("/tmp/x"));
    }

    #[test]
    fn bare_filename_root_is_current_dir() {
        let source = gguf_source("model.gguf", SourceKind::SingleFile);
        assert_eq!(source.root, PathBuf::from("."));
        assert_eq!(source.files, vec!["model.gguf".to_string()]);
    }
}
