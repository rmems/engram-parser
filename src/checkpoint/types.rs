// SPDX-License-Identifier: MIT OR Apache-2.0
//! Engram-owned, format-independent checkpoint value types.
//!
//! Every type here is plain data built from std types only. No upstream
//! (`safetensors`, `memmap2`) type appears in these signatures, and there
//! are no GPU, execution, or MoE concepts.

use std::collections::BTreeMap;
use std::fmt;
use std::path::PathBuf;

use crate::gguf::ggml_type_label;

/// On-disk checkpoint container format.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum CheckpointFormat {
    /// GGUF v3 single-file checkpoint.
    Gguf,
    /// Safetensors single file, Hugging Face shard index, or shard directory.
    Safetensors,
}

impl CheckpointFormat {
    /// Stable lowercase name (`"gguf"`, `"safetensors"`).
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Gguf => "gguf",
            Self::Safetensors => "safetensors",
        }
    }
}

impl fmt::Display for CheckpointFormat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// How the checkpoint input was laid out on disk.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SourceKind {
    /// One file holds every tensor (`.gguf`, single `.safetensors`).
    SingleFile,
    /// A Hugging Face `*.safetensors.index.json` shard index.
    ShardIndex,
    /// A directory of `.safetensors` shards.
    Directory,
    /// An in-memory buffer (for example [`crate::parse_bytes`]).
    InMemory,
}

impl SourceKind {
    /// Stable snake_case name.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::SingleFile => "single_file",
            Self::ShardIndex => "shard_index",
            Self::Directory => "directory",
            Self::InMemory => "in_memory",
        }
    }
}

impl fmt::Display for SourceKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Identity of an opened checkpoint: format, input path, and payload files.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CheckpointSource {
    /// Container format.
    pub format: CheckpointFormat,
    /// Input kind (single file, shard index, directory, in-memory).
    pub kind: SourceKind,
    /// Path the checkpoint was opened from, as given by the caller. For
    /// in-memory GGUF this is the caller-supplied label.
    pub path: PathBuf,
    /// Root that [`TensorLocation::source`] paths are relative to.
    pub root: PathBuf,
    /// Root-relative index file, when [`SourceKind::ShardIndex`].
    pub index_file: Option<String>,
    /// Sorted, de-duplicated root-relative files that hold tensor payloads.
    pub files: Vec<String>,
}

/// Order of [`TensorShape::dims`] in the format's native directory.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DimOrder {
    /// Row-major, outermost dimension first (Safetensors, NumPy, PyTorch).
    OutermostFirst,
    /// Innermost (fastest-varying) dimension first (GGUF / GGML `ne[]`).
    InnermostFirst,
}

/// Tensor shape normalized to **outermost-first** order.
///
/// GGUF stores dims innermost-first (`ne[0]` is the contiguous row
/// length). Those dims are reversed here, so a GGUF `[4096, 32000]`
/// token embedding becomes `[32000, 4096]`, which is the same order a
/// Safetensors header uses. [`Self::native_dims`] recovers the on-disk
/// order losslessly.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct TensorShape {
    dims: Vec<usize>,
    native_order: DimOrder,
}

impl TensorShape {
    /// Build a shape from dims that are already outermost-first.
    pub fn outermost_first(dims: Vec<usize>) -> Self {
        Self {
            dims,
            native_order: DimOrder::OutermostFirst,
        }
    }

    /// Build a shape from native innermost-first dims (GGUF `Tensor::dims`).
    pub fn from_innermost_first(native: &[usize]) -> Self {
        Self {
            dims: native.iter().rev().copied().collect(),
            native_order: DimOrder::InnermostFirst,
        }
    }

    /// Dims in outermost-first order.
    pub fn dims(&self) -> &[usize] {
        &self.dims
    }

    /// Order the source format stores dims in.
    pub fn native_order(&self) -> DimOrder {
        self.native_order
    }

    /// Dims in the source format's native order.
    pub fn native_dims(&self) -> Vec<usize> {
        match self.native_order {
            DimOrder::OutermostFirst => self.dims.clone(),
            DimOrder::InnermostFirst => self.dims.iter().rev().copied().collect(),
        }
    }

    /// Number of dimensions.
    pub fn rank(&self) -> usize {
        self.dims.len()
    }

    /// Product of dims, or `None` on overflow. A rank-0 shape has one element.
    pub fn element_count(&self) -> Option<usize> {
        self.dims
            .iter()
            .try_fold(1usize, |acc, &d| acc.checked_mul(d))
    }
}

/// Normalized tensor element type.
///
/// Scalar variants are shared by both formats. Packed GGML quantized
/// layouts keep their exact wire code in [`TensorDType::GgufPacked`], so
/// they are never conflated with a Safetensors label. Unrecognized labels
/// are preserved in [`TensorDType::Other`].
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum TensorDType {
    /// 64-bit float.
    F64,
    /// 32-bit float.
    F32,
    /// IEEE-754 half float.
    F16,
    /// bfloat16.
    BF16,
    /// 8-bit float, E4M3.
    F8E4M3,
    /// 8-bit float, E5M2.
    F8E5M2,
    /// Signed 64-bit integer.
    I64,
    /// Signed 32-bit integer.
    I32,
    /// Signed 16-bit integer.
    I16,
    /// Signed 8-bit integer.
    I8,
    /// Unsigned 64-bit integer.
    U64,
    /// Unsigned 32-bit integer.
    U32,
    /// Unsigned 16-bit integer.
    U16,
    /// Unsigned 8-bit integer.
    U8,
    /// One-byte boolean.
    Bool,
    /// Signed 4-bit integer, two elements per byte (Safetensors `I4`/`INT4`).
    I4,
    /// Unsigned 4-bit integer, two elements per byte (Safetensors `U4`).
    U4,
    /// A packed GGML quantized block layout (Q4_K, Q8_0, IQ*, …). The raw
    /// on-wire `ggml_type` code is preserved; payload bytes keep the native
    /// block layout.
    GgufPacked {
        /// GGUF on-wire `ggml_type` code.
        ggml_type: u32,
    },
    /// A dtype label this crate does not normalize (e.g. `C64`, `F8_E8M0`).
    Other(String),
}

impl TensorDType {
    /// Short label (`"F32"`, `"Q4_K"`, or the preserved [`Self::Other`] label).
    pub fn label(&self) -> &str {
        match self {
            Self::F64 => "F64",
            Self::F32 => "F32",
            Self::F16 => "F16",
            Self::BF16 => "BF16",
            Self::F8E4M3 => "F8_E4M3",
            Self::F8E5M2 => "F8_E5M2",
            Self::I64 => "I64",
            Self::I32 => "I32",
            Self::I16 => "I16",
            Self::I8 => "I8",
            Self::U64 => "U64",
            Self::U32 => "U32",
            Self::U16 => "U16",
            Self::U8 => "U8",
            Self::Bool => "BOOL",
            Self::I4 => "I4",
            Self::U4 => "U4",
            Self::GgufPacked { ggml_type } => ggml_type_label(*ggml_type),
            Self::Other(label) => label,
        }
    }

    /// `true` for sub-byte or block layouts where bytes ≠ elements × width.
    pub fn is_packed(&self) -> bool {
        matches!(self, Self::I4 | Self::U4 | Self::GgufPacked { .. })
    }

    /// Byte width of one element for plain scalar dtypes; `None` for packed
    /// and unknown dtypes.
    pub fn element_size(&self) -> Option<usize> {
        match self {
            Self::F64 | Self::I64 | Self::U64 => Some(8),
            Self::F32 | Self::I32 | Self::U32 => Some(4),
            Self::F16 | Self::BF16 | Self::I16 | Self::U16 => Some(2),
            Self::F8E4M3 | Self::F8E5M2 | Self::I8 | Self::U8 | Self::Bool => Some(1),
            Self::I4 | Self::U4 | Self::GgufPacked { .. } | Self::Other(_) => None,
        }
    }
}

impl fmt::Display for TensorDType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.label())
    }
}

/// Where a tensor payload lives.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct TensorLocation {
    /// File holding the payload, relative to [`CheckpointSource::root`].
    pub source: String,
    /// Payload start relative to that file's tensor-data section.
    pub data_offset: usize,
    /// Payload start from the beginning of the file, when the backend knows it.
    pub file_offset: Option<usize>,
}

impl TensorLocation {
    /// Build a location.
    pub fn new(source: impl Into<String>, data_offset: usize, file_offset: Option<usize>) -> Self {
        Self {
            source: source.into(),
            data_offset,
            file_offset,
        }
    }
}

/// Format-independent description of one tensor.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct TensorInfo {
    /// Exact tensor name.
    pub name: String,
    /// Normalized dtype.
    pub dtype: TensorDType,
    /// Dtype label exactly as the source format names it
    /// (Safetensors header string, or GGUF [`ggml_type_label`]).
    pub native_dtype: String,
    /// Outermost-first shape (native order recoverable).
    pub shape: TensorShape,
    /// Raw payload length in bytes (packed length for packed dtypes).
    pub byte_len: usize,
    /// Payload location.
    pub location: TensorLocation,
}

impl TensorInfo {
    /// Build a tensor description, e.g. for mock inventories in downstream
    /// tests. `native_dtype` defaults to [`TensorDType::label`].
    pub fn new(
        name: impl Into<String>,
        dtype: TensorDType,
        shape: TensorShape,
        byte_len: usize,
        location: TensorLocation,
    ) -> Self {
        let native_dtype = dtype.label().to_owned();
        Self {
            name: name.into(),
            dtype,
            native_dtype,
            shape,
            byte_len,
            location,
        }
    }

    /// Override the native dtype label.
    #[must_use]
    pub fn with_native_dtype(mut self, native_dtype: impl Into<String>) -> Self {
        self.native_dtype = native_dtype.into();
        self
    }
}

/// Owned scalar metadata value.
///
/// GGUF numeric metadata is exposed as the parser stores it: unsigned and
/// bool values as [`Self::UInt`], signed integers as [`Self::Int`], and
/// floats at their stored width. GGUF arrays are not captured by the
/// parser and are absent. Safetensors metadata is string-only.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq)]
pub enum MetadataValue {
    /// String value.
    String(String),
    /// Unsigned integer (GGUF `UINT*` and `BOOL`).
    UInt(u64),
    /// Signed integer (GGUF `INT*`).
    Int(i64),
    /// 32-bit float.
    F32(f32),
    /// 64-bit float.
    F64(f64),
}

impl MetadataValue {
    /// String payload, if this is a string.
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Self::String(s) => Some(s),
            _ => None,
        }
    }

    /// Integer payload as `u64`, if it is a non-negative integer.
    pub fn as_u64(&self) -> Option<u64> {
        match *self {
            Self::UInt(v) => Some(v),
            Self::Int(v) => u64::try_from(v).ok(),
            _ => None,
        }
    }

    /// Numeric payload widened to `f64`.
    pub fn as_f64(&self) -> Option<f64> {
        match *self {
            Self::F32(v) => Some(f64::from(v)),
            Self::F64(v) => Some(v),
            Self::UInt(v) => Some(v as f64),
            Self::Int(v) => Some(v as f64),
            Self::String(_) => None,
        }
    }
}

impl fmt::Display for MetadataValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::String(s) => f.write_str(s),
            Self::UInt(v) => write!(f, "{v}"),
            Self::Int(v) => write!(f, "{v}"),
            Self::F32(v) => write!(f, "{v}"),
            Self::F64(v) => write!(f, "{v}"),
        }
    }
}

/// Sorted metadata key-value store.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct CheckpointMetadata {
    entries: BTreeMap<String, MetadataValue>,
}

impl CheckpointMetadata {
    /// Empty store.
    pub fn new() -> Self {
        Self::default()
    }

    /// Insert or replace one entry.
    pub fn insert(&mut self, key: impl Into<String>, value: MetadataValue) {
        self.entries.insert(key.into(), value);
    }

    /// Look up a value by exact key.
    pub fn get(&self, key: &str) -> Option<&MetadataValue> {
        self.entries.get(key)
    }

    /// Number of entries.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the store is empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Keys in sorted order.
    pub fn keys(&self) -> impl Iterator<Item = &str> {
        self.entries.keys().map(String::as_str)
    }

    /// Entries in sorted key order.
    pub fn iter(&self) -> impl Iterator<Item = (&str, &MetadataValue)> {
        self.entries.iter().map(|(k, v)| (k.as_str(), v))
    }
}

impl FromIterator<(String, MetadataValue)> for CheckpointMetadata {
    fn from_iter<I: IntoIterator<Item = (String, MetadataValue)>>(iter: I) -> Self {
        Self {
            entries: iter.into_iter().collect(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gguf_shape_round_trips_native_order() {
        let shape = TensorShape::from_innermost_first(&[4096, 32000]);
        assert_eq!(shape.dims(), &[32000, 4096]);
        assert_eq!(shape.native_dims(), vec![4096, 32000]);
        assert_eq!(shape.native_order(), DimOrder::InnermostFirst);
        assert_eq!(shape.element_count(), Some(4096 * 32000));
        assert_eq!(
            TensorShape::outermost_first(vec![]).element_count(),
            Some(1)
        );
        assert_eq!(
            TensorShape::outermost_first(vec![usize::MAX, 2]).element_count(),
            None
        );
    }

    #[test]
    fn dtype_labels_and_packing() {
        let q4k = TensorDType::GgufPacked { ggml_type: 12 };
        assert_eq!(q4k.label(), "Q4_K");
        assert!(q4k.is_packed());
        assert_eq!(q4k.element_size(), None);
        assert_eq!(TensorDType::BF16.element_size(), Some(2));
        assert!(TensorDType::I4.is_packed());
        assert_eq!(TensorDType::Other("C64".into()).label(), "C64");
    }

    #[test]
    fn metadata_value_accessors() {
        assert_eq!(MetadataValue::Int(-1).as_u64(), None);
        assert_eq!(MetadataValue::Int(7).as_u64(), Some(7));
        assert_eq!(MetadataValue::F32(0.5).as_f64(), Some(0.5));
        assert_eq!(MetadataValue::String("x".into()).as_str(), Some("x"));
        let meta: CheckpointMetadata = [
            ("b".to_string(), MetadataValue::UInt(2)),
            ("a".to_string(), MetadataValue::UInt(1)),
        ]
        .into_iter()
        .collect();
        assert_eq!(meta.keys().collect::<Vec<_>>(), vec!["a", "b"]);
    }
}
