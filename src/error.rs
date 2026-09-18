// SPDX-License-Identifier: MIT OR Apache-2.0

//! Hand-rolled error types for `engram-parser`.
//!
//! Zero external dependencies: no `thiserror`, no `anyhow`. Callers may
//! match on [`ParserError`] variants or rely on the `std::error::Error`
//! trait for type-erased propagation.

use std::fmt;
use std::io;

/// Which [`crate::ParseLimits`] field was exhausted.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ParseLimitKind {
    /// [`crate::ParseLimits::max_kv_count`].
    KvCount,
    /// [`crate::ParseLimits::max_tensor_count`].
    TensorCount,
    /// [`crate::ParseLimits::max_string_bytes`].
    StringBytes,
    /// [`crate::ParseLimits::max_array_work_items`].
    ArrayWorkItems,
    /// [`crate::ParseLimits::max_tensor_rank`].
    TensorRank,
    /// [`crate::ParseLimits::max_metadata_bytes`].
    MetadataBytes,
}

impl ParseLimitKind {
    /// Stable name used in parser errors and tests.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::KvCount => "max_kv_count",
            Self::TensorCount => "max_tensor_count",
            Self::StringBytes => "max_string_bytes",
            Self::ArrayWorkItems => "max_array_work_items",
            Self::TensorRank => "max_tensor_rank",
            Self::MetadataBytes => "max_metadata_bytes",
        }
    }
}

impl fmt::Display for ParseLimitKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// File-declared field that failed `u64 -> usize` conversion.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum HostSizeField {
    /// Length prefix of a GGUF string (KV key, string value, or tensor name).
    StringLen,
    /// Header `kv_count`.
    KvCount,
    /// Header `tensor_count`.
    TensorCount,
    /// Tensor directory `n_dims`.
    TensorRank,
    /// One tensor dimension.
    TensorDim,
    /// Tensor `relative_offset` in the data region.
    RelativeOffset,
    /// `general.alignment` layout field.
    Alignment,
    /// GGUF array element count.
    ArrayLen,
}

impl HostSizeField {
    /// Stable name used in parser errors and tests.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::StringLen => "string_len",
            Self::KvCount => "kv_count",
            Self::TensorCount => "tensor_count",
            Self::TensorRank => "tensor_rank",
            Self::TensorDim => "tensor_dim",
            Self::RelativeOffset => "relative_offset",
            Self::Alignment => "general.alignment",
            Self::ArrayLen => "array_len",
        }
    }
}

impl fmt::Display for HostSizeField {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Unified error type for checkpoint parsing and MoE weight extraction.
#[derive(Debug)]
pub enum ParserError {
    /// The underlying file could not be read.
    Io {
        /// Path of the checkpoint being opened.
        path: String,
        /// Upstream `std::io::Error`.
        source: io::Error,
    },
    /// The file is not a valid checkpoint (bad magic, unsupported
    /// version, unknown value type, unexpected Safetensors layout, …).
    UnsupportedFormat {
        /// Path of the offending checkpoint.
        path: String,
        /// Human-readable explanation.
        reason: String,
    },
    /// A required tensor was not found in the checkpoint.
    MissingTensor {
        /// Tensor name (or suffix) that was looked up.
        name: String,
        /// Path of the checkpoint.
        path: String,
    },
    /// The checkpoint's on-disk layout is internally inconsistent
    /// (tensor extends past EOF, element-count overflow, etc.).
    InvalidLayout {
        /// Path of the checkpoint.
        path: String,
        /// Human-readable explanation.
        reason: String,
    },
    /// An expert index was out of range for the requested block or
    /// stacked tensor.
    ExpertOutOfRange {
        /// Block index that was queried.
        block: usize,
        /// Expert index that was queried.
        expert: usize,
        /// Number of experts actually available.
        available: usize,
    },
    /// A [`crate::ParseLimits`] budget was exhausted by a file-declared value.
    LimitExceeded {
        /// Path of the checkpoint.
        path: String,
        /// Budget that was exhausted.
        limit: ParseLimitKind,
        /// Declared or accumulated value that exceeded the budget.
        declared: u64,
        /// Configured budget for `limit`.
        budget: u64,
    },
    /// A file-declared `u64` could not be represented as a host `usize`.
    HostSizeOverflow {
        /// Path of the checkpoint.
        path: String,
        /// On-wire field that overflowed.
        field: HostSizeField,
        /// Declared value that does not fit in `usize`.
        value: u64,
    },
    /// A tensor name is claimed by more than one Safetensors shard.
    DuplicateTensorOwnership {
        /// Tensor name with conflicting owners.
        name: String,
        /// Checkpoint-relative shard paths that declare the tensor, sorted.
        shards: Vec<String>,
        /// Path of the checkpoint or index being inspected.
        path: String,
    },
    /// An index-referenced Safetensors shard is missing on disk.
    MissingShard {
        /// Shard path as declared in the index (checkpoint-relative).
        shard: String,
        /// Path of the index or checkpoint that referenced the shard.
        path: String,
    },
}

impl ParserError {
    pub(crate) fn limit_exceeded(
        path: impl Into<String>,
        limit: ParseLimitKind,
        declared: u64,
        budget: u64,
    ) -> Self {
        Self::LimitExceeded {
            path: path.into(),
            limit,
            declared,
            budget,
        }
    }

    pub(crate) fn host_size(path: impl Into<String>, field: HostSizeField, value: u64) -> Self {
        Self::HostSizeOverflow {
            path: path.into(),
            field,
            value,
        }
    }
}

impl fmt::Display for ParserError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io { path, source } => write!(f, "I/O error reading '{path}': {source}"),
            Self::UnsupportedFormat { path, reason } => {
                write!(f, "unsupported format in '{path}': {reason}")
            }
            Self::MissingTensor { name, path } => {
                write!(f, "missing tensor '{name}' in '{path}'")
            }
            Self::InvalidLayout { path, reason } => {
                write!(f, "invalid layout in '{path}': {reason}")
            }
            Self::ExpertOutOfRange {
                block,
                expert,
                available,
            } => write!(
                f,
                "expert index out of range: block={block}, expert={expert}, available={available}"
            ),
            Self::LimitExceeded {
                path,
                limit,
                declared,
                budget,
            } => write!(
                f,
                "parse limit {limit} exceeded in '{path}': declared {declared}, budget {budget}"
            ),
            Self::HostSizeOverflow { path, field, value } => write!(
                f,
                "host-size conversion failed in '{path}': field {field} value {value} does not fit usize"
            ),
            Self::DuplicateTensorOwnership { name, shards, path } => {
                write!(
                    f,
                    "duplicate tensor ownership for '{name}' across shards {} in '{path}'",
                    shards.join(", ")
                )
            }
            Self::MissingShard { shard, path } => {
                write!(f, "missing shard '{shard}' referenced by '{path}'")
            }
        }
    }
}

impl std::error::Error for ParserError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io { source, .. } => Some(source),
            Self::UnsupportedFormat { .. }
            | Self::MissingTensor { .. }
            | Self::InvalidLayout { .. }
            | Self::ExpertOutOfRange { .. }
            | Self::LimitExceeded { .. }
            | Self::HostSizeOverflow { .. }
            | Self::DuplicateTensorOwnership { .. }
            | Self::MissingShard { .. } => None,
        }
    }
}

/// Convenience alias used throughout the crate.
pub type Result<T> = std::result::Result<T, ParserError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn limit_and_host_errors_name_the_field() {
        let limit = ParserError::limit_exceeded("mem://e", ParseLimitKind::StringBytes, 17, 16);
        let host = ParserError::host_size("mem://e", HostSizeField::RelativeOffset, u64::MAX);
        assert!(limit.to_string().contains("max_string_bytes"));
        assert!(host.to_string().contains("relative_offset"));
        assert!(std::error::Error::source(&limit).is_none());
        assert!(std::error::Error::source(&host).is_none());
    }

    #[test]
    fn limit_kind_match_is_exhaustive() {
        let kinds = [
            ParseLimitKind::KvCount,
            ParseLimitKind::TensorCount,
            ParseLimitKind::StringBytes,
            ParseLimitKind::ArrayWorkItems,
            ParseLimitKind::TensorRank,
            ParseLimitKind::MetadataBytes,
        ];
        for kind in kinds {
            let name = match kind {
                ParseLimitKind::KvCount => "max_kv_count",
                ParseLimitKind::TensorCount => "max_tensor_count",
                ParseLimitKind::StringBytes => "max_string_bytes",
                ParseLimitKind::ArrayWorkItems => "max_array_work_items",
                ParseLimitKind::TensorRank => "max_tensor_rank",
                ParseLimitKind::MetadataBytes => "max_metadata_bytes",
            };
            assert_eq!(kind.as_str(), name);
        }
    }

    #[test]
    fn host_field_match_is_exhaustive() {
        let fields = [
            HostSizeField::StringLen,
            HostSizeField::KvCount,
            HostSizeField::TensorCount,
            HostSizeField::TensorRank,
            HostSizeField::TensorDim,
            HostSizeField::RelativeOffset,
            HostSizeField::Alignment,
            HostSizeField::ArrayLen,
        ];
        for field in fields {
            let name = match field {
                HostSizeField::StringLen => "string_len",
                HostSizeField::KvCount => "kv_count",
                HostSizeField::TensorCount => "tensor_count",
                HostSizeField::TensorRank => "tensor_rank",
                HostSizeField::TensorDim => "tensor_dim",
                HostSizeField::RelativeOffset => "relative_offset",
                HostSizeField::Alignment => "general.alignment",
                HostSizeField::ArrayLen => "array_len",
            };
            assert_eq!(field.as_str(), name);
        }
    }
}
