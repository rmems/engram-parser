// SPDX-License-Identifier: MIT OR Apache-2.0

//! Hand-rolled error types for `engram-parser`.
//!
//! Zero external dependencies: no `thiserror`, no `anyhow`. Callers may
//! match on [`ParserError`] variants or rely on the `std::error::Error`
//! trait for type-erased propagation.

use std::fmt;
use std::io;

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
        }
    }
}

impl std::error::Error for ParserError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io { source, .. } => Some(source),
            _ => None,
        }
    }
}

/// Convenience alias used throughout the crate.
pub type Result<T> = std::result::Result<T, ParserError>;
