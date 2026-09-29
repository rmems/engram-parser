// SPDX-License-Identifier: MIT OR Apache-2.0
//! Format-independent checkpoint and tensor access.
//!
//! [`Checkpoint`] is one consumer-facing contract over GGUF and
//! Safetensors. Research code can inventory tensors, read metadata, and
//! fetch raw payload bytes without branching on the container format:
//!
//! ```no_run
//! use engram_parser::{Checkpoint, open_checkpoint};
//!
//! fn inventory(ckpt: &dyn Checkpoint) -> engram_parser::Result<()> {
//!     println!("{} from {}", ckpt.format(), ckpt.source().path.display());
//!     for t in ckpt.tensors() {
//!         println!("{} {} {:?} {}B", t.name, t.dtype, t.shape.dims(), t.byte_len);
//!     }
//!     if let Some(first) = ckpt.tensors().first() {
//!         assert_eq!(ckpt.tensor_bytes(&first.name)?.len(), first.byte_len);
//!     }
//!     Ok(())
//! }
//!
//! let ckpt = open_checkpoint("model.gguf")?;
//! inventory(&ckpt)?;
//! # Ok::<(), engram_parser::ParserError>(())
//! ```
//!
//! # Backends
//!
//! The backends are thin adapters over the existing format modules, which
//! stay fully reachable:
//!
//! | Backend | Wraps | Feature | Payload |
//! |---|---|---|---|
//! | [`GgufBackend`] | [`crate::GgufLayout`] | default | borrowed |
//! | `GgufMmapBackend` | `GgufLayoutMmap` | `mmap` | borrowed |
//! | `SafetensorsBackend` | `safetensors::SafetensorsCheckpoint` | `safetensors` | owned (bounded read) |
//! | `SafetensorsMmapBackend` | `safetensors::SafetensorsCheckpointMmap` | `safetensors` + `mmap` | borrowed |
//!
//! [`open_checkpoint`] picks a backend at runtime and returns an
//! [`AnyCheckpoint`], whose `as_*` accessors hand back the
//! format-specific handle (for example `GgufLayout` for
//! [`crate::list_experts`] / [`crate::extract_expert`]).
//!
//! # Contract
//!
//! - **Inventory order.** [`Checkpoint::tensors`] is sorted by name and
//!   identical across calls and across owned/mmap backends of one file.
//! - **Shape order.** [`TensorShape::dims`] is outermost-first for every
//!   format. GGUF dims (innermost-first on disk) are reversed;
//!   [`TensorShape::native_dims`] restores the on-disk order.
//! - **Dtypes.** Plain scalar types normalize to shared variants. GGML
//!   quantized layouts stay [`TensorDType::GgufPacked`] with the exact
//!   wire code; [`TensorInfo::native_dtype`] keeps the source label.
//! - **Payloads.** [`Checkpoint::tensor_bytes`] returns raw bytes exactly
//!   as stored (no decode, no dequant, packed blocks intact), with length
//!   equal to [`TensorInfo::byte_len`]. Borrowed for GGUF and mmap
//!   backends; owned for the bounded-read Safetensors backend.
//! - **Location.** [`TensorLocation::source`] is relative to
//!   [`CheckpointSource::root`]; `data_offset` is relative to the file's
//!   tensor-data section; `file_offset` is absolute when known.
//! - **Errors.** Unknown names return [`crate::ParserError::MissingTensor`].
//!
//! This module has no numerical execution, device, or MoE concepts.
//! Model topology and role discovery are out of scope.

use std::borrow::Cow;

use crate::error::{ParserError, Result};

mod gguf;
mod open;
#[cfg(feature = "safetensors")]
mod safetensors;
mod types;

pub use gguf::GgufBackend;
#[cfg(feature = "mmap")]
pub use gguf::GgufMmapBackend;
#[cfg(feature = "mmap")]
pub use open::open_checkpoint_mmap;
pub use open::{AnyCheckpoint, open_checkpoint};
#[cfg(feature = "safetensors")]
pub use safetensors::SafetensorsBackend;
#[cfg(all(feature = "safetensors", feature = "mmap"))]
pub use safetensors::SafetensorsMmapBackend;
pub use types::{
    CheckpointFormat, CheckpointMetadata, CheckpointSource, DimOrder, MetadataValue, SourceKind,
    TensorDType, TensorInfo, TensorLocation, TensorShape,
};

/// Consumer-facing, format-independent checkpoint contract.
///
/// Object safe: use `&dyn Checkpoint` or generics. See the
/// [module docs](self) for ordering, shape, dtype, and payload guarantees.
pub trait Checkpoint: Send + Sync {
    /// Checkpoint identity: format, input path, root, payload files.
    fn source(&self) -> &CheckpointSource;

    /// Container format.
    fn format(&self) -> CheckpointFormat {
        self.source().format
    }

    /// Every tensor, sorted by name.
    fn tensors(&self) -> &[TensorInfo];

    /// Look up one tensor by exact name.
    fn tensor(&self, name: &str) -> Option<&TensorInfo> {
        let tensors = self.tensors();
        tensors
            .binary_search_by(|t| t.name.as_str().cmp(name))
            .ok()
            .map(|index| &tensors[index])
    }

    /// Scalar metadata.
    fn metadata(&self) -> &CheckpointMetadata;

    /// Raw payload bytes of `name`, length [`TensorInfo::byte_len`].
    fn tensor_bytes(&self, name: &str) -> Result<Cow<'_, [u8]>>;
}

/// Shared, precomputed state behind every backend.
#[derive(Debug, Clone)]
pub(crate) struct Catalog {
    source: CheckpointSource,
    tensors: Vec<TensorInfo>,
    metadata: CheckpointMetadata,
}

impl Catalog {
    pub(crate) fn new(
        source: CheckpointSource,
        mut tensors: Vec<TensorInfo>,
        metadata: CheckpointMetadata,
    ) -> Self {
        tensors.sort_by(|a, b| a.name.cmp(&b.name));
        Self {
            source,
            tensors,
            metadata,
        }
    }

    /// Fail with `MissingTensor` unless `name` is in the inventory.
    pub(crate) fn require(&self, name: &str) -> Result<&TensorInfo> {
        self.tensors
            .binary_search_by(|t| t.name.as_str().cmp(name))
            .map(|index| &self.tensors[index])
            .map_err(|_| ParserError::MissingTensor {
                name: name.to_owned(),
                path: self.source.path.display().to_string(),
            })
    }

    /// Reject a backend payload whose length disagrees with the inventory.
    pub(crate) fn check_len(&self, info: &TensorInfo, len: usize) -> Result<()> {
        if len == info.byte_len {
            return Ok(());
        }
        Err(ParserError::InvalidLayout {
            path: self.source.path.display().to_string(),
            reason: format!(
                "tensor '{}' payload is {len} bytes, inventory says {}",
                info.name, info.byte_len
            ),
        })
    }
}

/// Implement the inventory half of [`Checkpoint`] via a `catalog` field.
macro_rules! delegate_catalog {
    () => {
        fn source(&self) -> &$crate::checkpoint::CheckpointSource {
            &self.catalog.source
        }

        fn tensors(&self) -> &[$crate::checkpoint::TensorInfo] {
            &self.catalog.tensors
        }

        fn metadata(&self) -> &$crate::checkpoint::CheckpointMetadata {
            &self.catalog.metadata
        }
    };
}
pub(crate) use delegate_catalog;
