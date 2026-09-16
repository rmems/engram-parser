// SPDX-License-Identifier: MIT OR Apache-2.0
//! Header-only Safetensors inspection, deterministic manifests, and MoE
//! candidate discovery.
//!
//! Gated behind the off-by-default `safetensors` cargo feature. This module
//! is a one-way generalization of the reusable metadata surface in
//! `corinth-canal/src/moe/safetensors/` (`discovery`, `json`, `paths`,
//! `manifest`, `validate`). Corinth-specific `config` (HF `config.json`)
//! and `map` (payload mmap / token extract) are not ported.
//!
//! Zero-dependency: the upstream `safetensors` crate, `serde_json`, and
//! `corinth-canal` are forbidden. Header and shard-index JSON are parsed
//! in-crate.
//!
//! ```no_run
//! use engram_parser::safetensors::inspect_safetensors_checkpoint;
//!
//! let manifest = inspect_safetensors_checkpoint("model.safetensors")?;
//! println!("tensors = {}", manifest.checkpoint.tensor_count);
//! # Ok::<(), engram_parser::ParserError>(())
//! ```

mod discovery;
mod json;
mod manifest;
mod paths;
mod validate;

#[cfg(test)]
mod tests;

use crate::error::ParserError;
use std::path::Path;

pub use discovery::{
    SafetensorsCandidateSummary, SafetensorsExpertGroup, SafetensorsRouterCandidate,
    classify_tensor, discover_candidates,
};
pub use manifest::{
    INDEX_UNREFERENCED_SHARDS_KEY, SafetensorsCheckpointSource, SafetensorsManifest,
    SafetensorsTensorRecord, inspect_safetensors_checkpoint, write_safetensors_manifest,
};
pub use validate::dtype_size_bytes;

pub(super) fn relative_path(path: &Path, root: &Path) -> String {
    let stripped = path.strip_prefix(root).unwrap_or(path);
    let lossy = stripped.to_string_lossy();
    #[cfg(windows)]
    {
        lossy.replace('\\', "/")
    }
    #[cfg(not(windows))]
    {
        lossy.into_owned()
    }
}

pub(super) fn model_load(path: &Path, reason: String) -> ParserError {
    ParserError::InvalidLayout {
        path: path.display().to_string(),
        reason,
    }
}

pub(super) fn io_error(path: &Path, source: std::io::Error) -> ParserError {
    ParserError::Io {
        path: path.display().to_string(),
        source,
    }
}

pub(super) fn unsupported(path: &Path, reason: String) -> ParserError {
    ParserError::UnsupportedFormat {
        path: path.display().to_string(),
        reason,
    }
}

pub(super) fn duplicate_tensor_ownership(
    path: &Path,
    name: impl Into<String>,
    mut shards: Vec<String>,
) -> ParserError {
    shards.sort();
    shards.dedup();
    ParserError::DuplicateTensorOwnership {
        name: name.into(),
        shards,
        path: path.display().to_string(),
    }
}

pub(super) fn missing_shard(path: &Path, shard: impl Into<String>) -> ParserError {
    ParserError::MissingShard {
        shard: shard.into(),
        path: path.display().to_string(),
    }
}
