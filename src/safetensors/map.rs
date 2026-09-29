// SPDX-License-Identifier: MIT OR Apache-2.0
//! Memory-mapped Safetensors checkpoint access (`safetensors` + `mmap`).
//!
//! Maps each shard read-only and runs canonical validation through the
//! upstream `safetensors` crate at open time. Raw payload access returns
//! slices borrowed from the mappings, so large sharded checkpoints are
//! never copied into RAM.
//!
//! The upstream `SafeTensors` view is constructed and validated per shard
//! at open and then dropped: the struct only stores the owning [`Mmap`]
//! plus durable offsets, avoiding a self-referential borrow. Upstream
//! types never appear in the public API.

use super::checkpoint::{ResolvedShard, resolve_checkpoint_shards};
use super::manifest::{
    SafetensorsManifest, SafetensorsTensorRecord, inspect_safetensors_checkpoint,
};
use super::{io_error, model_load};
use crate::error::{ParserError, Result};
use memmap2::{Mmap, MmapOptions};
use std::collections::BTreeMap;
use std::path::Path;

/// A Safetensors checkpoint backed by read-only shard mappings.
///
/// Discovery, determinism, and path policy are identical to
/// [`super::open_safetensors_checkpoint`]; payload access returns
/// `&[u8]` borrowed from the mappings instead of an owned copy.
#[derive(Debug)]
pub struct SafetensorsCheckpointMmap {
    manifest: SafetensorsManifest,
    /// `source_shard` -> mapped shard.
    shards: BTreeMap<String, MappedShard>,
    /// tensor name -> index into `manifest.tensors`.
    lookup: BTreeMap<String, usize>,
}

#[derive(Debug)]
struct MappedShard {
    path: std::path::PathBuf,
    mmap: Mmap,
    /// Absolute byte offset of the data section (`8 + header_len`).
    data_begin: usize,
}

/// Open a Safetensors file, shard index, or directory with read-only
/// memory mappings and upstream-crate validation.
///
/// Requires both the `safetensors` and `mmap` cargo features.
pub fn open_safetensors_checkpoint_mmap(
    path: impl AsRef<Path>,
) -> Result<SafetensorsCheckpointMmap> {
    let manifest = inspect_safetensors_checkpoint(path.as_ref())?;
    let resolved = resolve_checkpoint_shards(path.as_ref(), &manifest)?;

    let mut shards = BTreeMap::new();
    for ResolvedShard {
        relative,
        path: shard_path,
        file,
        data_begin,
    } in resolved
    {
        // SAFETY: `file` is a readable regular-file descriptor opened at
        // checkpoint-open time and the mapping is read-only. Callers must
        // not truncate the file for the lifetime of the returned
        // checkpoint (standard mmap invariant). This crate never
        // host-registers the mapping.
        let file = file
            .into_inner()
            .unwrap_or_else(|poison| poison.into_inner());
        let mmap =
            unsafe { MmapOptions::new().map(&file) }.map_err(|e| io_error(&shard_path, e))?;
        // Canonical validation by the upstream crate: header structure,
        // dtype table, shape/dtype consistency, and full coverage of the
        // data section.
        let tensors = ::safetensors::SafeTensors::deserialize(&mmap)
            .map_err(|e| safetensors_error(&shard_path, e))?;
        // Ensure the upstream tensor table covers every manifest record
        // that resolves to this shard.
        let names: std::collections::BTreeSet<&str> =
            tensors.iter().map(|(name, _)| name).collect();
        for record in &manifest.tensors {
            if record.source_shard == relative && !names.contains(record.name.as_str()) {
                return Err(model_load(
                    &shard_path,
                    format!(
                        "upstream safetensors validation is missing tensor '{}'",
                        record.name
                    ),
                ));
            }
        }
        // `data_begin` is `8 + header_len` with `header_len` capped at
        // MAX_HEADER_BYTES, so it always fits in usize.
        let data_begin = data_begin as usize;
        shards.insert(
            relative,
            MappedShard {
                path: shard_path,
                mmap,
                data_begin,
            },
        );
    }

    let lookup = manifest
        .tensors
        .iter()
        .enumerate()
        .map(|(index, tensor)| (tensor.name.clone(), index))
        .collect();
    Ok(SafetensorsCheckpointMmap {
        manifest,
        shards,
        lookup,
    })
}

impl SafetensorsCheckpointMmap {
    /// The deterministic manifest produced at open time.
    pub fn manifest(&self) -> &SafetensorsManifest {
        &self.manifest
    }

    /// Look up a tensor record by exact name.
    pub fn tensor(&self, name: &str) -> Result<&SafetensorsTensorRecord> {
        let index = self
            .lookup
            .get(name)
            .ok_or_else(|| ParserError::MissingTensor {
                name: name.to_owned(),
                path: self
                    .shards
                    .values()
                    .next()
                    .map(|shard| shard.path.display().to_string())
                    .unwrap_or_else(|| "<safetensors>".to_string()),
            })?;
        Ok(&self.manifest.tensors[*index])
    }

    /// Absolute file offset of `source_shard`'s data section (`8 + header_len`).
    pub(crate) fn shard_data_begin(&self, source_shard: &str) -> Option<u64> {
        self.shards
            .get(source_shard)
            .map(|shard| shard.data_begin as u64)
    }

    /// Borrowed raw payload bytes of `name`, validated at open time.
    ///
    /// The returned slice borrows from the owning shard's mapping; no
    /// copy is made.
    pub fn tensor_bytes<'a>(&'a self, name: &str) -> Result<&'a [u8]> {
        let tensor = self.tensor(name)?;
        let shard = self.shards.get(&tensor.source_shard).ok_or_else(|| {
            model_load(
                Path::new(&tensor.source_shard),
                format!("tensor '{name}' has no resolved shard"),
            )
        })?;
        // `data_offsets` were validated against this shard's data
        // section at open (`end <= data_len`); `data_begin + end` is
        // bounded by file_len and cannot overflow. `Mmap::get` is the
        // final bounds check.
        let start = shard.data_begin + tensor.data_offsets[0];
        let end = shard.data_begin + tensor.data_offsets[1];
        shard.mmap.get(start..end).ok_or_else(|| {
            model_load(
                &shard.path,
                format!("tensor '{name}' payload range exceeds the mapping"),
            )
        })
    }
}

/// Map an upstream `safetensors` error onto the engram error surface.
pub(super) fn safetensors_error(path: &Path, error: ::safetensors::SafeTensorError) -> ParserError {
    use ::safetensors::SafeTensorError as E;
    match error {
        E::TensorNotFound(name) => ParserError::MissingTensor {
            name,
            path: path.display().to_string(),
        },
        E::InvalidHeader(_) | E::InvalidHeaderDeserialization(_) | E::JsonError(_) => {
            ParserError::UnsupportedFormat {
                path: path.display().to_string(),
                reason: error.to_string(),
            }
        }
        E::IoError(source) => ParserError::Io {
            path: path.display().to_string(),
            source,
        },
        other => ParserError::InvalidLayout {
            path: path.display().to_string(),
            reason: other.to_string(),
        },
    }
}
