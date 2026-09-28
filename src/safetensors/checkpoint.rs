// SPDX-License-Identifier: MIT OR Apache-2.0
//! Raw tensor payload access over a discovered Safetensors checkpoint.
//!
//! [`open_safetensors_checkpoint`] reuses the same deterministic
//! single-file / shard-index / directory discovery as
//! [`inspect_safetensors_checkpoint`], then serves validated raw payload
//! bytes by tensor name. Each lookup reads only that tensor's byte range,
//! so sharded checkpoints do not require whole-checkpoint copies.
//!
//! Engram owns checkpoint discovery, shard/path policy, and the public
//! tensor representation; the upstream `safetensors` crate is used for
//! canonical header validation where a whole shard buffer is available
//! (see the `mmap` backend in [`super::map`]).

use super::manifest::{
    MAX_HEADER_BYTES, SafetensorsManifest, SafetensorsTensorRecord, inspect_safetensors_checkpoint,
    list_safetensors_files,
};
use super::paths::parent_or_current;
use super::{io_error, model_load, relative_path};
use crate::error::{ParserError, Result};
use std::collections::BTreeMap;
use std::fs::{self, File};
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

/// One shard resolved from a checkpoint manifest.
#[derive(Debug)]
pub(super) struct ResolvedShard {
    /// Checkpoint-relative shard path (matches `source_shard`).
    pub(super) relative: String,
    /// Absolute shard path under the checkpoint root.
    pub(super) path: PathBuf,
    /// Handle opened at checkpoint-open time; keeps reads pinned to the
    /// inspected inode even if the path is later replaced.
    pub(super) file: File,
    /// Absolute byte offset of the data section (`8 + header_len`).
    pub(super) data_begin: u64,
}

/// A Safetensors checkpoint open for raw payload reads.
///
/// Holds the deterministic [`SafetensorsManifest`] plus per-shard access
/// metadata. Tensor payloads are read on demand with a seek + bounded
/// `read_exact`, so memory use is limited to the requested tensor.
#[derive(Debug)]
pub struct SafetensorsCheckpoint {
    manifest: SafetensorsManifest,
    /// `source_shard` -> resolved shard.
    shards: BTreeMap<String, ResolvedShard>,
    /// tensor name -> index into `manifest.tensors`.
    lookup: BTreeMap<String, usize>,
}

/// Open a Safetensors file, shard index, or directory for raw payload
/// reads. Discovery and validation are identical to
/// [`inspect_safetensors_checkpoint`]: headers are checked by engram's
/// fail-closed parser (duplicate-key rejection, contiguous
/// `data_offsets`, dtype/shape byte sizes). Upstream-crate canonical
/// validation runs per shard in the `mmap` backend
/// ([`super::map`]) — upstream `SafeTensors::read_metadata` requires a
/// whole-shard buffer, so it cannot run here without breaking the
/// bounded-memory contract.
pub fn open_safetensors_checkpoint(path: impl AsRef<Path>) -> Result<SafetensorsCheckpoint> {
    let manifest = inspect_safetensors_checkpoint(path.as_ref())?;
    let shards = resolve_checkpoint_shards(path.as_ref(), &manifest)?;
    Ok(SafetensorsCheckpoint::new(manifest, shards))
}

impl SafetensorsCheckpoint {
    pub(super) fn new(
        manifest: SafetensorsManifest,
        shards: Vec<ResolvedShard>,
    ) -> SafetensorsCheckpoint {
        let lookup = manifest
            .tensors
            .iter()
            .enumerate()
            .map(|(index, tensor)| (tensor.name.clone(), index))
            .collect();
        let shards = shards
            .into_iter()
            .map(|shard| (shard.relative.clone(), shard))
            .collect();
        SafetensorsCheckpoint {
            manifest,
            shards,
            lookup,
        }
    }

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
                path: self.display_path(),
            })?;
        Ok(&self.manifest.tensors[*index])
    }

    /// Read the validated raw payload bytes of `name`.
    ///
    /// Reads only `data_offsets[0]..data_offsets[1]` of the owning shard,
    /// so memory use is bounded by the tensor's byte size.
    pub fn tensor_bytes(&self, name: &str) -> Result<Vec<u8>> {
        let tensor = self.tensor(name)?;
        let shard = self.shard_for(tensor)?;
        let mut file = shard
            .file
            .try_clone()
            .map_err(|e| io_error(&shard.path, e))?;
        let mut bytes = vec![0u8; tensor.byte_size];
        let start = shard
            .data_begin
            .checked_add(tensor.data_offsets[0] as u64)
            .ok_or_else(|| {
                model_load(
                    &shard.path,
                    format!("tensor '{name}' payload offset overflow"),
                )
            })?;
        file.seek(SeekFrom::Start(start))
            .and_then(|_| file.read_exact(&mut bytes))
            .map_err(|e| io_error(&shard.path, e))?;
        Ok(bytes)
    }

    /// Borrowed view of the payload bytes for a discovery result.
    ///
    /// Convenience wrapper over [`Self::tensor_bytes`] for names produced
    /// by [`SafetensorsManifest::candidates`] or
    /// [`SafetensorsTensorRecord::name`].
    pub fn resolve_tensor_bytes(&self, record: &SafetensorsTensorRecord) -> Result<Vec<u8>> {
        self.tensor_bytes(&record.name)
    }

    fn display_path(&self) -> String {
        self.shards
            .values()
            .next()
            .map(|shard| shard.path.display().to_string())
            .unwrap_or_else(|| "<safetensors>".to_string())
    }

    fn shard_for(&self, tensor: &SafetensorsTensorRecord) -> Result<&ResolvedShard> {
        self.shards.get(&tensor.source_shard).ok_or_else(|| {
            model_load(
                Path::new(&tensor.source_shard),
                format!("tensor '{}' has no resolved shard", tensor.name),
            )
        })
    }
}

/// Resolve every shard referenced by `manifest` under the checkpoint
/// root derived from `path`.
pub(super) fn resolve_checkpoint_shards(
    path: &Path,
    manifest: &SafetensorsManifest,
) -> Result<Vec<ResolvedShard>> {
    let metadata = fs::metadata(path).map_err(|e| io_error(path, e))?;
    let root = if metadata.is_dir() {
        path.to_path_buf()
    } else {
        parent_or_current(path).to_path_buf()
    };

    // Manifest `source_shard` strings are lossy; recover the real
    // discovered `PathBuf` so non-UTF-8 shard names still resolve.
    let discovered: BTreeMap<String, PathBuf> = list_safetensors_files(&root)?
        .into_iter()
        .map(|path| (relative_path(&path, &root), path))
        .collect();

    let mut shard_paths: Vec<String> = manifest
        .tensors
        .iter()
        .map(|tensor| tensor.source_shard.clone())
        .collect();
    shard_paths.sort();
    shard_paths.dedup();

    shard_paths
        .into_iter()
        .map(|relative| {
            let shard_path = discovered
                .get(&relative)
                .cloned()
                .unwrap_or_else(|| root.join(&relative));
            let mut file = File::open(&shard_path).map_err(|e| io_error(&shard_path, e))?;
            let data_begin = shard_data_begin(&shard_path, &mut file)?;
            Ok(ResolvedShard {
                relative,
                path: shard_path,
                file,
                data_begin,
            })
        })
        .collect()
}

/// Absolute offset of the data section: `8 + header_len`, validated
/// against the header-size cap and the file length.
pub(super) fn shard_data_begin(path: &Path, file: &mut File) -> Result<u64> {
    let file_len = file.metadata().map_err(|e| io_error(path, e))?.len();
    let mut len_bytes = [0u8; 8];
    file.read_exact(&mut len_bytes)
        .map_err(|e| model_load(path, format!("read Safetensors header length: {e}")))?;
    let header_len = u64::from_le_bytes(len_bytes);
    if header_len > MAX_HEADER_BYTES as u64 {
        return Err(model_load(
            path,
            format!("Safetensors header length {header_len} exceeds limit {MAX_HEADER_BYTES}"),
        ));
    }
    let data_begin = header_len
        .checked_add(8)
        .ok_or_else(|| model_load(path, "Safetensors header length overflow".to_string()))?;
    if data_begin > file_len {
        return Err(model_load(
            path,
            "Safetensors header extends beyond file".to_string(),
        ));
    }
    Ok(data_begin)
}
