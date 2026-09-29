// SPDX-License-Identifier: MIT OR Apache-2.0
//! Runtime format selection: [`open_checkpoint`] and [`AnyCheckpoint`].

use std::borrow::Cow;
use std::fs::{self, File};
use std::io::Read;
use std::path::Path;

use super::{Checkpoint, CheckpointMetadata, CheckpointSource, GgufBackend, TensorInfo};
use crate::error::{ParserError, Result};

/// A checkpoint whose backend was chosen at runtime.
///
/// Implements [`Checkpoint`] by delegation. The `as_*` accessors return
/// the format-specific handle when consumers need format-only features.
#[non_exhaustive]
#[derive(Debug)]
pub enum AnyCheckpoint {
    /// Owned GGUF layout.
    Gguf(GgufBackend),
    /// Memory-mapped GGUF layout.
    #[cfg(feature = "mmap")]
    GgufMmap(super::GgufMmapBackend),
    /// Safetensors with bounded owned reads.
    #[cfg(feature = "safetensors")]
    Safetensors(super::SafetensorsBackend),
    /// Memory-mapped Safetensors shards.
    #[cfg(all(feature = "safetensors", feature = "mmap"))]
    SafetensorsMmap(super::SafetensorsMmapBackend),
}

// Accessors below use `_ => None`, which is unreachable when only the
// default GGUF variant is compiled in.
#[allow(unreachable_patterns)]
impl AnyCheckpoint {
    /// The active backend as a trait object.
    pub fn as_dyn(&self) -> &dyn Checkpoint {
        match self {
            Self::Gguf(backend) => backend,
            #[cfg(feature = "mmap")]
            Self::GgufMmap(backend) => backend,
            #[cfg(feature = "safetensors")]
            Self::Safetensors(backend) => backend,
            #[cfg(all(feature = "safetensors", feature = "mmap"))]
            Self::SafetensorsMmap(backend) => backend,
        }
    }

    /// Owned GGUF layout, for [`crate::list_experts`] / [`crate::extract_expert`].
    pub fn as_gguf(&self) -> Option<&crate::GgufLayout> {
        match self {
            Self::Gguf(backend) => Some(backend.layout()),
            _ => None,
        }
    }

    /// Memory-mapped GGUF layout.
    #[cfg(feature = "mmap")]
    pub fn as_gguf_mmap(&self) -> Option<&crate::GgufLayoutMmap> {
        match self {
            Self::GgufMmap(backend) => Some(backend.layout()),
            _ => None,
        }
    }

    /// Safetensors handle (manifest, MoE candidates, record lookup).
    #[cfg(feature = "safetensors")]
    pub fn as_safetensors(&self) -> Option<&crate::safetensors::SafetensorsCheckpoint> {
        match self {
            Self::Safetensors(backend) => Some(backend.checkpoint()),
            _ => None,
        }
    }

    /// Memory-mapped Safetensors handle.
    #[cfg(all(feature = "safetensors", feature = "mmap"))]
    pub fn as_safetensors_mmap(&self) -> Option<&crate::safetensors::SafetensorsCheckpointMmap> {
        match self {
            Self::SafetensorsMmap(backend) => Some(backend.checkpoint()),
            _ => None,
        }
    }
}

impl Checkpoint for AnyCheckpoint {
    fn source(&self) -> &CheckpointSource {
        self.as_dyn().source()
    }

    fn tensors(&self) -> &[TensorInfo] {
        self.as_dyn().tensors()
    }

    fn tensor(&self, name: &str) -> Option<&TensorInfo> {
        self.as_dyn().tensor(name)
    }

    fn metadata(&self) -> &CheckpointMetadata {
        self.as_dyn().metadata()
    }

    fn tensor_bytes(&self, name: &str) -> Result<Cow<'_, [u8]>> {
        self.as_dyn().tensor_bytes(name)
    }
}

/// Open any supported checkpoint, reading GGUF into memory and serving
/// Safetensors with bounded per-tensor reads.
///
/// Detection, in order:
/// 1. a directory is a Safetensors shard directory;
/// 2. a file starting with the `GGUF` magic is GGUF;
/// 3. a `*.safetensors` file or `*.safetensors.index.json` is Safetensors;
/// 4. a `*.gguf` file without the magic is handed to the GGUF parser,
///    which reports the bad header;
/// 5. anything else is [`ParserError::UnsupportedFormat`].
///
/// Safetensors input without the `safetensors` feature returns
/// [`ParserError::FeatureDisabled`]; no other backend is substituted.
pub fn open_checkpoint(path: impl AsRef<Path>) -> Result<AnyCheckpoint> {
    let path = path.as_ref();
    match detect(path)? {
        Detected::Gguf => Ok(AnyCheckpoint::Gguf(GgufBackend::open(path)?)),
        Detected::Safetensors => open_safetensors(path),
    }
}

/// Like [`open_checkpoint`], but memory-maps GGUF files and Safetensors
/// shards so payloads are borrowed from the mappings.
///
/// Callers must not truncate the files while the checkpoint is alive.
#[cfg(feature = "mmap")]
pub fn open_checkpoint_mmap(path: impl AsRef<Path>) -> Result<AnyCheckpoint> {
    let path = path.as_ref();
    match detect(path)? {
        Detected::Gguf => Ok(AnyCheckpoint::GgufMmap(super::GgufMmapBackend::open(path)?)),
        Detected::Safetensors => open_safetensors_mmap(path),
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Detected {
    Gguf,
    Safetensors,
}

const GGUF_MAGIC: &[u8; 4] = b"GGUF";

fn detect(path: &Path) -> Result<Detected> {
    let io_err = |source| ParserError::Io {
        path: path.display().to_string(),
        source,
    };
    let metadata = fs::metadata(path).map_err(io_err)?;
    if metadata.is_dir() {
        return Ok(Detected::Safetensors);
    }

    let mut magic = [0u8; 4];
    let mut file = File::open(path).map_err(io_err)?;
    let read = read_prefix(&mut file, &mut magic).map_err(io_err)?;
    if read == magic.len() && &magic == GGUF_MAGIC {
        return Ok(Detected::Gguf);
    }

    let name = path
        .file_name()
        .map(|n| n.to_string_lossy().to_ascii_lowercase())
        .unwrap_or_default();
    if name.ends_with(".safetensors") || name.ends_with(".safetensors.index.json") {
        return Ok(Detected::Safetensors);
    }
    if name.ends_with(".gguf") {
        return Ok(Detected::Gguf);
    }
    Err(ParserError::UnsupportedFormat {
        path: path.display().to_string(),
        reason: "not a GGUF file (no GGUF magic), .safetensors file, \
                 .safetensors.index.json index, or Safetensors directory"
            .to_string(),
    })
}

/// Read up to `buf.len()` bytes, tolerating short files.
fn read_prefix(file: &mut File, buf: &mut [u8]) -> std::io::Result<usize> {
    let mut filled = 0;
    while filled < buf.len() {
        match file.read(&mut buf[filled..])? {
            0 => break,
            n => filled += n,
        }
    }
    Ok(filled)
}

#[cfg(feature = "safetensors")]
fn open_safetensors(path: &Path) -> Result<AnyCheckpoint> {
    Ok(AnyCheckpoint::Safetensors(super::SafetensorsBackend::open(
        path,
    )?))
}

#[cfg(not(feature = "safetensors"))]
fn open_safetensors(path: &Path) -> Result<AnyCheckpoint> {
    Err(safetensors_disabled(path))
}

#[cfg(all(feature = "mmap", feature = "safetensors"))]
fn open_safetensors_mmap(path: &Path) -> Result<AnyCheckpoint> {
    Ok(AnyCheckpoint::SafetensorsMmap(
        super::SafetensorsMmapBackend::open(path)?,
    ))
}

#[cfg(all(feature = "mmap", not(feature = "safetensors")))]
fn open_safetensors_mmap(path: &Path) -> Result<AnyCheckpoint> {
    Err(safetensors_disabled(path))
}

#[cfg(not(feature = "safetensors"))]
fn safetensors_disabled(path: &Path) -> ParserError {
    ParserError::FeatureDisabled {
        path: path.display().to_string(),
        feature: "safetensors",
    }
}
