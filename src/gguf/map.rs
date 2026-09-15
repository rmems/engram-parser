// SPDX-License-Identifier: MIT OR Apache-2.0

//! Memory-mapped GGUF checkpoint access (optional `mmap` feature).
//!
//! Maps the file with read-only [`memmap2`] and reuses the same header /
//! tensor-directory parser as [`super::parse_bytes`]. CUDA host-register is
//! intentionally out of scope (that belongs in myelin-accelerator). Callers
//! can still obtain **page-aligned** tensor byte ranges so a later GPU layer
//! can pin pages without this crate depending on CUDA.

use std::collections::HashMap;
use std::fs::File;
use std::path::Path;

use memmap2::{Mmap, MmapOptions};

use super::layout::{GgufLayout, GgufMetadata, parse_layout, tensor_payload_bytes};
use super::tensor::Tensor;
use crate::error::{ParserError, Result};

/// GGUF layout backed by a read-only file mapping instead of an owned `Vec<u8>`.
///
/// Header, KV metadata, and the tensor directory are identical to
/// [`GgufLayout`]. Tensor payloads are borrowed from the mapping.
pub struct GgufLayoutMmap {
    /// Path the checkpoint was mapped from (kept for error messages).
    pub path: String,
    /// Parsed KV metadata.
    pub metadata: GgufMetadata,
    /// Name -> tensor directory entry.
    pub tensors: HashMap<String, Tensor>,
    /// Byte alignment specified by the file (default 32).
    pub alignment: usize,
    /// Absolute byte offset within the mapping where tensor payloads begin.
    pub tensor_data_offset: usize,
    mmap: Mmap,
}

impl std::fmt::Debug for GgufLayoutMmap {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GgufLayoutMmap")
            .field("path", &self.path)
            .field("metadata", &self.metadata)
            .field("tensors", &self.tensors)
            .field("alignment", &self.alignment)
            .field("tensor_data_offset", &self.tensor_data_offset)
            .field("mmap_len", &self.mmap.len())
            .finish_non_exhaustive()
    }
}

/// Page-aligned view covering a tensor payload inside a mapping.
///
/// `pages` starts at an OS-page boundary and covers the tensor; it may
/// include bytes before/after the payload (and may be truncated at EOF
/// when the file is smaller than a page). [`Self::tensor_bytes`] returns
/// the exact payload.
#[derive(Debug, Clone, Copy)]
pub struct PageAlignedTensorBytes<'a> {
    /// Page-aligned (or file-start) range covering the tensor.
    pub pages: &'a [u8],
    /// Offset of the exact tensor payload within [`Self::pages`].
    pub tensor_offset: usize,
    /// Exact tensor payload length in bytes.
    pub tensor_len: usize,
    /// Page size used to compute the range.
    pub page_size: usize,
}

impl<'a> PageAlignedTensorBytes<'a> {
    /// Exact tensor payload, equivalent to [`GgufLayoutMmap::tensor_bytes`].
    pub fn tensor_bytes(self) -> &'a [u8] {
        &self.pages[self.tensor_offset..self.tensor_offset + self.tensor_len]
    }
}

/// Memory-map a `.gguf` file and parse its header, KV metadata, and tensor
/// directory without copying the file into a `Vec<u8>`.
///
/// Requires the `mmap` cargo feature (`memmap2`). The default
/// [`super::load_gguf`] path is unchanged.
pub fn load_gguf_mmap<P: AsRef<Path>>(path: P) -> Result<GgufLayoutMmap> {
    let path_ref = path.as_ref();
    let path_str = path_ref.display().to_string();
    let file = File::open(path_ref).map_err(|e| ParserError::Io {
        path: path_str.clone(),
        source: e,
    })?;
    // SAFETY: `file` is a readable regular-file descriptor. `MmapOptions::map`
    // creates a read-only mapping. Callers must not truncate the file for the
    // lifetime of the returned [`GgufLayoutMmap`] (standard mmap invariant).
    // This crate never host-registers the mapping with CUDA.
    let mmap = unsafe { MmapOptions::new().map(&file) }.map_err(|e| ParserError::Io {
        path: path_str.clone(),
        source: e,
    })?;
    let (metadata, tensors, alignment, tensor_data_offset) = parse_layout(&mmap, &path_str)?;
    Ok(GgufLayoutMmap {
        path: path_str,
        metadata,
        tensors,
        alignment,
        tensor_data_offset,
        mmap,
    })
}

impl GgufLayoutMmap {
    /// Length of the mapped file in bytes.
    pub fn len(&self) -> usize {
        self.mmap.len()
    }

    /// Whether the mapping is empty.
    pub fn is_empty(&self) -> bool {
        self.mmap.is_empty()
    }

    /// Borrowed view of the entire mapped file.
    pub fn as_bytes(&self) -> &[u8] {
        &self.mmap
    }

    /// Return a borrowed slice of the raw tensor payload bytes.
    pub fn tensor_bytes<'a>(&'a self, tensor: &Tensor) -> Result<&'a [u8]> {
        tensor_payload_bytes(&self.mmap, &self.path, tensor)
    }

    /// Lookup a tensor by exact name.
    pub fn tensor(&self, name: &str) -> Result<&Tensor> {
        self.tensors
            .get(name)
            .ok_or_else(|| ParserError::MissingTensor {
                name: name.to_owned(),
                path: self.path.clone(),
            })
    }

    /// Page-aligned byte range covering `tensor`, suitable for a later
    /// CUDA host-register in another crate. Does not register anything here.
    pub fn tensor_page_aligned_bytes<'a>(
        &'a self,
        tensor: &Tensor,
    ) -> Result<PageAlignedTensorBytes<'a>> {
        let payload = tensor_payload_bytes(&self.mmap, &self.path, tensor)?;
        let start = tensor.absolute_offset;
        let end = start
            .checked_add(tensor.byte_len)
            .ok_or_else(|| ParserError::InvalidLayout {
                path: self.path.clone(),
                reason: format!("tensor '{}' page-range overflow", tensor.name),
            })?;
        let page = os_page_size();
        if page == 0 {
            return Err(ParserError::InvalidLayout {
                path: self.path.clone(),
                reason: "invalid OS page size: 0".to_string(),
            });
        }
        let aligned_start = (start / page) * page;
        let aligned_end = align_up_saturating(end, page, self.mmap.len());
        if aligned_end > self.mmap.len() || aligned_start > self.mmap.len() {
            return Err(ParserError::InvalidLayout {
                path: self.path.clone(),
                reason: format!(
                    "tensor '{}' page-aligned range exceeds mapping",
                    tensor.name
                ),
            });
        }
        let pages = &self.mmap[aligned_start..aligned_end];
        let tensor_offset = start - aligned_start;
        debug_assert_eq!(
            &pages[tensor_offset..tensor_offset + tensor.byte_len],
            payload
        );
        Ok(PageAlignedTensorBytes {
            pages,
            tensor_offset,
            tensor_len: tensor.byte_len,
            page_size: page,
        })
    }

    /// Compare directory metadata with an owned [`GgufLayout`] of the same file.
    pub fn directory_matches(&self, owned: &GgufLayout) -> bool {
        self.alignment == owned.alignment
            && self.tensor_data_offset == owned.tensor_data_offset
            && self.tensors.len() == owned.tensors.len()
            && self.metadata.architecture() == owned.metadata.architecture()
    }
}

/// OS page size used for [`GgufLayoutMmap::tensor_page_aligned_bytes`].
pub fn os_page_size() -> usize {
    match unix_page_size() {
        Some(n) if n > 0 => n,
        _ => 4096,
    }
}

#[cfg(unix)]
fn unix_page_size() -> Option<usize> {
    // POSIX `_SC_PAGESIZE` numeric ids (no `libc` crate).
    #[cfg(any(target_os = "linux", target_os = "android"))]
    const SC_PAGESIZE: i32 = 30;
    #[cfg(any(target_os = "macos", target_os = "ios"))]
    const SC_PAGESIZE: i32 = 29;
    #[cfg(target_os = "freebsd")]
    const SC_PAGESIZE: i32 = 47;
    #[cfg(any(target_os = "netbsd", target_os = "openbsd"))]
    const SC_PAGESIZE: i32 = 28;
    #[cfg(not(any(
        target_os = "linux",
        target_os = "android",
        target_os = "macos",
        target_os = "ios",
        target_os = "freebsd",
        target_os = "netbsd",
        target_os = "openbsd"
    )))]
    const SC_PAGESIZE: i32 = 29;

    unsafe extern "C" {
        fn sysconf(name: i32) -> i64;
    }

    // SAFETY: `sysconf(_SC_PAGESIZE)` is a pure query with no preconditions.
    let n = unsafe { sysconf(SC_PAGESIZE) };
    (n > 0).then_some(n as usize)
}

#[cfg(not(unix))]
fn unix_page_size() -> Option<usize> {
    None
}

fn align_up_saturating(value: usize, alignment: usize, limit: usize) -> usize {
    if alignment <= 1 {
        return value.min(limit);
    }
    value
        .div_ceil(alignment)
        .saturating_mul(alignment)
        .min(limit)
}
