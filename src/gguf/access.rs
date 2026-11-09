// SPDX-License-Identifier: MIT OR Apache-2.0

//! Shared tensor-directory + payload access for owned and mmap GGUF layouts.

use std::collections::HashMap;

use super::layout::GgufLayout;
use super::tensor::Tensor;
use crate::error::Result;

/// Tensor directory and payload access shared by [`GgufLayout`] and
/// [`GgufLayoutMmap`](super::map::GgufLayoutMmap).
///
/// MoE helpers ([`crate::analysis::moe::list_experts_from`],
/// [`crate::analysis::moe::extract_expert_from`]) and other analysis code
/// use this trait so owned and mmap-backed checkpoints cannot drift.
pub trait GgufTensorAccess {
    /// Source path label (filesystem path or synthetic label).
    fn gguf_path(&self) -> &str;

    /// Parsed tensor directory (`name` -> [`Tensor`] metadata).
    fn gguf_tensors(&self) -> &HashMap<String, Tensor>;

    /// Borrowed raw payload bytes for `tensor`.
    fn tensor_bytes<'a>(&'a self, tensor: &Tensor) -> Result<&'a [u8]>;
}

impl GgufTensorAccess for GgufLayout {
    fn gguf_path(&self) -> &str {
        &self.path
    }

    fn gguf_tensors(&self) -> &HashMap<String, Tensor> {
        &self.tensors
    }

    fn tensor_bytes<'a>(&'a self, tensor: &Tensor) -> Result<&'a [u8]> {
        GgufLayout::tensor_bytes(self, tensor)
    }
}

#[cfg(feature = "mmap")]
impl GgufTensorAccess for super::map::GgufLayoutMmap {
    fn gguf_path(&self) -> &str {
        &self.path
    }

    fn gguf_tensors(&self) -> &HashMap<String, Tensor> {
        &self.tensors
    }

    fn tensor_bytes<'a>(&'a self, tensor: &Tensor) -> Result<&'a [u8]> {
        super::map::GgufLayoutMmap::tensor_bytes(self, tensor)
    }
}
