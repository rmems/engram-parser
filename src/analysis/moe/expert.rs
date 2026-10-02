// SPDX-License-Identifier: MIT OR Apache-2.0

//! Per-expert raw weight bundle.
//!
//! Each expert has up to three projections (`gate`, `up`, `down`),
//! stored as opaque byte buffers plus shape/dtype metadata. The parser
//! never assembles a functional neural layer — callers receive raw
//! bytes and are responsible for any downstream compute.

use crate::gguf::DType;

/// Owned raw tensor buffer: bytes + shape + dtype, no references into
/// the source [`GgufLayout`](crate::gguf::GgufLayout).
///
/// This is an *analysis result type*, not a tensor abstraction:
/// expert extraction may slice a stacked GGUF tensor, so the result
/// holds expert-local `dims` plus owned `bytes`. It does not replace
/// [`TensorInfo`](crate::checkpoint::TensorInfo) /
/// [`Checkpoint::tensor_bytes`](crate::checkpoint::Checkpoint::tensor_bytes),
/// which remain the core checkpoint contract. Its `dtype`/`ggml_type`
/// fields make this extraction path GGUF-specific; any normalized
/// cross-format expert tensor type is planned v0.4 work.
#[derive(Debug, Clone)]
pub struct RawTensor {
    /// Source tensor name (e.g. `"blk.0.ffn_gate_exps.weight"`).
    pub source_name: String,
    /// Dimensions of this expert's slice (expert dimension stripped).
    pub dims: Vec<usize>,
    /// Dtype preserved from the GGUF directory.
    pub dtype: DType,
    /// Raw `ggml_type` code (useful for round-tripping unknown quants).
    pub ggml_type: u32,
    /// Raw byte payload — length equals the byte size of one expert's
    /// chunk within a stacked tensor, or the whole tensor if the
    /// checkpoint stores one tensor per expert.
    pub bytes: Vec<u8>,
    /// True if this buffer was sliced from a stacked `[n_experts, ...]`
    /// tensor; false if the checkpoint stored a dedicated per-expert
    /// tensor.
    pub stacked_slice: bool,
}

/// Raw weights for a single MoE expert.
///
/// Fields are `Option<_>` because GGUF checkpoints differ on which
/// expert projections they store. OLMoE exposes `gate / up / down`;
/// other MoE families may omit or rename a subset.
#[derive(Debug, Clone)]
pub struct MoeExpertWeights {
    /// Transformer block this expert belongs to.
    pub block: usize,
    /// Expert index within the block.
    pub expert: usize,
    /// `ffn_gate` projection, if present in the checkpoint.
    pub gate: Option<RawTensor>,
    /// `ffn_up` projection, if present.
    pub up: Option<RawTensor>,
    /// `ffn_down` projection, if present.
    pub down: Option<RawTensor>,
}

impl MoeExpertWeights {
    /// Convenience: `true` if all three projections were located.
    pub fn is_complete(&self) -> bool {
        self.gate.is_some() && self.up.is_some() && self.down.is_some()
    }
}
