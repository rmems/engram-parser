// SPDX-License-Identifier: MIT OR Apache-2.0

//! Model-analysis specializations built on top of format parsing.
//!
//! Modules here interpret checkpoint contents for a specific model
//! family or architecture pattern. They are consumers of the
//! format-specific handles (`gguf`, `safetensors`) and the shared
//! [`checkpoint`](crate::checkpoint) contract — not part of the core
//! parsing substrate.
//!
//! - [`transformer`]: format-independent Transformer tensor-role and
//!   topology discovery over the [`Checkpoint`](crate::checkpoint::Checkpoint)
//!   contract. Dense attention/MLP/norm/embedding roles and MoE
//!   router/expert roles share one topology layer; unknown and ambiguous
//!   tensors stay representable. [`transformer::TransformerTopology::components`]
//!   groups classified tensors into a facts-only component view (no ANN/SNN
//!   placement labels). Descriptive only, no execution.
//! - [`moe`]: Mixture-of-Experts expert enumeration and per-expert
//!   raw-weight extraction. GGUF-specific: it operates on
//!   [`GgufLayout`](crate::gguf::GgufLayout) and returns
//!   GGUF-typed [`RawTensor`](moe::RawTensor) slices.

pub mod moe;
pub mod transformer;
