// SPDX-License-Identifier: MIT OR Apache-2.0

//! Mixture-of-Experts weight extraction (GGUF-specific).
//!
//! Locates MoE expert tensors inside a parsed [`GgufLayout`](crate::gguf::GgufLayout)
//! and returns raw per-expert byte buffers. Performs no neural-network
//! math — callers receive `Vec<u8>` + shape/dtype metadata and are
//! responsible for any downstream compute.
//!
//! # Scope
//!
//! This is a model-analysis specialization, not part of the core
//! checkpoint contract: [`extract_expert`] and [`list_experts`] take a
//! concrete `&GgufLayout`, and [`RawTensor`] carries the GGUF-specific
//! [`DType`](crate::gguf::DType) / `ggml_type` fields of its source
//! tensor. It does not accept [`Checkpoint`](crate::checkpoint::Checkpoint)
//! — a format-independent MoE/topology layer is planned v0.4 work.
//!
//! Safetensors has its own candidate-discovery surface (the
//! feature-gated `safetensors` module); the two paths are not yet one
//! generic MoE API.
//!
//! # Compatibility
//!
//! This is the canonical MoE namespace as of v0.3. The legacy
//! [`crate::moe`] module and the crate-root re-exports
//! ([`crate::extract_expert`], [`crate::list_experts`],
//! [`crate::MoeExpertWeights`], [`crate::RawTensor`]) re-export this
//! same implementation and continue to compile.

mod expert;
mod extract;

pub use expert::{MoeExpertWeights, RawTensor};
pub use extract::{extract_expert, list_experts};
