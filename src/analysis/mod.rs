// SPDX-License-Identifier: MIT OR Apache-2.0

//! Model-analysis specializations built on top of format parsing.
//!
//! Modules here interpret checkpoint contents for a specific model
//! family or architecture pattern. They are consumers of the
//! format-specific handles (`gguf`, `safetensors`) and the shared
//! [`checkpoint`](crate::checkpoint) contract — not part of the core
//! parsing substrate.
//!
//! Today this namespace holds a single specialization:
//!
//! - [`moe`]: Mixture-of-Experts expert enumeration and per-expert
//!   raw-weight extraction. GGUF-specific: it operates on
//!   [`GgufLayout`](crate::gguf::GgufLayout) and returns
//!   GGUF-typed [`RawTensor`](moe::RawTensor) slices.
//!
//! Format-independent MoE/topology discovery (model roles, transformer
//! structure) is deliberately *not* here yet — that is v0.4 work.

pub mod moe;
