// SPDX-License-Identifier: MIT OR Apache-2.0

//! Compatibility re-exports for the legacy `engram_parser::moe` path.
//!
//! The canonical namespace for MoE analysis moved to
//! [`crate::analysis::moe`] in v0.3. Everything here is a re-export of
//! that single implementation — no duplicated logic — so existing
//! consumers keep compiling unchanged. New code should import from
//! `engram_parser::analysis::moe`.

pub use crate::analysis::moe::{MoeExpertWeights, RawTensor, extract_expert, list_experts};
