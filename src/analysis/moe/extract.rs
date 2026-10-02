// SPDX-License-Identifier: MIT OR Apache-2.0

//! Per-expert weight extraction from a parsed [`GgufLayout`].
//!
//! Two on-disk conventions are supported:
//!
//! 1. **Stacked** — a single tensor `blk.{b}.ffn_{role}_exps.weight`
//!    with shape `[n_experts, inner, outer]` (GGML stores dimensions
//!    innermost-first, so the highest-rank axis is the expert axis).
//!    We compute the per-expert byte stride and slice out one chunk.
//! 2. **Per-expert** — one tensor per expert, e.g.
//!    `blk.{b}.ffn_{role}.{e}.weight`. Whole-tensor payload is cloned.
//!
//! No arithmetic is performed — only byte-level slicing plus a `Vec`
//! clone so the returned [`MoeExpertWeights`] is self-owning.

use std::collections::BTreeSet;

use super::expert::{MoeExpertWeights, RawTensor};
use crate::error::{ParserError, Result};
use crate::gguf::{GgufLayout, Tensor};

/// Enumerate every `(block, expert)` pair present in the checkpoint.
///
/// A pair is reported if *any* of the three projections (`gate`, `up`,
/// `down`) is discoverable for it under either the stacked or the
/// per-expert naming scheme.
pub fn list_experts(layout: &GgufLayout) -> Vec<(usize, usize)> {
    let mut pairs: BTreeSet<(usize, usize)> = BTreeSet::new();

    for tensor in layout.tensors.values() {
        record_expert_pairs(tensor, &mut pairs);
    }

    pairs.into_iter().collect()
}

fn record_expert_pairs(tensor: &Tensor, pairs: &mut BTreeSet<(usize, usize)>) {
    if let Some((block, _role)) = parse_stacked_name(&tensor.name) {
        if let Some(n_experts) = stacked_expert_count(tensor) {
            for expert in 0..n_experts {
                pairs.insert((block, expert));
            }
        }
        return;
    }

    if let Some((block, _role, expert)) = parse_per_expert_name(&tensor.name) {
        pairs.insert((block, expert));
    }
}

/// Extract the raw weights for a single `(block, expert)` pair.
///
/// Returns [`MoeExpertWeights`] whose `gate / up / down` fields are
/// `None` when that projection was not found in the checkpoint. If
/// *none* of the three projections can be located, returns
/// [`ParserError::MissingTensor`].
pub fn extract_expert(
    layout: &GgufLayout,
    block: usize,
    expert: usize,
) -> Result<MoeExpertWeights> {
    let gate = extract_role(layout, block, expert, "gate")?;
    let up = extract_role(layout, block, expert, "up")?;
    let down = extract_role(layout, block, expert, "down")?;

    if gate.is_none() && up.is_none() && down.is_none() {
        return Err(ParserError::MissingTensor {
            name: format!("blk.{block}.ffn_(gate|up|down)[_exps].weight"),
            path: layout.path.clone(),
        });
    }

    Ok(MoeExpertWeights {
        block,
        expert,
        gate,
        up,
        down,
    })
}

fn extract_role(
    layout: &GgufLayout,
    block: usize,
    expert: usize,
    role: &str,
) -> Result<Option<RawTensor>> {
    // Prefer the stacked convention first.
    let stacked_name = format!("blk.{block}.ffn_{role}_exps.weight");
    if let Some(tensor) = layout.tensors.get(&stacked_name) {
        return Ok(Some(slice_stacked_expert(layout, block, tensor, expert)?));
    }

    // Fall back to per-expert tensors. GGUF files in the wild use
    // either `blk.B.ffn_ROLE.E.weight` or `blk.B.ffn_ROLE_E.weight`;
    // check both.
    for candidate in [
        format!("blk.{block}.ffn_{role}.{expert}.weight"),
        format!("blk.{block}.ffn_{role}_{expert}.weight"),
    ] {
        if let Some(tensor) = layout.tensors.get(&candidate) {
            let bytes = layout.tensor_bytes(tensor)?.to_vec();
            return Ok(Some(RawTensor {
                source_name: tensor.name.clone(),
                dims: tensor.dims.clone(),
                dtype: tensor.dtype,
                ggml_type: tensor.ggml_type,
                bytes,
                stacked_slice: false,
            }));
        }
    }

    Ok(None)
}

/// Slice the `expert`-th chunk out of a stacked `[inner, outer, n_experts]`
/// tensor.
///
/// GGML stores dims innermost-first, so the expert axis is the last
/// dimension in the `dims` vector. The per-expert chunk consists of
/// the first `n_elements / n_experts` elements per expert, laid out
/// contiguously in the tensor buffer.
fn slice_stacked_expert(
    layout: &GgufLayout,
    block: usize,
    tensor: &Tensor,
    expert: usize,
) -> Result<RawTensor> {
    let n_experts = validate_stacked_expert_count(layout, block, tensor, expert)?;
    let bytes = layout.tensor_bytes(tensor)?;
    let (start, end) = stacked_slice_range(layout, tensor, expert, n_experts, bytes.len())?;
    let per_expert_dims: Vec<usize> = tensor.dims[..tensor.dims.len() - 1].to_vec();

    Ok(RawTensor {
        source_name: tensor.name.clone(),
        dims: per_expert_dims,
        dtype: tensor.dtype,
        ggml_type: tensor.ggml_type,
        bytes: bytes[start..end].to_vec(),
        stacked_slice: true,
    })
}

fn validate_stacked_expert_count(
    layout: &GgufLayout,
    block: usize,
    tensor: &Tensor,
    expert: usize,
) -> Result<usize> {
    let n_experts = stacked_expert_count(tensor).ok_or_else(|| ParserError::InvalidLayout {
        path: layout.path.clone(),
        reason: format!(
            "stacked tensor '{}' has rank {} (dims={:?}); expected rank >= 2",
            tensor.name,
            tensor.dims.len(),
            tensor.dims
        ),
    })?;

    if n_experts == 0 {
        return Err(ParserError::InvalidLayout {
            path: layout.path.clone(),
            reason: format!("stacked tensor '{}' has zero experts", tensor.name),
        });
    }

    if expert >= n_experts {
        return Err(ParserError::ExpertOutOfRange {
            block,
            expert,
            available: n_experts,
        });
    }

    Ok(n_experts)
}

fn stacked_slice_range(
    layout: &GgufLayout,
    tensor: &Tensor,
    expert: usize,
    n_experts: usize,
    buffer_len: usize,
) -> Result<(usize, usize)> {
    if !tensor.byte_len.is_multiple_of(n_experts) {
        return Err(ParserError::InvalidLayout {
            path: layout.path.clone(),
            reason: format!(
                "stacked tensor '{}' byte length {} is not divisible by n_experts={n_experts}",
                tensor.name, tensor.byte_len,
            ),
        });
    }

    let stride = tensor.byte_len / n_experts;
    let start = expert
        .checked_mul(stride)
        .ok_or_else(|| ParserError::InvalidLayout {
            path: layout.path.clone(),
            reason: format!("stacked stride overflow for tensor '{}'", tensor.name),
        })?;
    let end = start
        .checked_add(stride)
        .ok_or_else(|| ParserError::InvalidLayout {
            path: layout.path.clone(),
            reason: format!("stacked end overflow for tensor '{}'", tensor.name),
        })?;
    if end > buffer_len {
        return Err(ParserError::InvalidLayout {
            path: layout.path.clone(),
            reason: format!(
                "stacked slice range [{start}..{end}] for tensor '{}' exceeds buffer len {buffer_len}",
                tensor.name,
            ),
        });
    }

    Ok((start, end))
}

/// For a stacked MoE tensor, the expert axis is the outermost (last)
/// dimension in GGML's innermost-first dim order. A tensor with a
/// single dim cannot be interpreted as stacked.
fn stacked_expert_count(tensor: &Tensor) -> Option<usize> {
    if tensor.dims.len() < 2 {
        return None;
    }
    tensor.dims.last().copied()
}

/// Parse a stacked MoE tensor name into `(block, role)`. Returns
/// `None` if the name does not match the `blk.{B}.ffn_{role}_exps.weight`
/// convention.
const STACKED_NAME_SUFFIXES: &[(&str, &str)] = &[
    ("ffn_gate_exps.weight", "gate"),
    ("ffn_up_exps.weight", "up"),
    ("ffn_down_exps.weight", "down"),
];

fn parse_stacked_name(name: &str) -> Option<(usize, &'static str)> {
    let rest = name.strip_prefix("blk.")?;
    let (block_str, tail) = rest.split_once('.')?;
    let block: usize = block_str.parse().ok()?;

    for (suffix, role) in STACKED_NAME_SUFFIXES {
        if tail == *suffix {
            return Some((block, role));
        }
    }
    None
}

const PER_EXPERT_NAME_PATTERNS: &[(&str, &str)] = &[
    ("ffn_gate.", "gate"),
    ("ffn_up.", "up"),
    ("ffn_down.", "down"),
    ("ffn_gate_", "gate"),
    ("ffn_up_", "up"),
    ("ffn_down_", "down"),
];

/// Parse a per-expert MoE tensor name into `(block, role, expert)`.
/// Matches both `blk.B.ffn_ROLE.E.weight` and `blk.B.ffn_ROLE_E.weight`
/// forms.
fn parse_per_expert_name(name: &str) -> Option<(usize, &'static str, usize)> {
    let rest = name.strip_prefix("blk.")?;
    let (block_str, tail) = rest.split_once('.')?;
    let block: usize = block_str.parse().ok()?;

    for (prefix, role) in PER_EXPERT_NAME_PATTERNS {
        if let Some(expert) = parse_expert_index(tail, prefix) {
            return Some((block, role, expert));
        }
    }
    None
}

fn parse_expert_index(tail: &str, prefix: &str) -> Option<usize> {
    let sub = tail.strip_prefix(prefix)?;
    sub.strip_suffix(".weight")?.parse().ok()
}
