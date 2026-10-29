// SPDX-License-Identifier: MIT OR Apache-2.0
//! Format-independent Transformer tensor-role and topology discovery.
//!
//! Dense and Mixture-of-Experts checkpoints are described through one
//! vocabulary: every tensor gets a [`TensorRole`] (embedding, attention
//! Q/K/V/O, MLP gate/up/down, norms, output head, MoE router, …) and a
//! [`RoleScope`] (global, layer, routed expert, stacked experts, shared
//! expert). Layers are then assembled into a [`TransformerTopology`] that
//! downstream ANN, SNN, or hybrid research can map into its own compute
//! domain.
//!
//! Discovery runs on the normalized [`checkpoint`](crate::checkpoint)
//! contract only ([`TensorInfo`] names, shapes, dtypes and
//! [`CheckpointMetadata`]), so GGUF and Safetensors inventories produce
//! comparable topologies. Nothing is executed, dequantized, or read from
//! payload bytes.
//!
//! # Honesty rules
//!
//! - A tensor that matches no known convention is [`TensorRole::Unknown`]
//!   with [`RoleConfidence::Unknown`]; it is never guessed into a role.
//! - A name whose meaning depends on the model family (a bare `mlp.gate`
//!   is a router in Qwen-MoE and would be a gate projection elsewhere)
//!   stays [`RoleConfidence::Ambiguous`] with the candidates listed in
//!   [`RoleAssignment::alternatives`] unless the checkpoint itself
//!   supplies evidence (other expert tensors, a router-like shape).
//! - Convention-dependent mappings (`w1`/`w3`/`w2` = gate/up/down) are
//!   [`RoleConfidence::Heuristic`] and carry their reason in
//!   [`RoleAssignment::evidence`].
//! - `*_hint` fields on [`TopologySummary`] are read from shapes and
//!   metadata, not validated.
//!
//! # Component view
//!
//! [`TransformerTopology::components`] groups classified tensors into
//! [`Component`] records (attention block, MLP block, MoE router, routed or
//! stacked experts, shared expert, norms, global embeddings/head/norm).
//! Each component carries member roles, shapes, dtypes, a parameter count,
//! scope-derived `conditional` vs dense flags, router/gate markers, and
//! the weakest member [`RoleConfidence`]. Unknown and ambiguous tensors
//! remain in [`TransformerComponents::unclassified`]; they are never merged
//! into a component by guesswork. [`TransformerComponents::to_json`] emits
//! a stable, diff-friendly snapshot.
//!
//! ## Downstream placement (example only)
//!
//! A crate such as **hybrid-fusion** owns ANN/SNN/hybrid *placement*
//! policy. engram-parser stays descriptive: it never labels components
//! ANN, SNN, or hybrid. One possible consumer mapping (not a rule):
//!
//! ```ignore
//! use engram_parser::analysis::transformer::{ComponentKind, TransformerTopology};
//!
//! fn example_placement_hint(topo: &TransformerTopology) {
//!     for layer in &topo.components().layers {
//!         for component in &layer.components {
//!             let _policy = match component.kind {
//!                 ComponentKind::AttentionBlock | ComponentKind::MlpBlock => "often ANN in v0.6 experiments",
//!                 ComponentKind::MoeRouter | ComponentKind::RoutedExpert
//!                 | ComponentKind::StackedExperts | ComponentKind::SharedExpert => "conditional; planner chooses backend per expert",
//!                 ComponentKind::Norms => "fuse with adjacent block in planner",
//!                 _ => "global / embeddings — planner-specific",
//!             };
//!         }
//!     }
//! }
//! ```
//!
//! Real policies live in hybrid-fusion v0.6 ([RM-1809](https://linear.app/rpd-34/issue/RM-1809));
//! this crate only supplies the facts above.
//!
//! # Example
//!
//! ```no_run
//! use engram_parser::analysis::transformer::discover_topology;
//! use engram_parser::open_checkpoint;
//!
//! let ckpt = open_checkpoint("model.safetensors")?;
//! let topo = discover_topology(&ckpt);
//! println!("{topo}");
//! for layer in &topo.layers {
//!     println!("layer {}: {} / {}", layer.index, layer.attention, layer.feed_forward);
//! }
//! let json = topo.to_json();
//! # let _ = json;
//! # Ok::<(), engram_parser::ParserError>(())
//! ```
//!
//! # Determinism and bounds
//!
//! Output depends only on the set of `(name, shape, dtype)` tuples and
//! the metadata; input order does not matter. `Display` and
//! [`TransformerTopology::to_json`] are stable across runs.
//!
//! Layers are grouped by `(stack, index)`, where the stack is the name
//! prefix up to the layer marker (`blk`, `model.layers`, `visual.blocks`),
//! so multi-tower checkpoints never merge unrelated layers. Work and
//! memory are bounded by the inventory size plus [`MAX_MISSING_LAYERS`];
//! layer indices and declared counts from untrusted input never size an
//! allocation.

mod components;
mod json;
mod role;
mod topology;

pub use components::{
    Component, ComponentKind, ComponentMember, LayerComponents, TransformerComponents,
    UnclassifiedTensor,
};
pub use role::{
    ClassifyContext, ParamKind, RoleAssignment, RoleConfidence, RoleScope, TensorRole,
    classify_tensor,
};
pub use topology::{
    AttentionKind, FeedForwardKind, LayerTopology, MAX_MISSING_LAYERS, TopologySummary,
    TransformerTopology,
};

use crate::checkpoint::{Checkpoint, CheckpointMetadata, TensorInfo};

/// Discover the Transformer topology of an opened checkpoint.
///
/// Works for any [`Checkpoint`] backend (GGUF, Safetensors, mmap or
/// owned). Reads only the tensor inventory and scalar metadata.
pub fn discover_topology(ckpt: &dyn Checkpoint) -> TransformerTopology {
    discover_topology_from_parts(ckpt.tensors(), ckpt.metadata())
}

/// Discover the topology from a bare inventory and metadata.
///
/// Useful for mock inventories in downstream tests. Tensors may arrive in
/// any order; duplicate names keep the first occurrence after sorting.
pub fn discover_topology_from_parts(
    tensors: &[TensorInfo],
    metadata: &CheckpointMetadata,
) -> TransformerTopology {
    topology::assemble(tensors, metadata)
}
