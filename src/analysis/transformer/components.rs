// SPDX-License-Identifier: MIT OR Apache-2.0
//! Facts-only component view over a [`TransformerTopology`].
//!
//! Role-tagged tensors are grouped into named components (attention, MLP,
//! MoE router, experts, norms, global embeddings/head/norm). Each component
//! records shapes, dtypes, parameter counts, scope-derived density vs
//! conditional routing, router/gate flags, and the weakest member
//! confidence. [`RoleConfidence::Ambiguous`] and [`RoleConfidence::Unknown`]
//! tensors are never folded into a component; they stay in
//! [`TransformerComponents::unclassified`].

use std::collections::BTreeMap;
use std::fmt;

use super::json::JsonWriter;
use super::role::{ParamKind, RoleAssignment, RoleConfidence, RoleScope, TensorRole};
use super::topology::{LayerTopology, TransformerTopology};

/// Structural kind of a grouped component.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum ComponentKind {
    /// Token and/or position embedding tables.
    Embeddings,
    /// Vocabulary logits projection.
    OutputHead,
    /// Final normalization before the output head.
    FinalNorm,
    /// Rotary frequency table(s) at global scope.
    RotaryFrequencies,
    /// Attention projections and per-head norms in one layer.
    AttentionBlock,
    /// Layer normalization tensors (pre/post attention and MLP).
    Norms,
    /// Dense feed-forward block in one layer.
    MlpBlock,
    /// MoE router / expert-selection gate.
    MoeRouter,
    /// One routed expert's projections.
    RoutedExpert,
    /// All routed experts stored in stacked tensors.
    StackedExperts,
    /// Always-on shared expert block.
    SharedExpert,
}

impl ComponentKind {
    /// Stable snake_case name.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Embeddings => "embeddings",
            Self::OutputHead => "output_head",
            Self::FinalNorm => "final_norm",
            Self::RotaryFrequencies => "rotary_frequencies",
            Self::AttentionBlock => "attention_block",
            Self::Norms => "norms",
            Self::MlpBlock => "mlp_block",
            Self::MoeRouter => "moe_router",
            Self::RoutedExpert => "routed_expert",
            Self::StackedExperts => "stacked_experts",
            Self::SharedExpert => "shared_expert",
        }
    }
}

impl fmt::Display for ComponentKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.pad(self.as_str())
    }
}

/// One tensor belonging to a component.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ComponentMember {
    pub name: String,
    pub role: TensorRole,
    pub scope: RoleScope,
    pub param: ParamKind,
    pub confidence: RoleConfidence,
    pub dims: Vec<usize>,
    pub dtype: String,
}

/// A grouped component with aggregate facts.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Component {
    pub kind: ComponentKind,
    /// Layer stack when `layer` is set; `None` for global components.
    pub stack: Option<String>,
    /// Layer index when the component is layer-scoped.
    pub layer: Option<usize>,
    /// Expert index for [`ComponentKind::RoutedExpert`].
    pub expert_index: Option<usize>,
    pub members: Vec<ComponentMember>,
    /// Sum of element counts across member tensors; saturates at
    /// `usize::MAX` rather than overflowing.
    pub parameter_count: usize,
    /// `false` for dense/global/layer-scoped blocks and always-on shared
    /// experts; `true` for routers and routed (per-expert or stacked) experts.
    pub conditional: bool,
    /// `true` when the component is a router or holds a [`TensorRole::MoeRouter`].
    pub is_router: bool,
    /// `true` when any member is a feed-forward gate projection (separate
    /// gate or fused gate-up).
    pub is_gate: bool,
    /// Least-trusted confidence among members.
    pub weakest_confidence: RoleConfidence,
}

/// Per-layer component groups in `(stack, index)` order.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LayerComponents {
    pub stack: String,
    pub index: usize,
    pub components: Vec<Component>,
}

/// Tensor that was not placed into any component.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UnclassifiedTensor {
    pub name: String,
    pub role: TensorRole,
    pub scope: RoleScope,
    pub param: ParamKind,
    pub confidence: RoleConfidence,
    pub alternatives: Vec<TensorRole>,
    pub stack: Option<String>,
    pub layer: Option<usize>,
    pub dims: Vec<usize>,
    pub dtype: String,
}

/// Component inventory derived from a topology.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TransformerComponents {
    pub global: Vec<Component>,
    pub layers: Vec<LayerComponents>,
    /// Unknown and ambiguous tensors, sorted by name.
    pub unclassified: Vec<UnclassifiedTensor>,
}

impl TransformerTopology {
    /// Build the facts-only component view for this topology.
    pub fn components(&self) -> TransformerComponents {
        build_components(self)
    }
}

impl TransformerComponents {
    /// Deterministic JSON rendering (fixed key order, sorted arrays).
    pub fn to_json(&self) -> String {
        let mut out = String::new();
        let mut w = JsonWriter::new(&mut out);
        w.begin_object();

        w.key("global");
        w.begin_array();
        for c in &self.global {
            write_component(&mut w, c);
        }
        w.end_array();

        w.key("layers");
        w.begin_array();
        for layer in &self.layers {
            w.begin_object();
            w.key("stack");
            w.string(&layer.stack);
            w.key("index");
            w.usize(layer.index);
            w.key("components");
            w.begin_array();
            for c in &layer.components {
                write_component(&mut w, c);
            }
            w.end_array();
            w.end_object();
        }
        w.end_array();

        w.key("unclassified");
        w.begin_array();
        for t in &self.unclassified {
            write_unclassified(&mut w, t);
        }
        w.end_array();

        w.end_object();
        out
    }
}

fn build_components(topo: &TransformerTopology) -> TransformerComponents {
    let mut unclassified = Vec::new();
    let global = build_global(&topo.global, &mut unclassified);
    let layers = topo
        .layers
        .iter()
        .map(|layer| build_layer_components(layer, &mut unclassified))
        .collect();
    unclassified.sort_by(|a, b| a.name.cmp(&b.name));
    TransformerComponents {
        global,
        layers,
        unclassified,
    }
}

fn build_global(
    tensors: &[RoleAssignment],
    unclassified: &mut Vec<UnclassifiedTensor>,
) -> Vec<Component> {
    let mut buckets: BTreeMap<ComponentKind, Vec<&RoleAssignment>> = BTreeMap::new();
    for t in tensors {
        if is_unclassified(t) {
            unclassified.push(unclassified_from(t));
            continue;
        }
        if let Some(kind) = global_kind(t.role) {
            buckets.entry(kind).or_default().push(t);
        } else {
            unclassified.push(unclassified_from(t));
        }
    }
    let order = [
        ComponentKind::Embeddings,
        ComponentKind::OutputHead,
        ComponentKind::FinalNorm,
        ComponentKind::RotaryFrequencies,
    ];
    order
        .into_iter()
        .filter_map(|kind| {
            buckets
                .remove(&kind)
                .map(|members| make_component(kind, None, None, None, members))
        })
        .collect()
}

/// Tensors of one layer, bucketed by the component they belong to.
#[derive(Default)]
struct LayerBuckets<'a> {
    attention: Vec<&'a RoleAssignment>,
    norms: Vec<&'a RoleAssignment>,
    mlp: Vec<&'a RoleAssignment>,
    router: Vec<&'a RoleAssignment>,
    stacked: Vec<&'a RoleAssignment>,
    shared: Vec<&'a RoleAssignment>,
    experts: BTreeMap<usize, Vec<&'a RoleAssignment>>,
}

impl<'a> LayerBuckets<'a> {
    /// Place `t` in a bucket; `false` when no component can hold it.
    fn push(&mut self, t: &'a RoleAssignment) -> bool {
        match t.scope {
            RoleScope::Expert(i) => self.experts.entry(i).or_default().push(t),
            RoleScope::StackedExperts => self.stacked.push(t),
            RoleScope::SharedExpert => self.shared.push(t),
            RoleScope::Layer | RoleScope::Global => match layer_bucket(t.role) {
                Some(LayerBucket::Router) => self.router.push(t),
                Some(LayerBucket::Norms) => self.norms.push(t),
                Some(LayerBucket::Attention) => self.attention.push(t),
                Some(LayerBucket::Mlp) => self.mlp.push(t),
                None => return false,
            },
        }
        true
    }

    fn into_components(self, stack: &str, layer: usize) -> Vec<Component> {
        let mut components = Vec::new();
        let mut add = |kind, expert_index, members: Vec<&RoleAssignment>| {
            if !members.is_empty() {
                components.push(make_component(
                    kind,
                    Some(stack.to_owned()),
                    Some(layer),
                    expert_index,
                    members,
                ));
            }
        };
        add(ComponentKind::AttentionBlock, None, self.attention);
        add(ComponentKind::Norms, None, self.norms);
        add(ComponentKind::MlpBlock, None, self.mlp);
        add(ComponentKind::MoeRouter, None, self.router);
        for (expert_index, members) in self.experts {
            add(ComponentKind::RoutedExpert, Some(expert_index), members);
        }
        add(ComponentKind::StackedExperts, None, self.stacked);
        add(ComponentKind::SharedExpert, None, self.shared);
        components
    }
}

#[derive(Clone, Copy)]
enum LayerBucket {
    Attention,
    Norms,
    Mlp,
    Router,
}

/// Bucket for a layer-scoped (non-expert) role.
fn layer_bucket(role: TensorRole) -> Option<LayerBucket> {
    if role == TensorRole::MoeRouter {
        Some(LayerBucket::Router)
    } else if is_layer_norm(role) || role == TensorRole::RotaryFrequencies {
        // Layer-local rotary tables are reported with norms.
        Some(LayerBucket::Norms)
    } else if role.is_attention() {
        Some(LayerBucket::Attention)
    } else if role.is_feed_forward() {
        Some(LayerBucket::Mlp)
    } else {
        None
    }
}

fn build_layer_components(
    layer: &LayerTopology,
    unclassified: &mut Vec<UnclassifiedTensor>,
) -> LayerComponents {
    let mut buckets = LayerBuckets::default();
    for t in &layer.tensors {
        if is_unclassified(t) || !buckets.push(t) {
            unclassified.push(unclassified_from(t));
        }
    }
    LayerComponents {
        components: buckets.into_components(&layer.stack, layer.index),
        stack: layer.stack.clone(),
        index: layer.index,
    }
}

fn global_kind(role: TensorRole) -> Option<ComponentKind> {
    match role {
        TensorRole::TokenEmbedding | TensorRole::PositionEmbedding => {
            Some(ComponentKind::Embeddings)
        }
        TensorRole::OutputHead => Some(ComponentKind::OutputHead),
        TensorRole::OutputNorm => Some(ComponentKind::FinalNorm),
        TensorRole::RotaryFrequencies => Some(ComponentKind::RotaryFrequencies),
        _ => None,
    }
}

fn is_layer_norm(role: TensorRole) -> bool {
    matches!(
        role,
        TensorRole::AttentionNorm
            | TensorRole::PostAttentionNorm
            | TensorRole::MlpNorm
            | TensorRole::PostMlpNorm
    )
}

fn is_unclassified(t: &RoleAssignment) -> bool {
    t.confidence == RoleConfidence::Ambiguous
        || t.confidence == RoleConfidence::Unknown
        || t.role == TensorRole::Unknown
}

fn unclassified_from(t: &RoleAssignment) -> UnclassifiedTensor {
    UnclassifiedTensor {
        name: t.name.clone(),
        role: t.role,
        scope: t.scope,
        param: t.param,
        confidence: t.confidence,
        alternatives: t.alternatives.clone(),
        stack: t.stack.clone(),
        layer: t.layer,
        dims: t.dims.clone(),
        dtype: t.dtype.clone(),
    }
}

fn make_component(
    kind: ComponentKind,
    stack: Option<String>,
    layer: Option<usize>,
    expert_index: Option<usize>,
    mut members: Vec<&RoleAssignment>,
) -> Component {
    members.sort_by(|a, b| a.name.cmp(&b.name));
    let member_records: Vec<ComponentMember> = members
        .iter()
        .map(|t| ComponentMember {
            name: t.name.clone(),
            role: t.role,
            scope: t.scope,
            param: t.param,
            confidence: t.confidence,
            dims: t.dims.clone(),
            dtype: t.dtype.clone(),
        })
        .collect();
    // Saturating: unchecked caller-supplied shapes must not panic or wrap.
    let parameter_count = member_records.iter().fold(0usize, |total, m| {
        let elements = m.dims.iter().fold(1usize, |n, &d| n.saturating_mul(d));
        total.saturating_add(elements)
    });
    // Shared experts are always-on, so they are not conditional.
    let conditional = matches!(
        kind,
        ComponentKind::MoeRouter | ComponentKind::RoutedExpert | ComponentKind::StackedExperts
    ) || member_records
        .iter()
        .any(|m| matches!(m.scope, RoleScope::Expert(_) | RoleScope::StackedExperts));
    let is_router = kind == ComponentKind::MoeRouter
        || member_records
            .iter()
            .any(|m| m.role == TensorRole::MoeRouter);
    let is_gate = member_records
        .iter()
        .any(|m| matches!(m.role, TensorRole::MlpGate | TensorRole::MlpGateUp));
    let weakest_confidence = member_records
        .iter()
        .map(|m| m.confidence)
        .max()
        .unwrap_or(RoleConfidence::Exact);
    Component {
        kind,
        stack,
        layer,
        expert_index,
        members: member_records,
        parameter_count,
        conditional,
        is_router,
        is_gate,
        weakest_confidence,
    }
}

fn write_component(w: &mut JsonWriter<'_>, c: &Component) {
    w.begin_object();
    w.key("kind");
    w.string(c.kind.as_str());
    w.key("stack");
    w.opt_string(c.stack.as_deref());
    w.key("layer");
    w.opt_usize(c.layer);
    w.key("expert_index");
    w.opt_usize(c.expert_index);
    w.key("parameter_count");
    w.usize(c.parameter_count);
    w.key("conditional");
    w.bool(c.conditional);
    w.key("is_router");
    w.bool(c.is_router);
    w.key("is_gate");
    w.bool(c.is_gate);
    w.key("weakest_confidence");
    w.string(c.weakest_confidence.as_str());
    w.key("members");
    w.begin_array();
    for m in &c.members {
        w.begin_object();
        w.key("name");
        w.string(&m.name);
        w.key("role");
        w.string(m.role.as_str());
        w.key("scope");
        w.string(&m.scope.to_string());
        w.key("param");
        w.string(m.param.as_str());
        w.key("confidence");
        w.string(m.confidence.as_str());
        w.key("dims");
        w.usize_array(&m.dims);
        w.key("dtype");
        w.string(&m.dtype);
        w.end_object();
    }
    w.end_array();
    w.end_object();
}

fn write_unclassified(w: &mut JsonWriter<'_>, t: &UnclassifiedTensor) {
    w.begin_object();
    w.key("name");
    w.string(&t.name);
    w.key("role");
    w.string(t.role.as_str());
    w.key("scope");
    w.string(&t.scope.to_string());
    w.key("param");
    w.string(t.param.as_str());
    w.key("confidence");
    w.string(t.confidence.as_str());
    w.key("alternatives");
    w.string_array(t.alternatives.iter().map(|r| r.as_str()));
    w.key("stack");
    w.opt_string(t.stack.as_deref());
    w.key("layer");
    w.opt_usize(t.layer);
    w.key("dims");
    w.usize_array(&t.dims);
    w.key("dtype");
    w.string(&t.dtype);
    w.end_object();
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analysis::transformer::TensorRole;
    use crate::analysis::transformer::discover_topology_from_parts;
    use crate::checkpoint::{
        CheckpointMetadata, TensorDType, TensorInfo, TensorLocation, TensorShape,
    };

    fn mock(name: &str, dims: &[usize]) -> TensorInfo {
        TensorInfo::new(
            name,
            TensorDType::F32,
            TensorShape::outermost_first(dims.to_vec()),
            dims.iter().product::<usize>() * 4,
            TensorLocation::new("model", 0, None),
        )
    }

    #[test]
    fn dense_layer_groups_attention_mlp_and_norms() {
        let inv = vec![
            mock("blk.0.attn_norm.weight", &[8]),
            mock("blk.0.attn_q.weight", &[8, 8]),
            mock("blk.0.ffn_norm.weight", &[8]),
            mock("blk.0.ffn_gate.weight", &[8, 16]),
            mock("blk.0.ffn_up.weight", &[8, 16]),
            mock("blk.0.ffn_down.weight", &[16, 8]),
        ];
        let topo = discover_topology_from_parts(&inv, &CheckpointMetadata::new());
        let view = topo.components();
        let layer = &view.layers[0];
        assert_eq!(layer.components.len(), 3);
        assert_eq!(layer.components[0].kind, ComponentKind::AttentionBlock);
        assert_eq!(layer.components[1].kind, ComponentKind::Norms);
        assert_eq!(layer.components[2].kind, ComponentKind::MlpBlock);
        assert!(layer.components[2].is_gate);
        assert!(!layer.components[2].conditional);
    }

    #[test]
    fn unknown_tensors_stay_unclassified() {
        let inv = vec![
            mock("blk.0.attn_q.weight", &[8, 8]),
            mock("blk.0.mystery.weight", &[3]),
        ];
        let topo = discover_topology_from_parts(&inv, &CheckpointMetadata::new());
        let view = topo.components();
        assert_eq!(view.unclassified.len(), 1);
        assert_eq!(view.unclassified[0].name, "blk.0.mystery.weight");
        assert_eq!(view.unclassified[0].confidence, RoleConfidence::Unknown);
    }

    #[test]
    fn moe_groups_router_stacked_and_shared_experts() {
        let inv = vec![
            mock("blk.0.ffn_gate_inp.weight", &[8, 4]),
            mock("blk.0.ffn_gate_exps.weight", &[4, 16, 8]),
            mock("blk.0.ffn_up_exps.weight", &[4, 16, 8]),
            mock("blk.0.ffn_down_exps.weight", &[16, 8, 4]),
            mock("blk.0.ffn_up_shexp.weight", &[8, 16]),
        ];
        let topo = discover_topology_from_parts(&inv, &CheckpointMetadata::new());
        let view = topo.components();
        let kinds: Vec<_> = view.layers[0].components.iter().map(|c| c.kind).collect();
        assert_eq!(
            kinds,
            [
                ComponentKind::MoeRouter,
                ComponentKind::StackedExperts,
                ComponentKind::SharedExpert,
            ]
        );
        assert!(view.layers[0].components[0].is_router);
        assert!(view.layers[0].components[1].conditional);
        assert!(!view.layers[0].components[2].conditional);
    }

    #[test]
    fn fused_gate_up_reports_is_gate() {
        let inv = vec![mock("blk.0.ffn_gate_up.weight", &[8, 32])];
        let topo = discover_topology_from_parts(&inv, &CheckpointMetadata::new());
        let view = topo.components();
        let mlp = &view.layers[0].components[0];
        assert_eq!(mlp.kind, ComponentKind::MlpBlock);
        assert_eq!(mlp.members[0].role, TensorRole::MlpGateUp);
        assert!(mlp.is_gate);
    }

    #[test]
    fn parameter_count_saturates_instead_of_overflowing() {
        let huge = 1usize << 40;
        let inv = vec![
            TensorInfo::new(
                "blk.0.attn_q.weight",
                TensorDType::F32,
                TensorShape::outermost_first(vec![huge, huge, 0]),
                0,
                TensorLocation::new("model", 0, None),
            ),
            TensorInfo::new(
                "blk.0.attn_k.weight",
                TensorDType::F32,
                TensorShape::outermost_first(vec![huge, huge]),
                0,
                TensorLocation::new("model", 0, None),
            ),
        ];
        let topo = discover_topology_from_parts(&inv, &CheckpointMetadata::new());
        let view = topo.components();
        assert_eq!(view.layers[0].components[0].parameter_count, usize::MAX);
    }

    #[test]
    fn per_expert_components_are_split_by_index() {
        let inv = vec![
            mock("blk.0.ffn_gate_inp.weight", &[3, 8]),
            mock("blk.0.ffn_gate.0.weight", &[16, 8]),
            mock("blk.0.ffn_up_0.weight", &[16, 8]),
            mock("blk.0.ffn_down.1.weight", &[8, 16]),
            mock("blk.0.ffn_up.1.weight", &[16, 8]),
            mock("blk.0.ffn_gate_2.weight", &[16, 8]),
        ];
        let topo = discover_topology_from_parts(&inv, &CheckpointMetadata::new());
        let view = topo.components();
        let experts: Vec<_> = view.layers[0]
            .components
            .iter()
            .filter(|c| c.kind == ComponentKind::RoutedExpert)
            .map(|c| c.expert_index)
            .collect();
        assert_eq!(experts, vec![Some(0), Some(1), Some(2)]);
    }

    #[test]
    fn ambiguous_gate_is_unclassified() {
        let inv = vec![
            mock("model.layers.0.mlp.gate.weight", &[16, 8]),
            mock("model.layers.0.mlp.up_proj.weight", &[16, 8]),
        ];
        let topo = discover_topology_from_parts(&inv, &CheckpointMetadata::new());
        let view = topo.components();
        assert_eq!(view.unclassified.len(), 1);
        assert_eq!(view.unclassified[0].name, "model.layers.0.mlp.gate.weight");
        assert_eq!(view.unclassified[0].role, TensorRole::Unknown);
        assert_eq!(view.unclassified[0].confidence, RoleConfidence::Ambiguous);
    }
}
